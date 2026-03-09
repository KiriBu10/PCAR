#### cross_attention_method.py ####

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import math
import time
import pickle
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Utils from your prior script
# ---------------------------------------------------------------------
from framework.utils import compute_mae as _compute_mae_utils, find_parallel_sequences, attach_parallel_sequences_fast

def compute_mae(y_true, y_pred):
    # y_true, y_pred: torch tensors on CPU
    return torch.mean(torch.abs(y_true - y_pred)).item()

def get_max_activity_id(df):
    max_id = 0
    for prefix in df['prefix_int']:
        if prefix:
            max_id = max(max_id, max(prefix))
    for parallels in df['parallel_sequences']:
        for p in parallels:
            if p['activity_sequence_int']:
                max_id = max(max_id, max(p['activity_sequence_int']))
    return max_id

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#set_seed(42)

# ---------------------------------------------------------------------
# Time-aware cross-execution model (A-only prediction)
# ---------------------------------------------------------------------
class TimeAwareCrossExecutionTransformer(nn.Module):
    """
    A-only prediction with time-aware cross-execution attention.
    - Per-sequence encoder: TransformerEncoder (token-level outputs)
    - Cross-attention: queries from A1 last token; keys/values from all (A1 + parallels)
    - Additive logit terms: time-causal mask, time-decay, type-graph bias, same-exec bonus
    """
    def __init__(self, vocab_size, num_types, d_model=64, nhead=4, ff_dim=128,
                 dropout=0.2, max_seq_len=50):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.time_feat_dim = 1 + 2*4 + 1  # t_norm + 4 sin/cos pairs + log_dt
        self.time_proj = nn.Linear(self.time_feat_dim, d_model)
        self.time_ln = nn.LayerNorm(d_model)
        
        self.d_model = d_model
        self.nhead = nhead
        self.max_seq_len = max_seq_len

        # Token embedding (activities); ID 0 is PAD
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Process type embedding (no pad idx; types are >0; pads are masked anyway)
        self.type_emb = nn.Embedding(num_types, d_model)

        # Shared encoder for any single execution (returns token-level states)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # --- Cross-attention projections (multi-head) ---
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_dropout_p = dropout

        # Time decay scale tau > 0 (use softplus to keep positive)
        self._tau = nn.Parameter(torch.tensor(1.0))
        # Type graph bias: [num_types x num_types] additive logits prior
        self.type_graph = nn.Parameter(torch.zeros(num_types, num_types))
        # Same-execution bonus (scalar additive logit)
        self.same_exec_bias = nn.Parameter(torch.tensor(0.1))

        # Prediction head (fuse main last token & cross-attended context)
        self.fuse = nn.LayerNorm(d_model * 2)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    # -------- helpers --------
    def _encode_one_sequence(self, x_tok, seq_pad_mask, type_id, t=None):
        B, T = x_tok.shape
        tok = self.token_emb(x_tok)                                   # (B,T,d)
        if type_id is not None:
            tok = tok + self.type_emb(type_id).unsqueeze(1)           # (B,T,d)
        if t is not None:
            t_feats = self._build_time_features(t, seq_pad_mask)      # (B,T,F)
            t_emb = self.time_proj(t_feats)                           # (B,T,d)
            tok = self.time_ln(tok + t_emb)

        h = self.encoder(tok, src_key_padding_mask=seq_pad_mask)      # (B,T,d)
        lengths = (~seq_pad_mask).sum(dim=1) - 1
        lengths = torch.clamp(lengths, min=0)
        return h, lengths


    # def _mha(self, q, k, v, attn_additive):
    #     """
    #     Simple multi-head attention using explicit projections & softmax.
    #     q: (B, Q, d), k: (B, K, d), v: (B, K, d)
    #     attn_additive: (B, h, Q, K) additive to logits (use -inf on disallowed)
    #     """
    #     B, Q, d = q.shape
    #     _, K, _ = k.shape
    #     h = self.nhead
    #     d_h = d // h

    #     # Projections & split heads
    #     qh = self.q_proj(q).view(B, Q, h, d_h).transpose(1, 2)  # (B,h,Q,d_h)
    #     kh = self.k_proj(k).view(B, K, h, d_h).transpose(1, 2)  # (B,h,K,d_h)
    #     vh = self.v_proj(v).view(B, K, h, d_h).transpose(1, 2)  # (B,h,K,d_h)

    #     # Scaled dot-product with additive mask/bias
    #     scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(d_h)  # (B,h,Q,K)
    #     if attn_additive is not None:
    #         scores = scores + attn_additive  # add -inf on disallowed, add real-valued biases

    #     attn = torch.softmax(scores, dim=-1)                             # (B,h,Q,K)
    #     if self.training and self.attn_dropout_p > 0:
    #         attn = F.dropout(attn, p=self.attn_dropout_p)

    #     out = torch.matmul(attn, vh)                                     # (B,h,Q,d_h)
    #     out = out.transpose(1, 2).contiguous().view(B, Q, d)             # (B,Q,d)
    #     return self.o_proj(out)                                          # (B,Q,d)
    
    def _mha(self, q, k, v, attn_additive):
        """
        q: (B, Q, d), k: (B, K, d), v: (B, K, d)
        attn_additive: (B, 1, K) or (B, h, Q, K) additive logits
        """
        # merge heads implicitly via SDPA: give it (B, nH, Q, dH) etc.
        B, Q, d = q.shape
        h = self.nhead
        d_h = d // h

        qh = self.q_proj(q).view(B, Q, h, d_h).transpose(1, 2)  # (B,h,Q,d_h)
        kh = self.k_proj(k).view(B, -1, h, d_h).transpose(1, 2) # (B,h,K,d_h)
        vh = self.v_proj(v).view(B, -1, h, d_h).transpose(1, 2) # (B,h,K,d_h)

        # SDPA expects (B*h, Q, dH), (B*h, K, dH)
        qh_ = qh.reshape(B*h, Q, d_h)
        kh_ = kh.reshape(B*h, -1, d_h)
        vh_ = vh.reshape(B*h, -1, d_h)

        # Build additive bias per head
        if attn_additive is not None:
            # accept (B,h,Q,K) or broadcast (B,1,K)
            if attn_additive.dim() == 3:  # (B,1,K)
                attn_bias = attn_additive.expand(B, h, Q, -1)
            else:
                attn_bias = attn_additive
            attn_bias = attn_bias.reshape(B*h, Q, -1)
        else:
            attn_bias = None

        out = F.scaled_dot_product_attention(
            qh_, kh_, vh_,
            attn_mask=attn_bias,      # additive mask/bias
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False
        )  # (B*h, Q, d_h)

        out = out.reshape(B, h, Q, d_h).transpose(1, 2).contiguous().view(B, Q, d)
        return self.o_proj(out)
    
    def _build_time_features(self, t, pad_mask, num_freqs=4, eps=1e-6):
        """
        t: (B,T) float timestamps (same unit across executions)
        pad_mask: (B,T) bool (True on PAD)
        returns: (B,T, F)
        """
        B, T = t.shape
        # Normalize time within each sequence to [0,1]
        t_min = t.masked_fill(pad_mask, float('inf')).amin(dim=1, keepdim=True)
        t_min = torch.where(torch.isfinite(t_min), t_min, torch.zeros_like(t_min))
        t_max = t.masked_fill(pad_mask, float('-inf')).amax(dim=1, keepdim=True)
        t_max = torch.where(torch.isfinite(t_max), t_max, t_min + 1.0)
        t_norm = (t - t_min) / (t_max - t_min + eps)  # (B,T)

        # Time deltas (prev gap); 0 at first or pad transitions
        t_shift = torch.roll(t, 1, dims=1)
        same_exec_step = (~pad_mask) & torch.roll((~pad_mask), 1, dims=1)
        dt = torch.where(same_exec_step, t - t_shift, torch.zeros_like(t))
        #log_dt = torch.log1p(dt)  # (B,T)
        # Time deltas (prev gap); 0 at first or pad transitions
        t_shift = torch.roll(t, 1, dims=1)
        same_exec_step = (~pad_mask) & torch.roll((~pad_mask), 1, dims=1)
        dt = torch.where(same_exec_step, t - t_shift, torch.zeros_like(t))

        # *** FIX: clamp to avoid log1p of negative values ***
        log_dt = torch.log1p(torch.clamp(dt, min=0.0))

        # Fourier features on normalized time
        freqs = t_norm.unsqueeze(-1) * (2*math.pi) * torch.tensor([1,2,4,8], device=t.device)  # (B,T,4)
        sinf = torch.sin(freqs); cosf = torch.cos(freqs)                                       # (B,T,4)

        feats = torch.cat([t_norm.unsqueeze(-1), sinf, cosf, log_dt.unsqueeze(-1)], dim=-1)    # (B,T, 1+4+4+1)
        feats = feats.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return feats

    # -------- forward --------
    def forward(self,
                x_main, t_main, main_type,          # (B,T), (B,T), (B,)
                x_par,  t_par,  par_mask, par_types # (B,N,T), (B,N,T), (B,N), (B,N)
                ):
        """
        x_main:   activities for main execution (A1), long
        t_main:   timestamps for main (float)
        main_type:(B,) long, process type id of main
        x_par:    activities for N parallel execs, long
        t_par:    timestamps for N parallel execs, float
        par_mask: (B,N) bool, 1=present execution, 0=PAD execution
        par_types:(B,N) long type id per parallel execution
        """
        device = x_main.device
        B, T = x_main.shape
        _, N, Tp = x_par.shape

        # --- Encode main (token-level) ---
        main_pad = (x_main == 0)                         # (B,T) True on PAD
        h_main, last_idx_main = self._encode_one_sequence(x_main, main_pad, main_type, t_main)  # (B,T,d), (B,)

        # Query: last token of main (A1)
        q = h_main[torch.arange(B, device=device), last_idx_main]  # (B,d)
        q = q.unsqueeze(1)                                         # (B,1,d)
        t_q = t_main[torch.arange(B, device=device), last_idx_main]  # (B,)

        # --- Encode parallels (flattened) ---
        x_par_flat = x_par.view(B*N, Tp)
        t_par_flat = t_par.view(B*N, Tp)
        types_par_flat = par_types.view(B*N)            # (B*N,)
        exec_present = par_mask.view(B*N)               # (B*N,) bool

        par_pad_flat = (x_par_flat == 0)                # (B*N,Tp) token pads
        # If execution is PAD, mark all tokens as PAD
        exec_pad = (~exec_present).unsqueeze(1).expand_as(par_pad_flat)
        par_pad_flat = par_pad_flat | exec_pad

        # Encode each execution with its type embedding
        h_par_flat, _ = self._encode_one_sequence(x_par_flat, par_pad_flat, types_par_flat, t_par_flat)
        h_par = h_par_flat.view(B, N, Tp, self.d_model)    # (B,N,Tp,d)
        par_pad = par_pad_flat.view(B, N, Tp)              # (B,N,Tp)

        # --- Build K/V sets: main history + all parallels ---
        k_main, v_main = h_main, h_main                               # (B,T,d)
        k_par  = h_par.view(B, N*Tp, self.d_model)                     # (B,N*Tp,d)
        v_par  = k_par
        k = torch.cat([k_main, k_par], dim=1)                          # (B, K, d)
        v = torch.cat([v_main, v_par], dim=1)

        # Per-key timestamps
        t_k_main = t_main                                              # (B,T)
        t_k_par  = t_par.view(B, N*Tp)                                 # (B,N*Tp)
        t_k = torch.cat([t_k_main, t_k_par], dim=1)                    # (B,K)

        # Key padding (True on PAD)
        key_pad_main = main_pad                                        # (B,T)
        key_pad_par  = par_pad.view(B, N*Tp)                           # (B,N*Tp)
        key_pad = torch.cat([key_pad_main, key_pad_par], dim=1)        # (B,K)

        # Per-key type ids
        type_k_main = main_type.unsqueeze(1).expand(B, T)              # (B,T)
        type_k_par  = par_types.view(B, N, 1).expand(B, N, Tp).reshape(B, N*Tp)
        type_k = torch.cat([type_k_main, type_k_par], dim=1)           # (B,K)

        # Same-execution flag for keys (True if key from main execution)
        same_exec_k = torch.zeros_like(key_pad, dtype=torch.bool)
        same_exec_k[:, :T] = True

        
        # Ensure we have at least one allowed key (avoid softmax over all -inf)
        # Allowed keys = not padded and not future
        # We'll compute this with current t_k, key_pad, and the query time t_q
        allowed = (~key_pad) & (t_k <= t_q.unsqueeze(1))  # (B, K)
        no_keys = (allowed.sum(dim=1) == 0)               # (B,)

        if no_keys.any():
            # Force the main "self" key (last valid token of main) to be available
            self_idx = last_idx_main.clone()
            bidx = torch.arange(B, device=x_main.device)
            # Unpad the self key
            key_pad[no_keys, self_idx[no_keys]] = False
            # Make its time equal to the query time so it's not "future"
            t_k[no_keys, self_idx[no_keys]] = t_q[no_keys]
            # (same_exec_k is already True for main keys)

        # --- Build additive attention mask/bias (B,h,Q,K); here Q=1 ---
        # 1) time causality: forbid t_k > t_q
        dt = t_k.unsqueeze(1) - t_q.unsqueeze(1).unsqueeze(2)          # (B,1,K)
        future_mask = torch.where(dt > 0,
                                  torch.full_like(dt, float('-inf')),
                                  torch.zeros_like(dt))                # (B,1,K)

        # 2) time-decay bias
        tau = F.softplus(self._tau) + 1e-6
        time_bias = -torch.abs(dt) / tau                                # (B,1,K)

        # 3) type graph bias: G[type_q, type_k]
        type_q = main_type.unsqueeze(1)                                 # (B,1)
        G = self.type_graph                                             # (num_types,num_types)
        #type_bias = G[type_q, type_k]                                   # (B,1,K)

        # 3) type graph bias: G[type_q, type_k]  (batched)
        # type_q: (B,1)  type_k: (B,K)  G: (num_types, num_types)
        rows = G.index_select(0, type_q.squeeze(1))     # (B, num_types)
        type_bias = rows.gather(1, type_k.clamp_min(0)) # (B, K)
        type_bias = type_bias.unsqueeze(1)              # (B,1,K)


        # 4) same-execution bonus
        same_bias = torch.where(same_exec_k.unsqueeze(1),
                                self.same_exec_bias,
                                torch.tensor(0.0, device=device))      # (B,1,K)

        # 5) padding mask -> -inf on pads
        pad_mask = torch.where(key_pad.unsqueeze(1),
                               torch.full_like(time_bias, float('-inf')),
                               torch.zeros_like(time_bias))             # (B,1,K)

        additive = future_mask + pad_mask #+ time_bias + type_bias + same_bias  # (B,1,K)
        additive = additive.expand(-1, self.nhead, -1)                          # (B,h,K)
        additive = additive.unsqueeze(2)                                        # (B,h,1,K)

        # --- Cross-attend (A1-only query over all keys) ---
        out = self._mha(q, k, v, additive)        # (B,1,d)
        out = out.squeeze(1)                      # (B,d)

        # --- Fuse with main last token and predict ---
        main_last = q.squeeze(1)                  # (B,d)
        fused = torch.cat([main_last, out], dim=-1)    # (B,2d)
        fused = self.fuse(fused)
        y_hat = self.head(fused).squeeze(-1)      # (B,)
        return y_hat

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class FastLUPIDataset(Dataset):
    """
    Returns:
      x_main:   (T,) long activity ids
      t_main:   (T,) float timestamps (same units across df)
      main_type:(,)  long process type id
      x_par:    (N,T) long activity ids for parallels
      t_par:    (N,T) float timestamps for parallels
      par_mask: (N,)  bool (1 if execution present)
      par_types:(N,)  long type id per parallel execution
      y:        scalar float (remaining_time)
    """
    def __init__(self, df, max_seq_len, max_parallel=10, process_to_int=None):
        self.df = df
        self.max_seq_len = max_seq_len
        self.max_parallel = max_parallel
        self.process_to_int = process_to_int or {}

    @staticmethod
    def _pad(seq, max_len, pad_value):
        seq = list(seq) if isinstance(seq, (list, tuple)) else [pad_value]
        return seq[-max_len:] + [pad_value] * max(0, max_len - len(seq))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Main execution (A1)
        x_main = torch.tensor(self._pad(row['prefix_int'], self.max_seq_len, 0), dtype=torch.long)
        t_main = torch.tensor(self._pad(row['list_timestamps_case'], self.max_seq_len, 0.0), dtype=torch.float32)
        y = torch.tensor(row['remaining_time'], dtype=torch.float32)

        # Process type for main
        if 'process_int' in row:
            main_type = int(row['process_int'])
        else:
            main_type = int(self.process_to_int.get(row['process'], 0))
        main_type = torch.tensor(main_type, dtype=torch.long)

        # Parallels (variable length set)
        parallel_x, parallel_t, par_mask, par_types = [], [], [], []
        for p in row['parallel_sequences'][:self.max_parallel]:
            parallel_x.append(self._pad(p.get('activity_sequence_int', []), self.max_seq_len, 0))
            parallel_t.append(self._pad(p.get('timestamps_case_related', []), self.max_seq_len, 0.0))
            par_mask.append(1)
            # type id
            if 'process_int' in p:
                par_types.append(int(p['process_int']))
            else:
                par_types.append(int(self.process_to_int.get(p.get('process', row['process']), main_type.item())))

        # Pad up to max_parallel
        while len(parallel_x) < self.max_parallel:
            parallel_x.append([0] * self.max_seq_len)
            parallel_t.append([0.0] * self.max_seq_len)
            par_mask.append(0)
            par_types.append(main_type.item())  # filler; will be masked

        return (
            x_main,
            t_main,
            main_type,
            torch.tensor(parallel_x, dtype=torch.long),
            torch.tensor(parallel_t, dtype=torch.float32),
            torch.tensor(par_mask, dtype=torch.bool),
            torch.tensor(par_types, dtype=torch.long),
            y
        )

# ---------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------
def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=100, patience=10):
    best_val = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_main, t_main, main_type, x_par, t_par, par_mask, par_types, y = batch
            x_main = x_main.to(device, non_blocking=True)
            t_main = t_main.to(device, non_blocking=True)
            main_type = main_type.to(device, non_blocking=True)
            x_par = x_par.to(device, non_blocking=True)
            t_par = t_par.to(device, non_blocking=True)
            par_mask = par_mask.to(device, non_blocking=True)
            par_types = par_types.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(x_main, t_main, main_type, x_par, t_par, par_mask, par_types)
            loss = loss_fn(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_main, t_main, main_type, x_par, t_par, par_mask, par_types, y = batch
                x_main = x_main.to(device, non_blocking=True)
                t_main = t_main.to(device, non_blocking=True)
                main_type = main_type.to(device, non_blocking=True)
                x_par = x_par.to(device, non_blocking=True)
                t_par = t_par.to(device, non_blocking=True)
                par_mask = par_mask.to(device, non_blocking=True)
                par_types = par_types.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                preds = model(x_main, t_main, main_type, x_par, t_par, par_mask, par_types)
                val_loss += loss_fn(preds, y).item()

        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in data_loader:
            x_main, t_main, main_type, x_par, t_par, par_mask, par_types, y = batch
            x_main = x_main.to(device)
            t_main = t_main.to(device)
            main_type = main_type.to(device)
            x_par = x_par.to(device)
            t_par = t_par.to(device)
            par_mask = par_mask.to(device)
            par_types = par_types.to(device)
            y = y.to(device)

            preds = model(x_main, t_main, main_type, x_par, t_par, par_mask, par_types)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    mae = compute_mae(targets_cat, preds_cat)
    return mae

# ---------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------
def run_experiments(n_runs,
                    train_df_parallel_sequences,
                    val_df_parallel_sequences,
                    test_df_parallel_sequences,
                    process_to_int,
                    save_path_prefix="models/timeaware_model_run.pt",
                    d_model=36, nhead=4, ff_dim=64, dropout=0.2,
                    max_seq_len=50, max_parallel=10,
                    batch_size=128, num_epochs=200, patience=10):
    results = []
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda") 

    # Vocab size from activities across main & parallels
    vocab_size = get_max_activity_id(train_df_parallel_sequences) + 1
    num_types = len(process_to_int) + 1  # +1 in case 0 is unused/pad

    # Datasets / loaders
    train_dataset = FastLUPIDataset(train_df_parallel_sequences, max_seq_len, max_parallel, process_to_int)
    val_dataset   = FastLUPIDataset(val_df_parallel_sequences,   max_seq_len, max_parallel, process_to_int)
    test_dataset  = FastLUPIDataset(test_df_parallel_sequences,  max_seq_len, max_parallel, process_to_int)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    for run in range(1, n_runs + 1):
        print(f"\n=== Run {run} ===")
        model = TimeAwareCrossExecutionTransformer(
            vocab_size=vocab_size,
            num_types=num_types,
            d_model=d_model, nhead=nhead, ff_dim=ff_dim,
            dropout=dropout, max_seq_len=max_seq_len
        ).to(device)
        assert next(model.parameters()).is_cuda, "Model did not move to CUDA"

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.L1Loss()

        start = time.time()
        model = train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs, patience)
        training_time = time.time() - start

        # Save / reload
        torch.save(model.state_dict(), save_path_prefix)
        model.load_state_dict(torch.load(save_path_prefix, map_location=device))

        test_mae = evaluate_model(model, test_loader, device)
        print(f"Test Remaining Time MAE: {test_mae:.4f}")

        results.append({
            "run": run,
            "training_time_sec": training_time,
            "test_mae": test_mae
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------------
# Main: scenarios loop (matches your previous workflow)
# ---------------------------------------------------------------------

# scenarios = ['scenario_1_A',
#              'scenario_1_B_75_unique',
#              'scenario_1_B_40_unique',
#              'scenario_1_B_20_unique',
#              'scenario_1_B']
# #scenarios = ['scenario_1_A']
# gen_available = ['scenario_1_B_75_unique',
#                  'scenario_1_B_40_unique',
#                  'scenario_1_B_20_unique',
#                  'scenario_1_B']
# #gen_available = []

# for scenario in scenarios:
#     print("\n==============================")
#     print(f"Scenario: {scenario}")
#     print("==============================")

#     file_name = f"dataset/RLRAM_l0.1_s00_{scenario}.csv"

#     # Load mappings
#     int_to_activity = pickle.load(open(f"dataset/{scenario}_int_to_activity.p", "rb"))
#     activity_to_int = pickle.load(open(f"dataset/{scenario}_activity_to_int.p", "rb"))
#     int_to_process  = pickle.load(open(f"dataset/{scenario}_int_to_process.p", "rb"))
#     process_to_int  = pickle.load(open(f"dataset/{scenario}_process_to_int.p", "rb"))

#     # Load train/val/test prefix splits
#     train_df = pd.read_pickle(f"dataset/{scenario}_train_prefix.pkl")
#     val_df   = pd.read_pickle(f"dataset/{scenario}_val_prefix.pkl")
#     test_df  = pd.read_pickle(f"dataset/{scenario}_test_prefix.pkl")

#     # (Optional) next_activity label encoding as in your code; not used for RT prediction
#     le = LabelEncoder()
#     all_labels = pd.concat([train_df['next_activity_int'], val_df['next_activity_int'], test_df['next_activity_int']])
#     le.fit(all_labels)
#     train_df['next_activity_int'] = le.transform(train_df['next_activity_int'])
#     val_df['next_activity_int']   = le.transform(val_df['next_activity_int'])
#     test_df['next_activity_int']  = le.transform(test_df['next_activity_int'])

#     # Build parallel sequences for each split using your helper
#     df_all = pd.read_csv(file_name, index_col='Unnamed: 0')
#     df_all = df_all[df_all['status']!='gateway']  # as in your calls

#     train_df_parallel_sequences = find_parallel_sequences(
#         df_a=train_df, df_b=df_all, considered_timeframe=100.0,
#         activity_to_int=activity_to_int, process_to_int=process_to_int
#     )
#     val_df_parallel_sequences = find_parallel_sequences(
#         df_a=val_df, df_b=df_all, considered_timeframe=100.0,
#         activity_to_int=activity_to_int, process_to_int=process_to_int
#     )
#     test_df_parallel_sequences = find_parallel_sequences(
#         df_a=test_df, df_b=df_all, considered_timeframe=100.0,
#         activity_to_int=activity_to_int, process_to_int=process_to_int
#     )
#     def _drop_empty(df_ps):
#         keep = df_ps['prefix_int'].map(lambda x: isinstance(x, (list, tuple)) and len(x) > 0)
#         return df_ps[keep].reset_index(drop=True)
#     train_df_parallel_sequences = _drop_empty(train_df_parallel_sequences)
#     val_df_parallel_sequences   = _drop_empty(val_df_parallel_sequences)
#     test_df_parallel_sequences  = _drop_empty(test_df_parallel_sequences)

#     # ---- Run experiments (A-only, time-aware cross-exec attention) ----
#     results = run_experiments(
#         n_runs=1,
#         train_df_parallel_sequences=train_df_parallel_sequences,
#         val_df_parallel_sequences=val_df_parallel_sequences,
#         test_df_parallel_sequences=test_df_parallel_sequences,
#         process_to_int=process_to_int,
#         save_path_prefix=f"models/timeaware_{scenario}.pt",
#         d_model=36, nhead=4, ff_dim=64, dropout=0.2,
#         max_seq_len=50, max_parallel=10,
#         batch_size=128, num_epochs=200, patience=10
#     )
#     print(results)
#     results.to_csv(f"results/TIMEAWARE_{scenario}.csv", index=False)

#     # ---- Optional: apply on GEN dataset if available ----
#     if scenario in gen_available:
#         train_df_gen = pd.read_pickle(f"dataset/{scenario}_train_prefix_gen.pkl")
#         val_df_gen   = pd.read_pickle(f"dataset/{scenario}_val_prefix_gen.pkl")
#         test_df_gen  = pd.read_pickle(f"dataset/{scenario}_test_prefix_gen.pkl")

#         train_df_gen['next_activity_int'] = le.transform(train_df_gen['next_activity_int'])
#         val_df_gen['next_activity_int']   = le.transform(val_df_gen['next_activity_int'])
#         test_df_gen['next_activity_int']  = le.transform(test_df_gen['next_activity_int'])

#         train_df_gen_parallel_sequences = find_parallel_sequences(
#             df_a=train_df_gen, df_b=df_all, considered_timeframe=100.0,
#             activity_to_int=activity_to_int, process_to_int=process_to_int
#         )
#         val_df_gen_parallel_sequences = find_parallel_sequences(
#             df_a=val_df_gen, df_b=df_all, considered_timeframe=100.0,
#             activity_to_int=activity_to_int, process_to_int=process_to_int
#         )
#         test_df_gen_parallel_sequences = find_parallel_sequences(
#             df_a=test_df_gen, df_b=df_all, considered_timeframe=100.0,
#             activity_to_int=activity_to_int, process_to_int=process_to_int
#         )

#         results_gen = run_experiments(
#             n_runs=1,
#             train_df_parallel_sequences=train_df_gen_parallel_sequences,
#             val_df_parallel_sequences=val_df_gen_parallel_sequences,
#             test_df_parallel_sequences=test_df_gen_parallel_sequences,
#             process_to_int=process_to_int,
#             save_path_prefix=f"models/timeaware_{scenario}_GEN.pt",
#             d_model=36, nhead=4, ff_dim=64, dropout=0.2,
#             max_seq_len=50, max_parallel=10,
#             batch_size=128, num_epochs=200, patience=10
#         )
#         print(results_gen)
#         results_gen.to_csv(f"results/TIMEAWARE_{scenario}_GEN.csv", index=False)

# -----------------------------
# parallel_timeaware_training.py  (training-only snippet)
# -----------------------------
import os
import pickle
import pandas as pd
import multiprocessing as mp
import torch
from sklearn.preprocessing import LabelEncoder
from framework.utils import find_parallel_sequences  # already used in your code

# ---- config ----
scenarios = [
    #  "scenario_1_A",
    # "scenario_1_B_75_unique",
    "scenario_1_B_40_unique",
    # "scenario_1_B_20_unique",
    # "scenario_1_B",
    'bpi2020_2processes_massive_share'
]
gen_available = {
    # "scenario_1_A",
    # "scenario_1_B_75_unique",
    "scenario_1_B_40_unique",
    # "scenario_1_B_20_unique",
    # "scenario_1_B",
    'bpi2020_2processes_massive_share'
}

BASE_PATH = "dataset"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---- tiny helper ----
def _drop_empty(df_ps: pd.DataFrame) -> pd.DataFrame:
    keep = df_ps["prefix_int"].map(lambda x: isinstance(x, (list, tuple)) and len(x) > 0)
    return df_ps[keep].reset_index(drop=True)

def _train_timeaware_task(scenario: str, is_gen: bool, task_idx: int, num_gpus: int) -> dict:
    """
    Runs one (scenario, split) task inside a worker process.
    Assigns a GPU (if available) based on task_idx % num_gpus.
    """
    label = f"{scenario}{' (GEN)' if is_gen else ''}"
    print(f"\n=== START {label} ===")

    # ---- GPU assignment (optional) ----
    if num_gpus > 0:
        dev_id = task_idx % num_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_id)
        try:
            torch.cuda.set_device(0)  # now 0 within this process' visible device
        except Exception:
            pass
        print(f"[{label}] -> GPU {dev_id}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU
        print(f"[{label}] -> CPU")

    # ---- file paths ----
    log_csv = os.path.join(BASE_PATH, f"RLRAM_l0.5_s00_{scenario}.csv")
    if not os.path.exists(log_csv):
        print(f"[SKIP] Missing {log_csv}")
        return {"scenario": scenario, "split": "GEN" if is_gen else "BASE", "status": "missing_log"}

    # mappings
    paths_needed = [
        os.path.join(BASE_PATH, f"{scenario}_int_to_activity.p"),
        os.path.join(BASE_PATH, f"{scenario}_activity_to_int.p"),
        os.path.join(BASE_PATH, f"{scenario}_int_to_process.p"),
        os.path.join(BASE_PATH, f"{scenario}_process_to_int.p"),
    ]
    if not all(os.path.exists(p) for p in paths_needed):
        miss = [os.path.basename(p) for p in paths_needed if not os.path.exists(p)]
        print(f"[SKIP] Missing mappings for {label}: {miss}")
        return {"scenario": scenario, "split": "GEN" if is_gen else "BASE", "status": "missing_maps"}

    int_to_activity = pickle.load(open(paths_needed[0], "rb"))
    activity_to_int = pickle.load(open(paths_needed[1], "rb"))
    int_to_process  = pickle.load(open(paths_needed[2], "rb"))
    process_to_int  = pickle.load(open(paths_needed[3], "rb"))

    # splits
    if is_gen:
        split_paths = [
            os.path.join(BASE_PATH, f"{scenario}_train_prefix_gen.pkl"),
            os.path.join(BASE_PATH, f"{scenario}_val_prefix_gen.pkl"),
            os.path.join(BASE_PATH, f"{scenario}_test_prefix_gen.pkl"),
        ]
    else:
        split_paths = [
            os.path.join(BASE_PATH, f"{scenario}_train_prefix.pkl"),
            os.path.join(BASE_PATH, f"{scenario}_val_prefix.pkl"),
            os.path.join(BASE_PATH, f"{scenario}_test_prefix.pkl"),
        ]
    if not all(os.path.exists(p) for p in split_paths):
        miss = [os.path.basename(p) for p in split_paths if not os.path.exists(p)]
        print(f"[SKIP] Missing splits for {label}: {miss}")
        return {"scenario": scenario, "split": "GEN" if is_gen else "BASE", "status": "missing_splits"}

    train_df = pd.read_pickle(split_paths[0])
    val_df   = pd.read_pickle(split_paths[1])
    test_df  = pd.read_pickle(split_paths[2])

    # (optional) label encoding to mirror your pipeline (not used by RT loss but keeps consistency)
    le = LabelEncoder()
    all_labels = pd.concat([train_df["next_activity_int"], val_df["next_activity_int"], test_df["next_activity_int"]])
    le.fit(all_labels)
    train_df["next_activity_int"] = le.transform(train_df["next_activity_int"])
    val_df["next_activity_int"]   = le.transform(val_df["next_activity_int"])
    test_df["next_activity_int"]  = le.transform(test_df["next_activity_int"])

    # event log (filter gateways same as your code)
    df_log = pd.read_csv(log_csv, index_col="Unnamed: 0")
    df_log = df_log[df_log["status"] != "gateway"]

    # build parallel sequences
    train_ps = find_parallel_sequences(
        df_a=train_df, df_b=df_log, considered_timeframe=100.0,
        activity_to_int=activity_to_int, process_to_int=process_to_int
    )
    val_ps = find_parallel_sequences(
        df_a=val_df, df_b=df_log, considered_timeframe=100.0,
        activity_to_int=activity_to_int, process_to_int=process_to_int
    )
    test_ps = find_parallel_sequences(
        df_a=test_df, df_b=df_log, considered_timeframe=100.0,
        activity_to_int=activity_to_int, process_to_int=process_to_int
    )

    # drop empties
    train_ps = _drop_empty(train_ps)
    val_ps   = _drop_empty(val_ps)
    test_ps  = _drop_empty(test_ps)

    # ---- run training (calls your run_experiments) ----
    save_suffix = "_GEN" if is_gen else ""
    results = run_experiments(
        n_runs=10,
        train_df_parallel_sequences=train_ps,
        val_df_parallel_sequences=val_ps,
        test_df_parallel_sequences=test_ps,
        process_to_int=process_to_int,
        save_path_prefix=os.path.join(MODELS_DIR, f"timeaware_{scenario}{save_suffix}.pt"),
        d_model=36, nhead=4, ff_dim=64, dropout=0.2,
        max_seq_len=50, max_parallel=10,
        batch_size=128, num_epochs=200, patience=15
    )

    out_csv = os.path.join(RESULTS_DIR, f"TIMEAWARE_{scenario}{save_suffix}.csv")
    results.to_csv(out_csv, index=False)
    print(f"[DONE] {label} -> {out_csv}")

    return {"scenario": scenario, "split": "GEN" if is_gen else "BASE", "status": "ok", "csv": out_csv}

# ---- parallel launcher ----
def launch_all_timeaware_tasks():
    # Build task list: base + GEN for those available
    tasks = []
    for sc in scenarios:
        tasks.append((sc, False))
        if sc in gen_available:
            tasks.append((sc, True))

    # Determine workers and GPUs
    num_workers = min(len(tasks), os.cpu_count() or 1)
    num_gpus = torch.cuda.device_count()
    print(f"Launching {num_workers} workers over {len(tasks)} tasks | GPUs: {num_gpus}")

    # starmap with (scenario, is_gen, task_idx, num_gpus)
    args = [(sc, is_gen, i, num_gpus) for i, (sc, is_gen) in enumerate(tasks)]

    with mp.Pool(processes=10) as pool:
        summaries = pool.starmap(_train_timeaware_task, args)

    print("\nAll tasks finished.")
    for s in summaries:
        print(s)

# Windows-safe entry (call this from your main)
if __name__ == "__main__":
    mp.freeze_support()
    launch_all_timeaware_tasks()
