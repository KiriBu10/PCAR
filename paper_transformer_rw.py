# run_transformer_on_xes_splits.py
import os
import time
import math
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config_train import VARIANTS as CFG_VARIANTS

# ==========================
# Repro
# ==========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==========================
# Model (your baseline)
# ==========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B,T,d)
        return x + self.pe[:, :x.size(1), :]

class RemainingTimeTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model=36, nhead=4, ff_dim=64, dropout=0.1, num_temporal_features=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.temporal_fc = nn.Linear(num_temporal_features, 32)
        self.dropout1 = nn.Dropout(dropout)
        self.concat_fc = nn.Linear(d_model + 32, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x, temporal_features):
        # x: (B,T)
        x = self.embedding(x)               # (B,T,d)
        x = self.pos_encoder(x)             # (B,T,d)
        x = x.transpose(0, 1)               # (T,B,d) for encoder
        x = self.transformer_encoder(x)     # (T,B,d)
        x = x.transpose(0, 1).permute(0, 2, 1)  # (B,d,T)
        x = self.global_pool(x).squeeze(-1)     # (B,d)

        temporal_x = F.relu(self.temporal_fc(temporal_features))  # (B,32)
        x = torch.cat([x, temporal_x], dim=1)                     # (B, d+32)
        x = self.dropout1(F.relu(self.concat_fc(x)))
        x = self.dropout2(x)
        return self.output_layer(x).squeeze(1)                    # (B,)

# ==========================
# Dataset (+ temporal column auto-detect)
# ==========================
def _ensure_temporal_feature(df: pd.DataFrame, col_name="temporal_feature") -> pd.DataFrame:
    """
    Create a single float column to feed as temporal feature:
    - prefer 'timestamp_case' if present (your original pipeline)
    - else 'prefix_time' (XES pipeline)
    - else last value of 'list_timestamps_case'
    """
    df = df.copy()

    if "timestamp_case" in df.columns:
        df[col_name] = df["timestamp_case"].astype(float)
        return df

    if "prefix_time" in df.columns:
        df[col_name] = df["prefix_time"].astype(float)
        return df

    if "list_timestamps_case" in df.columns:
        def _last_or0(lst):
            if isinstance(lst, (list, tuple)) and len(lst) > 0:
                return float(lst[-1])
            return 0.0
        df[col_name] = df["list_timestamps_case"].apply(_last_or0).astype(float)
        return df

    # fallback: zeros
    df[col_name] = 0.0
    return df

class ProcessDataset(Dataset):
    def __init__(self, df, max_seq_len, temporal_col="temporal_feature"):
        self.df = df
        self.sequences = df['prefix_int'].tolist()
        self.remaining_times = df['remaining_time'].astype(float).tolist()
        self.temporal = df[temporal_col].astype(float).tolist()
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        rt  = self.remaining_times[idx]
        tf  = self.temporal[idx]

        # left-trim & right-pad
        seq = list(seq)
        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        padded = seq + [0] * (self.max_seq_len - len(seq))

        x = torch.tensor(padded, dtype=torch.long)
        t = torch.tensor([tf], dtype=torch.float32)
        y = torch.tensor(rt,  dtype=torch.float32)
        return x, t, y

# ==========================
# Train / Eval
# ==========================
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=100, patience=10):
    best_val = float('inf')
    best_state = None
    no_imp = 0

    loss_fn = nn.L1Loss()

    for ep in range(1, num_epochs+1):
        model.train()
        tr_loss = 0.0
        for x, t, y in train_loader:
            x, t, y = x.to(device), t.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, t)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item())
        tr_loss /= max(1, len(train_loader))

        # val
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, t, y in val_loader:
                x, t, y = x.to(device), t.to(device), y.to(device)
                pred = model(x, t)
                vl += float(F.l1_loss(pred, y).item())
        vl /= max(1, len(val_loader))

        print(f"Epoch {ep:03d} | Train {tr_loss:.4f} | Val {vl:.4f}")

        if vl + 1e-9 < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def evaluate_mae(model, loader, device):
    model.eval()
    preds, gold = [], []
    for x, t, y in loader:
        x, t = x.to(device), t.to(device)
        p = model(x, t).cpu().numpy()
        preds.append(p)
        gold.append(y.numpy())
    if not preds:
        return float("nan")
    p = np.concatenate(preds)
    y = np.concatenate(gold)
    return float(np.mean(np.abs(y - p)))

# ==========================
# Core runner for one split
# ==========================
def run_transformer_on_split(train_df, val_df, test_df,
                             save_model_path: str,
                             results_csv_path: str,
                             max_seq_len=50,
                             d_model=36, nhead=4, ff_dim=64, dropout=0.2,
                             batch_size=128, num_epochs=200, patience=10,
                             n_runs=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure we have the temporal feature column
    train_df = _ensure_temporal_feature(train_df)
    val_df   = _ensure_temporal_feature(val_df)
    test_df  = _ensure_temporal_feature(test_df)

    # Vocab size (pad=0)
    def _max_id(df):
        mx = 0
        for seq in df["prefix_int"]:
            if isinstance(seq, (list, tuple)) and len(seq) > 0:
                mx = max(mx, max(seq))
        return mx
    vocab_size = max(_max_id(train_df), _max_id(val_df), _max_id(test_df)) + 1

    # Datasets/loaders
    tr_ds = ProcessDataset(train_df, max_seq_len)
    va_ds = ProcessDataset(val_df,   max_seq_len)
    te_ds = ProcessDataset(test_df,  max_seq_len)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    rows = []
    for run in range(1, n_runs+1):
        print(f"\n=== Run {run} ===")
        model = RemainingTimeTransformer(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model, nhead=nhead, ff_dim=ff_dim, dropout=dropout,
            num_temporal_features=1
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        t0 = time.time()
        model = train_model(model, tr_dl, va_dl, opt, device, num_epochs=num_epochs, patience=patience)
        train_time = time.time() - t0

        # save & reload
        torch.save(model.state_dict(), save_model_path.replace(".pt", f"_run{run}.pt"))
        model.load_state_dict(torch.load(save_model_path.replace(".pt", f"_run{run}.pt"), map_location=device))

        test_mae = evaluate_mae(model, te_dl, device)
        print(f"Test MAE: {test_mae:.4f}")

        rows.append({"run": run, "training_time_sec": train_time, "test_mae": test_mae})

    pd.DataFrame(rows).to_csv(results_csv_path, index=False)
    print(f"Saved results → {results_csv_path}")
    return rows

# ==========================
# Split file helpers (XES variants)
# ==========================
# def load_variant_split(stem: str, variant: str, dataset_dir="dataset_xes"):
#     """
#     Loads (train, val, test) PKLs for a given log stem (e.g., 'Sepsis')
#     and variant: 'STD' | 'TEST_PAR' | 'TEST_NOPAR' | 'TEST_SAMERES_PAR' | 'GEN'
#     """
#     base = Path(dataset_dir)
#     if variant == "STD":
#         tr = base / f"{stem}_STD_train_prefix.pkl"
#         va = base / f"{stem}_STD_val_prefix.pkl"
#         te = base / f"{stem}_STD_test_prefix.pkl"
#     elif variant == "TEST_PAR":
#         tr = base / f"{stem}_STD_train_prefix.pkl"
#         va = base / f"{stem}_STD_val_prefix.pkl"
#         te = base / f"{stem}_TEST_PAR_test_prefix.pkl"
#     elif variant == "TEST_NOPAR":
#         tr = base / f"{stem}_STD_train_prefix.pkl"
#         va = base / f"{stem}_STD_val_prefix.pkl"
#         te = base / f"{stem}_TEST_NOPAR_test_prefix.pkl"
#     elif variant == "TEST_SAMERES_PAR":
#         tr = base / f"{stem}_STD_train_prefix.pkl"
#         va = base / f"{stem}_STD_val_prefix.pkl"
#         te = base / f"{stem}_TEST_SAMERES_PAR_test_prefix.pkl"
#     elif variant == "GEN":
#         tr = base / f"{stem}_GEN_train_prefix.pkl"
#         va = base / f"{stem}_GEN_val_prefix.pkl"
#         te = base / f"{stem}_GEN_test_prefix.pkl"
#     else:
#         raise ValueError(f"Unknown variant: {variant}")

#     if not tr.exists() or not va.exists() or not te.exists():
#         missing = [p.name for p in [tr, va, te] if not p.exists()]
#         raise FileNotFoundError(f"Missing split files for {stem} / {variant}: {missing}")

#     return pd.read_pickle(tr), pd.read_pickle(va), pd.read_pickle(te)

def load_variant_split(stem: str, variant: str, dataset_dir="dataset_xes"):
    """
    Load (train, val, test) for a given log stem and variant.

    Rules:
      - STD  -> train/val/test from STD_*
      - GEN  -> train/val/test from GEN_*
      - else -> train/val from STD_*, test from <VARIANT>_test_prefix.pkl
    """
    base = Path(dataset_dir)

    def path(suffix: str) -> Path:
        return base / f"{stem}_{suffix}.pkl"

    if variant == "STD":
        tr = path("STD_train_prefix")
        va = path("STD_val_prefix")
        te = path("STD_test_prefix")
    elif variant == "GEN":
        tr = path("GEN_train_prefix")
        va = path("GEN_val_prefix")
        te = path("GEN_test_prefix")
    else:
        tr = path("STD_train_prefix")
        va = path("STD_val_prefix")
        te = path(f"{variant}_test_prefix")

    missing = [p.name for p in (tr, va, te) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files for {stem} / {variant}: {missing}")

    return pd.read_pickle(tr), pd.read_pickle(va), pd.read_pickle(te)


# ==========================
# Main
# ==========================
def main():
    set_seed(42)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    dataset_dir = "dataset_xes"   # change to "dataset" if you want to run on scenario_* PKLs
    logs = [p.stem for p in Path(dataset_dir).glob("*_STD_train_prefix.pkl")]
    # derive stems only once (avoid duplicates)
    stems = sorted(set(s.split("_STD_train_prefix")[0] for s in [Path(p).name for p in Path(dataset_dir).glob("*_STD_train_prefix.pkl")]))

    #variants = ["STD", "TEST_PAR", "TEST_NOPAR", "TEST_SAMERES_PAR", "GEN"]
    variants = list(CFG_VARIANTS)

    # Hyperparams (tweak if you like)
    d_model = 36
    nhead = 4
    ff_dim = 64
    dropout = 0.2
    max_seq_len = 50
    batch_size = 128
    num_epochs = 200
    patience = 10
    n_runs = 1

    for stem in stems:
        for variant in variants:
            try:
                train_df, val_df, test_df = load_variant_split(stem, variant, dataset_dir=dataset_dir)
                model_path = f"models/TRANSFORMER_{stem}_{variant}.pt"
                res_csv    = f"results/TRANSFORMER_{stem}_{variant}.csv"
                run_transformer_on_split(
                    train_df, val_df, test_df,
                    save_model_path=model_path,
                    results_csv_path=res_csv,
                    max_seq_len=max_seq_len,
                    d_model=d_model, nhead=nhead, ff_dim=ff_dim, dropout=dropout,
                    batch_size=batch_size, num_epochs=num_epochs, patience=patience,
                    n_runs=n_runs
                )
            except Exception as e:
                print(f"[SKIP] {stem} / {variant}: {e}")

if __name__ == "__main__":
    main()
