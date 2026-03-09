# amiri2024.py
# ============================================
# PGTNet (GraphGPS-style) runner for XES splits
# ============================================

import os
import math
import time
import random
from collections import defaultdict
from typing import Tuple, Dict, Any, Iterable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error

# ---- Laplacian spectral ops ----
try:
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import eigsh
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from torch.cuda.amp import autocast, GradScaler


# ----------------- reproducibility -----------------
def set_seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------- workload index -----------------
def _safe_end(row: pd.Series):
    return row["end_time"] if ("end_time" in row and pd.notna(row["end_time"])) else row["timestamp"]


def build_case_index(df_log: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build arrays for workload queries:
      returns (case_ids, starts, ends) with timestamps as float.
    """
    dfL = df_log.copy()
    if "status" in dfL.columns:
        dfL = dfL[dfL["status"] != "gateway"]
    if "end_time" not in dfL.columns:
        dfL["end_time"] = np.nan
    dfL["end_time_filled"] = dfL.apply(_safe_end, axis=1)
    dfL = dfL.sort_values(["case_id", "timestamp"], kind="mergesort")

    grp = dfL.groupby("case_id", sort=False)
    starts = grp["timestamp"].min().astype(float).reset_index(name="start")
    ends   = grp["end_time_filled"].max().astype(float).reset_index(name="end")
    win = starts.merge(ends, on="case_id")

    case_ids = win["case_id"].to_numpy()
    starts   = win["start"].to_numpy(dtype=np.float64)
    ends     = win["end"].to_numpy(dtype=np.float64)
    return case_ids, starts, ends


def workload_at_time(t: float, case_index: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> int:
    case_ids, starts, ends = case_index
    return int(((starts <= t) & (ends > t)).sum())


# ----------------- PE: Laplacian & RWSE -----------------
def laplacian_positional_encoding(edge_index: torch.Tensor, num_nodes: int, k_lap: int = 1) -> np.ndarray:
    """
    (N, k_lap) smallest non-trivial eigenvectors of the symmetric normalized Laplacian.
    Sparse + relaxed tol when SciPy is present; dense fallback otherwise.
    """
    if num_nodes <= 0 or edge_index.numel() == 0:
        return np.zeros((max(num_nodes, 0), k_lap), dtype=np.float32)

    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()

    if _HAVE_SCIPY:
        rows = np.concatenate([src, dst])
        cols = np.concatenate([dst, src])
        data = np.ones(len(rows), dtype=np.float32)
        A = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
        deg = np.asarray(A.sum(axis=1)).ravel()
        with np.errstate(divide='ignore'):
            d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        D_inv_sqrt = diags(d_inv_sqrt)
        L = diags(np.ones(num_nodes, dtype=np.float32)) - (D_inv_sqrt @ A @ D_inv_sqrt)

        k = min(k_lap + 1, num_nodes)
        try:
            # relaxed tol/maxiter -> much faster and fine for PE
            _, evecs = eigsh(L, k=k, which='SM', tol=1e-2, maxiter=500)
            evecs = evecs[:, 1:] if evecs.shape[1] > 1 else np.zeros((num_nodes, 0), dtype=np.float32)
        except Exception:
            Ld = L.toarray()
            _, evecs = np.linalg.eigh(Ld)
            evecs = evecs[:, 1:k] if Ld.shape[0] > 1 else np.zeros((num_nodes, 0), dtype=np.float32)
    else:
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for u, v in zip(src, dst):
            A[u, v] += 1.0
            A[v, u] += 1.0
        deg = A.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
        Ld = np.eye(num_nodes, dtype=np.float32) - (D_inv_sqrt @ A @ D_inv_sqrt)
        _, evecs = np.linalg.eigh(Ld)
        k = min(k_lap + 1, num_nodes)
        evecs = evecs[:, 1:k] if num_nodes > 1 else np.zeros((num_nodes, 0), dtype=np.float32)

    if evecs.shape[1] < k_lap:
        pad = np.zeros((num_nodes, k_lap - evecs.shape[1]), dtype=np.float32)
        X = np.concatenate([evecs.astype(np.float32), pad], axis=1)
    else:
        X = evecs[:, :k_lap].astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def rwse_encoding(edge_index: torch.Tensor, num_nodes: int, k_rw: int = 10) -> np.ndarray:
    """
    Random-Walk Structural Encoding: diag(P^k), k=1..K.
    Sparse CSR multiplies when SciPy is present; dense fallback otherwise.
    """
    if num_nodes <= 0:
        return np.zeros((0, k_rw), dtype=np.float32)

    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()

    if _HAVE_SCIPY:
        data = np.ones_like(src, dtype=np.float32)
        A = coo_matrix((data, (src, dst)), shape=(num_nodes, num_nodes)).tocsr()
        deg = np.asarray(A.sum(axis=1)).ravel()
        with np.errstate(divide='ignore'):
            inv_deg = 1.0 / np.maximum(deg, 1e-12)
        D_inv = diags(inv_deg)
        P = D_inv @ A  # row-normalized CSR

        ret = []
        Pk = P
        for _ in range(k_rw):
            ret.append(Pk.diagonal().astype(np.float32))
            Pk = Pk @ P
        X = np.stack(ret, axis=1)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # dense fallback
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for u, v in zip(src, dst):
        A[u, v] += 1.0
    deg_out = A.sum(axis=1)
    P = np.zeros_like(A)
    for i in range(num_nodes):
        if deg_out[i] > 0:
            P[i, :] = A[i, :] / deg_out[i]
        else:
            P[i, i] = 1.0
    ret = []
    Pk = P.copy()
    for _ in range(k_rw):
        ret.append(np.diag(Pk).copy())
        Pk = Pk @ P
    X = np.stack(ret, axis=1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------- prefix graph builder -----------------
def build_prefix_graph(row: pd.Series,
                       case_index: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       edge_feat_cfg: Iterable[str]) -> Dict[str, Any]:
    acts = row["prefix_int"] if isinstance(row["prefix_int"], list) else []
    times = row.get("list_timestamps_case", [])
    if not isinstance(times, list):
        times = []
    acts = [int(a) for a in acts if int(a) != 0]
    times = [float(t) for t in times][:len(acts)]
    # sanitize timestamps (length/finite/monotone)
    def _bad(ts):
        if len(ts) != len(acts): return True
        if len(ts) == 0: return False
        ts_arr = np.asarray(ts, dtype=np.float64)
        if not np.all(np.isfinite(ts_arr)): return True
        if np.any(np.diff(ts_arr) < -1e-9): return True
        return False
    if _bad(times):
        times = np.linspace(0.0, float(len(acts)), num=len(acts), endpoint=False, dtype=np.float64).tolist()

    L = len(acts)
    if L == 0:
        acts, times = [0], [0.0]
        L = 1

    uniq, seen = [], set()
    for a in acts:
        if a not in seen:
            uniq.append(a); seen.add(a)
    node_index = {a: i for i, a in enumerate(uniq)}
    node_ids = torch.tensor(uniq, dtype=torch.long)

    stats = defaultdict(lambda: {"cnt": 0, "tot_dt": 0.0, "last_t": 0.0, "last_dt": 0.0})
    for i in range(L - 1):
        u, v = node_index[acts[i]], node_index[acts[i + 1]]
        dt = max(0.0, times[i + 1] - times[i])
        s = stats[(u, v)]
        s["cnt"] += 1
        s["tot_dt"] += dt
        s["last_t"] = times[i + 1]
        s["last_dt"] = dt

    if len(stats) == 0:
        i0 = node_index[acts[-1]]
        stats[(i0, i0)] = {"cnt": 1, "tot_dt": 0.0, "last_t": times[-1], "last_dt": 0.0}

    max_cnt = max(1, max(s["cnt"] for s in stats.values()))
    max_tot = max(1e-6, max(s["tot_dt"] for s in stats.values()))
    elapsed = float(times[-1]) if len(times) else 0.0
    t_global = float(row.get("prefix_time", elapsed))
    wl = math.log1p(float(workload_at_time(t_global, case_index)))

    src, dst, e_feat = [], [], []
    for (u, v), s in stats.items():
        cnt_norm = s["cnt"] / max_cnt
        tot_norm = s["tot_dt"] / max_tot
        rec_norm = (elapsed - s["last_t"]) / (elapsed + 1e-6) if elapsed > 0 else 0.0
        last_dt = s["last_dt"]
        pref_len = L
        elapsed_s = elapsed
        feats = []
        for key in edge_feat_cfg:
            if key == "cnt_norm": feats.append(cnt_norm)
            elif key == "tot_dt_norm": feats.append(tot_norm)
            elif key == "recency_norm": feats.append(rec_norm)
            elif key == "log_workload": feats.append(wl)
            elif key == "last_dt": feats.append(last_dt)
            elif key == "prefix_len": feats.append(float(pref_len))
            elif key == "elapsed": feats.append(elapsed_s)
        src.append(u); dst.append(v); e_feat.append(feats)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    e_feat = np.asarray(e_feat, dtype=np.float32)
    e_feat = np.nan_to_num(e_feat, nan=0.0, posinf=0.0, neginf=0.0)
    edge_attr = torch.from_numpy(e_feat)
    return {"node_ids": node_ids, "edge_index": edge_index, "edge_attr": edge_attr}


def collate_graphs(batch):
    graphs, ys = zip(*batch)
    return list(graphs), torch.stack(ys, dim=0)


# ----------------- GraphGPS-style components -----------------
class TwoLayerLinearEdgeEncoder(nn.Module):
    def __init__(self, in_dim: int, dim_hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, out_dim)
        )
    def forward(self, e):  # (E, in_dim) -> (E, out_dim)
        return self.net(e)


class GINECore(nn.Module):
    """
    GINE-like local message passing core.
    Messages: ReLU(x_i + e_ij), sum aggregation, learnable eps, node MLP.
    """
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.node_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(inplace=True),
                       nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model)
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        m = torch.relu(x[src] + edge_attr)  # (E,d)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)
        out = (1.0 + self.eps) * x + agg
        return self.node_mlp(out)


class GPSLayer(nn.Module):
    """
    GraphGPS block (Pre-Norm):
      x = x + GINE(LN x)
      x = x + MHA(LN x)
      x = x + FFN(LN x)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float, attn_dropout: float = 0.5):
        super().__init__()
        self.norm_local = nn.LayerNorm(d_model)
        self.gine = GINECore(d_model, dropout=dropout)

        self.norm_attn = nn.LayerNorm(d_model)
        try:
            self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=attn_dropout, batch_first=True)
            self._batch_first = True
        except TypeError:
            self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=attn_dropout)
            self._batch_first = False
        self.attn_drop = nn.Dropout(attn_dropout)

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model)
        )
        self.ff_drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        xl = self.gine(self.norm_local(x), edge_index, edge_attr)
        x = x + xl

        xa = self.norm_attn(x)
        if self._batch_first:
            xa, _ = self.attn(xa.unsqueeze(0), xa.unsqueeze(0), xa.unsqueeze(0))
            xa = xa.squeeze(0)
        else:
            q = xa.unsqueeze(1)
            xa, _ = self.attn(q, q, q)
            xa = xa.squeeze(1)
        x = x + self.attn_drop(xa)

        xf = self.ff(self.norm_ff(x))
        x = x + self.ff_drop(xf)
        return x


class NodeEncoder(nn.Module):
    """
    Node stream = ActivityEmbedding + MLP(LapPE) + BN(RWSE) + MLP(RWSE) + optional degree bias.
    """
    def __init__(self, num_activities: int, d_model: int, k_lap: int, k_rw: int,
                 pe_dropout: float, use_deg_bias: bool = True, deg_vocab: int = 11, use_rw_bn: bool = True):
        super().__init__()
        self.act_emb = nn.Embedding(num_activities, d_model, padding_idx=0)
        self.lap_mlp = nn.Sequential(
            nn.Linear(k_lap, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(pe_dropout),
            nn.Linear(d_model, d_model)
        )
        self.rw_bn = nn.BatchNorm1d(k_rw) if use_rw_bn else nn.Identity()
        self.rw_mlp = nn.Sequential(
            nn.Linear(k_rw, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(pe_dropout),
            nn.Linear(d_model, d_model)
        )
        self.pe_drop = nn.Dropout(pe_dropout)
        self.use_deg_bias = use_deg_bias
        if use_deg_bias:
            self.in_deg_emb  = nn.Embedding(deg_vocab, d_model)
            self.out_deg_emb = nn.Embedding(deg_vocab, d_model)

    def forward(self, node_ids, lap, rw, deg_in=None, deg_out=None):
        # BN over feature dim for RWSE
        if isinstance(self.rw_bn, nn.BatchNorm1d):
            rw = self.rw_bn(rw)
        x = self.act_emb(node_ids) + self.lap_mlp(lap) + self.rw_mlp(rw)
        if self.use_deg_bias and deg_in is not None and deg_out is not None:
            x = x + self.in_deg_emb(deg_in) + self.out_deg_emb(deg_out)
        return self.pe_drop(x)


class PGTNetRepoish(nn.Module):
    """
    GraphGPS-style stack with add pooling head for scalar regression.
    """
    def __init__(self, num_activities: int, e_in_dim: int, d_model: int = 64, n_heads: int = 8, n_layers: int = 5,
                 k_lap: int = 1, k_rw: int = 10, pe_dropout: float = 0.1, model_dropout: float = 0.0,
                 attn_dropout: float = 0.5, graph_pooling: str = "add",
                 use_deg_bias: bool = True, deg_vocab: int = 11):
        super().__init__()
        self.graph_pooling = graph_pooling
        self.node_enc = NodeEncoder(num_activities, d_model, k_lap, k_rw, pe_dropout,
                                    use_deg_bias=use_deg_bias, deg_vocab=deg_vocab, use_rw_bn=True)
        self.edge_enc = TwoLayerLinearEdgeEncoder(e_in_dim, 2 * d_model, d_model, model_dropout)
        self.layers = nn.ModuleList([GPSLayer(d_model, n_heads, model_dropout, attn_dropout=attn_dropout)
                                     for _ in range(n_layers)])
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(model_dropout),
            nn.Linear(d_model, 1)
        )

    def encode_graph(self, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = self.readout[-1].weight.device
        node_ids  = graph["node_ids"].to(device)
        edge_index = graph["edge_index"].to(device)
        edge_attr  = graph["edge_attr"].to(device)

        if "lap" in graph and "rw" in graph:
            lap = graph["lap"].to(device); rw = graph["rw"].to(device)
        else:
            N = node_ids.size(0)
            lap = torch.from_numpy(laplacian_positional_encoding(edge_index, N, k_lap=self.node_enc.lap_mlp[0].in_features)).to(device)
            rw  = torch.from_numpy(rwse_encoding(edge_index, N, k_rw=self.node_enc.rw_mlp[0].in_features)).to(device)

        # degrees (cap at deg_vocab-1)
        if self.node_enc.use_deg_bias:
            N = node_ids.size(0)
            deg_in  = torch.bincount(edge_index[1], minlength=N).clamp_max(self.node_enc.in_deg_emb.num_embeddings - 1)
            deg_out = torch.bincount(edge_index[0], minlength=N).clamp_max(self.node_enc.out_deg_emb.num_embeddings - 1)
        else:
            deg_in = deg_out = None

        x = self.node_enc(node_ids, lap, rw, deg_in, deg_out)
        e = self.edge_enc(edge_attr)

        for layer in self.layers:
            x = layer(x, edge_index, e)

        if self.graph_pooling == "add":
            g = x.sum(dim=0)
        else:
            g = x.mean(dim=0)
        return self.readout(g).squeeze(-1)

    def forward(self, graphs: Iterable[Dict[str, torch.Tensor]]) -> torch.Tensor:
        preds = [self.encode_graph(g) for g in graphs]
        return torch.stack(preds, dim=0)


# ----------------- Cached dataset -----------------
class CachedPrefixDataset(torch.utils.data.Dataset):
    """
    One row = one prefix.
    Precompute graphs + LapPE/RWSE (optional) and keep in-memory.
    """
    def __init__(self,
                 df_split: pd.DataFrame,
                 case_index: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 edge_feat_keys: Iterable[str],
                 name: str,
                 precompute_graphs: bool,
                 k_lap: int,
                 k_rw: int):
        self.rows = df_split.reset_index(drop=True).copy()
        # keep only finite labels
        self.rows = self.rows[np.isfinite(self.rows["remaining_time"].astype(float).values)].reset_index(drop=True)
        self.y = self.rows["remaining_time"].astype(float).to_numpy()
        self.name = name
        self.case_index = case_index
        self.edge_feat_keys = tuple(edge_feat_keys)
        self.precompute_graphs = precompute_graphs
        self.k_lap = int(k_lap)
        self.k_rw = int(k_rw)
        self.graphs = None

    def __len__(self): return len(self.rows)

    def build_all(self):
        gs = []
        for _, row in self.rows.iterrows():
            g = build_prefix_graph(row, self.case_index, self.edge_feat_keys)
            if self.precompute_graphs:
                N = g["node_ids"].numel()
                lap = laplacian_positional_encoding(g["edge_index"], N, k_lap=self.k_lap)
                rw  = rwse_encoding(g["edge_index"], N, k_rw=self.k_rw)
                g["lap"] = torch.from_numpy(lap).float()
                g["rw"]  = torch.from_numpy(rw).float()
            gs.append(g)
        self.graphs = gs

    def __getitem__(self, idx: int):
        if self.graphs is None:
            row = self.rows.iloc[idx]
            g = build_prefix_graph(row, self.case_index, self.edge_feat_keys)
            if self.precompute_graphs:
                N = g["node_ids"].numel()
                lap = laplacian_positional_encoding(g["edge_index"], N, k_lap=self.k_lap)
                rw  = rwse_encoding(g["edge_index"], N, k_rw=self.k_rw)
                g["lap"] = torch.from_numpy(lap).float()
                g["rw"]  = torch.from_numpy(rw).float()
        else:
            g = self.graphs[idx]
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return g, y


# ----------------- Scheduler -----------------
def build_cosine_with_warmup(optimizer, max_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ----------------- Runner -----------------
def run_pgt_net_repoish_for_scenario(
    scenario: str,
    activity_to_int: Dict[str, int],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    df_log: pd.DataFrame,
    # model/config (mirrors YAML defaults)
    d_model: int = 64,
    num_heads: int = 8,
    n_layers: int = 5,
    k_lap: int = 1,
    k_rw: int = 10,
    pe_dropout: float = 0.1,
    model_dropout: float = 0.0,
    attn_dropout: float = 0.5,
    # edge feature selection (order matters)
    edge_feat_keys=("cnt_norm", "tot_dt_norm", "recency_norm", "log_workload", "last_dt", "prefix_len", "elapsed"),
    # optimization
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 200,
    patience: int = 10,
    save_dir: str = "models_pgt_repoish",
    # performance knobs
    precompute_graphs: bool = True,  # we precompute train & val before training; test right before eval
    num_workers: int = 0,
    use_amp: bool = None,
    use_compile: bool = False
) -> Dict[str, Any]:
    """
    Train & evaluate GraphGPS-style PGTNet on pre-built XES splits.
    Saves a checkpoint & test predictions, returns metrics.
    """
    set_seed_all(42)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_amp is None:
        use_amp = (device.type == "cuda")

    # Enable TF32 on Ampere+
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print(f"[{scenario}] device={device} | CUDA={torch.cuda.is_available()}")

    # workload index
    base_log = df_log[df_log["status"] != "gateway"] if "status" in df_log.columns else df_log
    case_index = build_case_index(base_log)

    # ----- Cached datasets -----
    tr_ds = CachedPrefixDataset(train_df, case_index, edge_feat_keys, "train", precompute_graphs, k_lap, k_rw)
    va_ds = CachedPrefixDataset(val_df,   case_index, edge_feat_keys, "val",   precompute_graphs, k_lap, k_rw)
    te_ds = CachedPrefixDataset(test_df,  case_index, edge_feat_keys, "test",  precompute_graphs, k_lap, k_rw)

    # Precompute train & val now; defer test to right before evaluation
    tr_ds.build_all(); print(f"[{scenario}] cached train graphs: {len(tr_ds)}")
    va_ds.build_all(); print(f"[{scenario}] cached val graphs:   {len(va_ds)}")
    # (do NOT build te_ds here)

    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs,
        num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0)
    )
    val_loader = torch.utils.data.DataLoader(
        va_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs,
        num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0)
    )

    num_activities = int(max(activity_to_int.values())) + 1 if len(activity_to_int) else 1
    e_in_dim = len(edge_feat_keys)
    model = PGTNetRepoish(
        num_activities=num_activities,
        e_in_dim=e_in_dim,
        d_model=d_model, n_heads=num_heads, n_layers=n_layers,
        k_lap=k_lap, k_rw=k_rw,
        pe_dropout=pe_dropout, model_dropout=model_dropout,
        attn_dropout=attn_dropout,
        graph_pooling="add",
        use_deg_bias=True, deg_vocab=11
    ).to(device)

    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"[{scenario}] torch.compile enabled")
        except Exception as e:
            print(f"[{scenario}] torch.compile failed: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = build_cosine_with_warmup(opt, max_epochs=max_epochs, warmup_epochs=50)
    scaler = GradScaler(enabled=use_amp)

    def _train_epoch() -> float:
        model.train()
        losses = []
        for graphs, ys in train_loader:
            ys = ys.to(device, non_blocking=True)
            graphs = [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in g.items()} for g in graphs]
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                pred = model(graphs)
                loss = F.l1_loss(pred, ys)
            scaler.scale(loss).backward()
            if True:  # clip_grad_norm (like YAML)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    @torch.no_grad()
    def _eval(loader) -> float:
        model.eval()
        Ys, Ps = [], []
        for graphs, ys in loader:
            ys = ys.to(device, non_blocking=True)
            graphs = [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in g.items()} for g in graphs]
            with autocast(enabled=use_amp):
                p = model(graphs)
            Ys.append(ys.cpu().numpy()); Ps.append(p.cpu().numpy())
        if not Ys:
            return 0.0
        y = np.concatenate(Ys); p = np.concatenate(Ps)
        return float(mean_absolute_error(y, p))

    # ----- train loop with early stopping -----
    best_val, best_state, no_imp = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        tr_loss = _train_epoch()
        val_mae = _eval(val_loader)
        print(f"[{scenario}] Epoch {ep:03d} | lr={sched.get_last_lr()[0]:.5f} | train L1={tr_loss:.4f} | val MAE={val_mae:.4f}")
        if val_mae + 1e-6 < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"[{scenario}] Early stopping.")
                break
        sched.step()

    training_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # Build test graphs now (once)
    te_ds.build_all(); print(f"[{scenario}] cached test graphs:  {len(te_ds)}")
    test_loader = torch.utils.data.DataLoader(
        te_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs,
        num_workers=num_workers, pin_memory=pin, persistent_workers=False
    )

    val_mae = _eval(val_loader)
    test_mae = _eval(test_loader)
    print(f"[{scenario}] DONE | val MAE={val_mae:.4f} | test MAE={test_mae:.4f} | time={training_time:.1f}s")

    # ----- save checkpoint + test preds -----
    model_path = os.path.join(save_dir, f"pgt_repoish_{scenario}.pt")
    torch.save(model.state_dict(), model_path)

    @torch.no_grad()
    def _dump_preds(loader, path_csv):
        model.eval()
        Ys, Ps = [], []
        for graphs, ys in loader:
            graphs = [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in g.items()} for g in graphs]
            with autocast(enabled=use_amp):
                p = model(graphs)
            Ys.append(ys.numpy()); Ps.append(p.cpu().numpy())
        y = np.concatenate(Ys); pred = np.concatenate(Ps)
        pd.DataFrame({"y_true": y, "y_pred": pred}).to_csv(path_csv, index=False)

    _dump_preds(test_loader, os.path.join(save_dir, f"pgt_repoish_preds_{scenario}.csv"))

    return {
        "scenario": scenario,
        "val_mae": float(val_mae),
        "test_mae": float(test_mae),
        "training_time_sec": float(training_time),
        "model_path": model_path,
        "params": dict(d_model=d_model, num_heads=num_heads, n_layers=n_layers, k_lap=k_lap, k_rw=k_rw)
    }


import os
import pickle
import pandas as pd
import multiprocessing as mp

# --- config ---
n_runs = 1
save_dir = "models_pgt"
results_dir = "results"
base_path = "dataset"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

MODEL_KW = dict(
    d_model=64, num_heads=4, n_layers=4,
    k_lap=8, k_rw=8,
    batch_size=64, lr=1e-3, weight_decay=1e-5,
    max_epochs=200, patience=12,
    save_dir=save_dir
)

scenarios = [
    # "scenario_1_A",
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

def _run_all_runs_for_task(scenario: str, is_gen: bool):
    """Runs n_runs for a (scenario, split-type) task; saves a CSV of results."""
    label = f"{scenario} (GEN)" if is_gen else scenario
    print(f"\n=== {label} ===")

    file_name = os.path.join(base_path, f"RLRAM_l0.5_s00_{scenario}.csv")
    if not os.path.exists(file_name):
        print(f"[SKIP] Missing event log: {file_name}")
        return {"scenario": scenario, "split": "GEN" if is_gen else "BASE", "status": "missing_log"}

    # load shared data once
    df_log = pd.read_csv(file_name, index_col="Unnamed: 0")
    act_map_path = os.path.join(base_path, f"{scenario}_activity_to_int.p")
    if not os.path.exists(act_map_path):
        print(f"[SKIP] Missing activity_to_int: {act_map_path}")
        return {"scenario": scenario, "split": "GEN" if is_gen else "BASE", "status": "missing_map"}
    activity_to_int = pickle.load(open(act_map_path, "rb"))

    if is_gen:
        # ensure GEN exists
        needed = [
            os.path.join(base_path, f"{scenario}_train_prefix_gen.pkl"),
            os.path.join(base_path, f"{scenario}_val_prefix_gen.pkl"),
            os.path.join(base_path, f"{scenario}_test_prefix_gen.pkl"),
        ]
        if not all(os.path.exists(p) for p in needed):
            missing = [os.path.basename(p) for p in needed if not os.path.exists(p)]
            print(f"[SKIP] Missing GEN splits for {scenario}: {missing}")
            return {"scenario": scenario, "split": "GEN", "status": "missing_gen_splits"}

        train_df = pd.read_pickle(needed[0])
        val_df   = pd.read_pickle(needed[1])
        test_df  = pd.read_pickle(needed[2])
        scen_arg = f"{scenario}_GEN"
        out_csv  = os.path.join(results_dir, f"PGT_{scenario}_GEN.csv")
    else:
        train_df = pd.read_pickle(os.path.join(base_path, f"{scenario}_train_prefix.pkl"))
        val_df   = pd.read_pickle(os.path.join(base_path, f"{scenario}_val_prefix.pkl"))
        test_df  = pd.read_pickle(os.path.join(base_path, f"{scenario}_test_prefix.pkl"))
        scen_arg = scenario
        out_csv  = os.path.join(results_dir, f"PGT_{scenario}.csv")

    rows = []
    for run in range(1, n_runs + 1):
        seed   = (410 if is_gen else 41) + run
        suffix = f"{'_GEN' if is_gen else ''}_run{run}"

        print(f"Run {run}/{n_runs} — {label}")
        stats = run_pgt_net_repoish_for_scenario(
            scenario=scen_arg,
            activity_to_int=activity_to_int,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            df_log=df_log,
            # if you add these params to your runner, uncomment:
            # seed=seed,
            # save_suffix=suffix,
            **MODEL_KW
        )

        rows.append({
            "run": run,
            "seed": stats.get("seed", seed),
            "val_mae": stats["val_mae"],
            "test_mae": stats["test_mae"],
            "training_time_sec": stats.get("training_time_sec", None),
            "model_path": stats.get("model_path", ""),
            "preds_path": stats.get("preds_path", ""),
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return {"scenario": scenario, "split": "GEN" if is_gen else "BASE", "status": "ok", "csv": out_csv}

def main():
    # Build the task list: base for all scenarios + GEN for those in gen_available
    tasks = []
    for sc in scenarios:
        tasks.append((sc, False))
        if sc in gen_available:
            tasks.append((sc, True))

    # Parallelize ACROSS tasks using starmap (no lambdas/closures)
    num_workers = min(len(tasks), os.cpu_count() or 1)
    print(f"Launching {num_workers} workers for {len(tasks)} tasks...")

    with mp.Pool(processes=10) as pool:#num_workers
        summaries = pool.starmap(_run_all_runs_for_task, tasks)

    print("\nAll tasks finished.")
    for s in summaries:
        print(s)

if __name__ == "__main__":
    mp.freeze_support()  # Windows-safe
    main()
