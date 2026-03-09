# run_lstm_on_xes_splits.py
import os
import time
import random
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config_train import VARIANTS as CFG_VARIANTS

# your metrics (same names you used)
from framework.utils import compute_mae, compute_accuracy, compute_f1

# ---------------- reproducibility ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- dataset ----------------
class PrefixDataset(Dataset):
    def __init__(self, df, max_len, num_activities):
        self.max_len = max_len
        self.num_activities = num_activities

        # sequences (pad on the right with 0)
        self.sequences = [torch.tensor(seq[:max_len] + [0]*(max_len - len(seq)), dtype=torch.long)
                          for seq in df["prefix_int"].tolist()]

        # labels
        self.activity_labels = torch.tensor(df["next_activity_int"].astype(int).values, dtype=torch.long)
        self.time_labels     = torch.tensor(df["remaining_time"].astype(float).values, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # one-hot tokens (B,T,C)
        x = F.one_hot(self.sequences[idx], num_classes=self.num_activities).float()
        return x, self.activity_labels[idx], self.time_labels[idx]

# ---------------- model ----------------
class LSTMModel(nn.Module):
    def __init__(self, num_activities, hidden_dim=100):
        super().__init__()
        self.lstm1 = nn.LSTM(num_activities, hidden_dim, batch_first=True)
        self.bn1   = nn.LayerNorm(hidden_dim)
        self.lstm2_1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.bn2_1   = nn.LayerNorm(hidden_dim)
        self.lstm2_2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.bn2_2   = nn.LayerNorm(hidden_dim)

        self.fc_activity = nn.Linear(hidden_dim, num_activities)
        self.fc_time     = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x, _  = self.lstm1(x)
        x1, _ = self.lstm2_1(x)
        x2, _ = self.lstm2_2(x)

        x1_bn = self.bn2_1(x1[:, -1, :])
        x2_bn = self.bn2_2(x2[:, -1, :])

        activity_out = self.fc_activity(x1_bn)
        time_out     = self.fc_time(x2_bn).squeeze(1)
        return activity_out, time_out

# ---------------- train / eval ----------------
def _build_dims(train_df, val_df, test_df):
    # MAX_LEN across all splits
    all_prefixes = pd.concat([train_df["prefix_int"], val_df["prefix_int"], test_df["prefix_int"]], ignore_index=True)
    max_len = max((len(p) for p in all_prefixes), default=1)

    # NUM_ACTIVITIES must cover both prefix_int tokens and next_activity_int labels
    def _max_in_prefixes(series):
        m = 0
        for seq in series:
            if seq:
                m = max(m, max(seq))
        return m

    max_tok = max(_max_in_prefixes(all_prefixes), 0)
    max_lbl = int(pd.concat([train_df["next_activity_int"], val_df["next_activity_int"], test_df["next_activity_int"]], ignore_index=True).max())
    num_activities = max(max_tok, max_lbl) + 1  # 0..num_activities-1
    return max_len, num_activities

def _evaluate(model, loader, device):
    model.eval()
    act_preds, act_targets = [], []
    time_preds, time_targets = [], []

    with torch.no_grad():
        for x, act_y, t_y in loader:
            x = x.to(device)
            act_y = act_y.to(device)
            t_y = t_y.to(device)

            act_out, t_out = model(x)
            pred = torch.argmax(act_out, dim=1)

            act_preds.extend(pred.detach().cpu().numpy())
            act_targets.extend(act_y.detach().cpu().numpy())
            time_preds.extend(t_out.detach().cpu().numpy())
            time_targets.extend(t_y.detach().cpu().numpy())

    acc = compute_accuracy(act_targets, act_preds)
    f1  = compute_f1(act_targets, act_preds, average='macro')
    mae = compute_mae(time_targets, time_preds)
    return acc, f1, mae

def run_lstm_on_split(stem: str, variant: str,
                      train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                      results_dir="results", models_dir="models_lstm_xes",
                      n_runs=10, batch_size=32, max_epochs=100, patience=15, hidden_dim=100, lr=2e-3):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if len(test_df) == 0:
        print(f"[SKIP] {stem}/{variant}: empty test set")
        return {"stem": stem, "variant": variant, "status": "empty_test"}

    # dims
    MAX_LEN, NUM_ACTIVITIES = _build_dims(train_df, val_df, test_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    # datasets / loaders
    train_ds = PrefixDataset(train_df, MAX_LEN, NUM_ACTIVITIES)
    val_ds   = PrefixDataset(val_df,   MAX_LEN, NUM_ACTIVITIES)
    test_ds  = PrefixDataset(test_df,  MAX_LEN, NUM_ACTIVITIES)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    for run in range(1, n_runs + 1):
        print(f"\n[{stem}/{variant}] Run {run}/{n_runs} on {device}")
        set_seed(40 + run)

        model = LSTMModel(NUM_ACTIVITIES, hidden_dim=hidden_dim).to(device)
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
        ce = nn.CrossEntropyLoss()
        l1 = nn.L1Loss()

        best_val = float("inf")
        best_state = None
        bad = 0

        t0 = time.time()
        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            for x, act_y, t_y in train_loader:
                x, act_y, t_y = x.to(device), act_y.to(device), t_y.to(device)
                optimizer.zero_grad()
                act_out, t_out = model(x)
                loss = ce(act_out, act_y) + l1(t_out, t_y)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
            train_loss /= max(1, len(train_loader))

            # val
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, act_y, t_y in val_loader:
                    x, act_y, t_y = x.to(device), act_y.to(device), t_y.to(device)
                    act_out, t_out = model(x)
                    val_loss += float((ce(act_out, act_y) + l1(t_out, t_y)).item())
            val_loss /= max(1, len(val_loader))
            print(f"[{stem}/{variant}] epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print("Early stopping.")
                    break

        train_time = time.time() - t0
        if best_state is not None:
            model.load_state_dict(best_state)

        # save & eval
        mpath = os.path.join(models_dir, f"lstm_{stem}_{variant}_run{run}.pt")
        torch.save(model.state_dict(), mpath)

        acc, f1, mae = _evaluate(model, test_loader, device)
        print(f"[{stem}/{variant}] Run {run} → Acc={acc:.4f}  F1={f1:.4f}  MAE={mae:.4f}")

        rows.append({
            "run": run,
            "test_activity_accuracy": acc,
            "test_activity_f1": f1,
            "test_mae": mae,
            "training_time_sec": train_time,
            "model_path": mpath
        })

    # write per-(stem,variant) csv
    out_csv = os.path.join(results_dir, f"LSTM_{stem}_{variant}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[DONE] {stem}/{variant} → {out_csv}")
    return {"stem": stem, "variant": variant, "status": "ok", "csv": out_csv}

# ---------------- split loading ----------------
def _load_variant_split(dataset_dir: Path, stem: str, variant: str):
    base = dataset_dir

    def P(suffix: str) -> Path:
        return base / f"{stem}_{suffix}.pkl"

    if variant == "STD":
        tr = P("STD_train_prefix")
        va = P("STD_val_prefix")
        te = P("STD_test_prefix")
    elif variant == "GEN":
        tr = P("GEN_train_prefix")
        va = P("GEN_val_prefix")
        te = P("GEN_test_prefix")
    else:
        # default rule: train/val from STD, test from <VARIANT>_test_prefix
        tr = P("STD_train_prefix")
        va = P("STD_val_prefix")
        te = P(f"{variant}_test_prefix")

    missing = [p.name for p in (tr, va, te) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing split(s): {missing}")

    return pd.read_pickle(tr), pd.read_pickle(va), pd.read_pickle(te)


# ---------------- worker (Windows-safe) ----------------
def _worker(stem: str, variant: str,
            dataset_dir: str = "dataset_xes",
            results_dir: str = "results",
            models_dir: str = "models_lstm_xes",
            n_runs: int = 10,
            batch_size: int = 32,
            max_epochs: int = 100,
            patience: int = 15,
            hidden_dim: int = 100,
            lr: float = 2e-3):
    dataset_dir = Path(dataset_dir)
    try:
        tr, va, te = _load_variant_split(dataset_dir, stem, variant)
        return run_lstm_on_split(
            stem, variant, tr, va, te,
            results_dir=results_dir, models_dir=models_dir,
            n_runs=n_runs, batch_size=batch_size,
            max_epochs=max_epochs, patience=patience,
            hidden_dim=hidden_dim, lr=lr
        )
    except Exception as e:
        print(f"[SKIP] {stem}/{variant}: {e}")
        return {"stem": stem, "variant": variant, "status": f"error: {e}"}

# ---------------- main ----------------
def main():
    #set_seed(42)

    dataset_dir = Path("dataset_xes")     # where your XES splits live
    results_dir = "results"
    models_dir  = "models_lstm_xes"

    # discover logs (stems)
    stems = sorted(set(p.name.replace("_STD_train_prefix.pkl", "") for p in dataset_dir.glob("*_STD_train_prefix.pkl")))

    # use config-controlled variants (same list as Transformer)
    variants = list(CFG_VARIANTS)

    # build tasks (you can keep your constants or pull them up as vars)
    tasks = [(stem, var, str(dataset_dir), results_dir, models_dir, 10, 32, 100, 15, 100, 2e-3)
            for stem in stems for var in variants]

    # num_workers = min(len(tasks), os.cpu_count() or 1)
    # print(f"Launching {num_workers} workers for {len(tasks)} tasks...")

    with mp.Pool(processes=5) as pool:
        summaries = pool.starmap(_worker, tasks)

    print("\nAll tasks finished.")
    for s in summaries:
        print(s)

if __name__ == "__main__":
    mp.freeze_support()  # Windows-safe
    main()
