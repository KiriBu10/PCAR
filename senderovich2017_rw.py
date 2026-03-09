# =======================
# Senderovich Level-3 + XGBoost
# =======================
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pickle
from tqdm import tqdm
import time

from config_train import VARIANTS as CFG_VARIANTS

# ---------- Utilities ----------
def _safe_end(x):
    # fall back to timestamp if end_time is missing
    return x["end_time"] if pd.notna(x["end_time"]) else x["timestamp"]

# def build_case_index(df_log: pd.DataFrame, activity_to_int: dict):
#     """
#     Build per-case timelines and fast arrays for active-case queries.
#     Returns:
#       case_ids: np.array shape (C,)
#       starts:   np.array shape (C,)
#       ends:     np.array shape (C,)
#       times_by_case: dict[case_id] -> np.array of event start times (sorted)
#       acts_by_case:  dict[case_id] -> np.array of activity_int (sorted by the times above)
#     """
#     dfL = df_log[df_log["status"] != "gateway"].copy()
#     if "end_time" not in dfL:
#         dfL["end_time"] = np.nan
#     dfL["end_time_filled"] = dfL.apply(_safe_end, axis=1)

#     # map activities to ints (unknown -> 0)
#     dfL["activity_int"] = dfL["activity"].map(activity_to_int).fillna(0).astype(int)

#     # ensure sorting per case
#     dfL = dfL.sort_values(["case_id", "timestamp"], kind="mergesort")

#     grp = dfL.groupby("case_id", sort=False)
#     starts = grp["timestamp"].min().astype(float)
#     ends   = grp["end_time_filled"].max().astype(float)

#     # build arrays per case
#     times_by_case = {}
#     acts_by_case  = {}
#     for cid, g in grp:
#         times_by_case[cid] = g["timestamp"].to_numpy(dtype=float)
#         acts_by_case[cid]  = g["activity_int"].to_numpy(dtype=int)

#     case_ids = starts.index.to_numpy(dtype=int)#or int
#     starts   = starts.to_numpy()
#     ends     = ends.to_numpy()

#     return case_ids, starts, ends, times_by_case, acts_by_case

def build_case_index(df_log: pd.DataFrame, activity_to_int: dict):
    dfL = df_log[df_log["status"] != "gateway"].copy()
    if "end_time" not in dfL:
        dfL["end_time"] = np.nan
    dfL["end_time_filled"] = dfL.apply(_safe_end, axis=1)

    # map activities to ints (unknown -> 0)
    dfL["activity_int"] = dfL["activity"].map(activity_to_int).fillna(0).astype(int)

    # ensure sorting per case
    dfL = dfL.sort_values(["case_id", "timestamp"], kind="mergesort")

    grp = dfL.groupby("case_id", sort=False)
    starts = grp["timestamp"].min().astype(float)
    ends   = grp["end_time_filled"].max().astype(float)

    # build arrays per case
    times_by_case = {}
    acts_by_case  = {}
    for cid, g in grp:
        times_by_case[cid] = g["timestamp"].to_numpy(dtype=float)
        acts_by_case[cid]  = g["activity_int"].to_numpy(dtype=int)

    # **FIX HERE: don't force case_ids to int**
    case_ids = starts.index.to_numpy()  # <- no dtype=int
    starts   = starts.to_numpy()
    ends     = ends.to_numpy()

    return case_ids, starts, ends, times_by_case, acts_by_case


def last_k_pattern_at(times_arr: np.ndarray, acts_arr: np.ndarray, t: float, k: int):
    """
    Return tuple of last k activities (as ints) up to time t for a single case.
    If no event at or before t, returns empty tuple ().
    """
    # index of last event with start_time <= t
    idx = np.searchsorted(times_arr, t, side="right") - 1
    if idx < 0:
        return ()
    start = max(0, idx - k + 1)
    return tuple(acts_arr[start:idx+1])

def level3_feature_counts_at_time(t: float,
                                  case_ids: np.ndarray,
                                  starts: np.ndarray,
                                  ends: np.ndarray,
                                  times_by_case: dict,
                                  acts_by_case: dict,
                                  k: int,
                                  focal_case_id: int):
    """
    Compute Level-3 dictionary { "PAT[a|b|c]": count } for all cases active at time t,
    excluding focal_case_id.
    """
    # active mask
    mask = (starts <= t) & (ends > t)
    active_ids = case_ids[mask]

    feat = Counter()
    for cid in active_ids:
        if cid == focal_case_id:
            continue
        pat = last_k_pattern_at(times_by_case[cid], acts_by_case[cid], t, k)
        if len(pat) == 0:
            continue
        key = "PAT[" + "|".join(map(str, pat)) + "]"
        feat[key] += 1
    return dict(feat)

def build_row_features(row, l3_counts: dict, vocab_size_acts: int):
    """
    Minimal intra-case + inter-case features:
      - elapsed since start (relative, from your 'timestamp_case')
      - prefix length
      - last activity (one-hot via DictVectorizer by using a string key)
      - Level-3 counts (already in dict)
    """
    x = {}
    # elapsed since start (relative time inside the case, consistent with your list_timestamps_case[-1])
    elapsed = row.get("timestamp_case", None)
    if elapsed is None or pd.isna(elapsed):
        # fallback: relative time = last item in list_timestamps_case
        lst = row.get("list_timestamps_case", [])
        elapsed = lst[-1] if len(lst) else 0.0
    x["elapsed_case"] = float(elapsed)

    # prefix length
    pref = row["prefix_int"] if isinstance(row["prefix_int"], list) else []
    x["prefix_len"] = len(pref)

    # last activity (categorical)
    last_act = pref[-1] if len(pref) else 0
    x[f"LAST_ACT={last_act}"] = 1  # one-hot via DictVectorizer

    # merge Level-3 counts
    x.update(l3_counts)
    return x

def build_dataset_level3(df_samples: pd.DataFrame,
                         case_index,
                         k_last=3,
                         top_m_patterns=2000,
                         fit_vectorizer: bool = True,
                         dict_vectorizer: DictVectorizer = None,
                         vocab_size_acts: int = None):
    """
    Creates (X, y, dv, pattern_stats) for a split.
    If fit_vectorizer=True, it also prunes to top_m_patterns by frequency on this split (use for train only).
    """
    case_ids, starts, ends, times_by_case, acts_by_case = case_index

    # First pass: build raw dict features and collect pattern frequencies (train only)
    X_dicts = []
    y = []
    pat_freq = Counter()

    for _, row in tqdm(df_samples.iterrows(), total=len(df_samples), desc="Level3 features"):
        t = float(row["prefix_time"])  # global time of the prediction point
        focal_cid = int(row["case_id"])
        l3 = level3_feature_counts_at_time(t, case_ids, starts, ends, times_by_case, acts_by_case, k_last, focal_cid)
        # track frequencies for pruning (train only)
        pat_freq.update(l3.keys())
        feats = build_row_features(row, l3, vocab_size_acts)
        X_dicts.append(feats)
        y.append(float(row["remaining_time"]))
    y = np.asarray(y, dtype=float)

    # Optional pruning to top-M patterns to keep the space compact (train only)
    keep_patterns = None
    if fit_vectorizer and top_m_patterns is not None:
        # sort by frequency then by key for determinism
        top_keys = [k for k, _ in pat_freq.most_common(top_m_patterns)]
        keep_patterns = set(top_keys)

        def _prune(d):
            if not d:
                return d
            out = {k: v for k, v in d.items() if not k.startswith("PAT[") or k in keep_patterns}
            return out

        X_dicts = [_prune(d) for d in X_dicts]

    # Fit/transform with DictVectorizer
    if fit_vectorizer:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(X_dicts)
    else:
        assert dict_vectorizer is not None, "Provide the fitted DictVectorizer for val/test."
        # also prune unseen patterns implicitly (dv ignores unknown keys)
        X = dict_vectorizer.transform(X_dicts)
        dv = dict_vectorizer

    return X, y, dv, pat_freq

def train_xgb_regressor(X, y, seed=None):
    """
    XGBoost regressor with sensible defaults.
    Tweak if you see over/underfit.
    """
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        #subsample=0.8,
        #colsample_bytree=0.8,
        #reg_lambda=1.0,
        objective="reg:squarederror",
        #random_state=seed,
        n_jobs=0,
        tree_method="hist"
    )
    model.fit(X, y)
    return model

# ---------- Runner per scenario ----------
def run_level3_xgboost_for_scenario(scenario,
                                    activity_to_int,
                                    train_df,
                                    val_df,
                                    test_df,
                                    df_log,
                                    k_last=3,
                                    top_m_patterns=2000,
                                    save_dir="models_level3"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Build case index once per scenario
    case_index = build_case_index(df_log[df_log["status"]!="gateway"], activity_to_int)
    vocab_size_acts = max(activity_to_int.values()) + 1

    # TRAIN
    X_tr, y_tr, dv, pat_freq = build_dataset_level3(
        train_df, case_index,
        k_last=k_last, top_m_patterns=top_m_patterns,
        fit_vectorizer=True, dict_vectorizer=None,
        vocab_size_acts=vocab_size_acts
    )
    t0 = time.time()
    model = train_xgb_regressor(X_tr, y_tr)#, seed=42)
    training_time = time.time() - t0

    # VAL
    X_va, y_va, _, _ = build_dataset_level3(
        val_df, case_index,
        k_last=k_last, top_m_patterns=None,
        fit_vectorizer=False, dict_vectorizer=dv,
        vocab_size_acts=vocab_size_acts
    )
    val_pred = model.predict(X_va)
    val_mae = mean_absolute_error(y_va, val_pred)
    print(f"[{scenario}] VAL MAE = {val_mae:.4f}")

    # TEST
    X_te, y_te, _, _ = build_dataset_level3(
        test_df, case_index,
        k_last=k_last, top_m_patterns=None,
        fit_vectorizer=False, dict_vectorizer=dv,
        vocab_size_acts=vocab_size_acts
    )
    test_pred = model.predict(X_te)
    test_mae = mean_absolute_error(y_te, test_pred)
    print(f"[{scenario}] TEST MAE = {test_mae:.4f}")

    # Save artifacts
    with open(f"{save_dir}/l3_dv_{scenario}.pkl", "wb") as f:
        pickle.dump(dv, f)
    with open(f"{save_dir}/l3_model_{scenario}.pkl", "wb") as f:
        pickle.dump(model, f)

    # Also save quick diagnostics
    pd.DataFrame({
        "y_true": y_te,
        "y_pred": test_pred
    }).to_csv(f"{save_dir}/l3_preds_{scenario}.csv", index=False)

    return {
        "scenario": scenario,
        "val_mae": val_mae,
        "test_mae": test_mae,
        "training_time_sec":training_time,
        "n_features": X_tr.shape[1],
        "top_m_patterns": top_m_patterns,
        "unique_patterns_seen_train": sum(1 for k in pat_freq if k.startswith("PAT["))
    }




# run_level3_on_xes_variants.py
import os
import pickle
import pandas as pd
from pathlib import Path
from framework.utils import numeric_case_id_series


# <<< make sure this import works >>>
# from your_module import run_level3_xgboost_for_scenario

# ---------------- helpers ----------------
def load_df_log_from_xes(xes_path: str) -> pd.DataFrame:
    """Load a XES file and return a df_log compatible with your Level-3 code."""
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.objects.conversion.log import converter as log_converter

    log = xes_importer.apply(xes_path)
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

    case_col = "case:concept:name"
    act_col  = "concept:name"
    time_col = "time:timestamp"

    # resource column if present
    res_col = None
    for c in ["org:resource", "Resource", "resource"]:
        if c in df.columns:
            res_col = c
            break

    if not all(c in df.columns for c in [case_col, act_col, time_col]):
        raise ValueError(f"Missing required XES columns in {xes_path}")

    proc_name = Path(xes_path).stem
    case_ids_raw = df[case_col].astype(str)
    case_ids_num = numeric_case_id_series(case_ids_raw).astype(int)

    out = pd.DataFrame({
        "process":   proc_name,
        "case_id":   case_ids_num,
        "activity":  df[act_col].astype(str),
        # absolute seconds (float). Your Level-3 only needs an orderable time axis.
        "timestamp": pd.to_datetime(df[time_col]).astype("int64") / 1e9,
        "resource":  (df[res_col].astype(str) if res_col else "").fillna(""),
        "status":    "running",
    })
    # Optional columns your code might read but not strictly need:
    out["end_time"]  = pd.NA
    out["cycle_time"] = pd.NA

    out = out.sort_values(["case_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    return out



def try_read(path):
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)

def load_variant_pkls(base_path: str, log: str, variant: str):
    """
    Returns (train_df, val_df, test_df) for a given variant.

    Rules:
      - STD  -> train/val/test from STD_*
      - GEN  -> train/val/test from GEN_*
      - else -> train/val from STD_*, test from <VARIANT>_test_prefix.pkl
    """
    base = Path(base_path)

    def pkl(suffix: str) -> Path:
        return base / f"{log}_{suffix}.pkl"

    if variant == "STD":
        tr = pkl("STD_train_prefix")
        va = pkl("STD_val_prefix")
        te = pkl("STD_test_prefix")
    elif variant == "GEN":
        tr = pkl("GEN_train_prefix")
        va = pkl("GEN_val_prefix")
        te = pkl("GEN_test_prefix")
    else:
        # TEST_* or any other test-only variant
        tr = pkl("STD_train_prefix")
        va = pkl("STD_val_prefix")
        te = pkl(f"{variant}_test_prefix")

    missing = [p.name for p in (tr, va, te) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files for {log} / {variant}: {missing}")

    tr_df = pd.read_pickle(tr)
    va_df = pd.read_pickle(va)
    te_df = pd.read_pickle(te)
    if len(te_df) == 0:
        raise ValueError(f"Empty test set for {log} / {variant}")

    return tr_df, va_df, te_df


# ---------------- config ----------------
n_runs = 1
k_last = 3
top_m_patterns = 2000
save_dir = "models_level3_xes"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

base_path = "dataset_xes"      # where the generated splits live
raw_xes_dir = "raw_dataset"    # where the .xes are

# choose which logs to run (file stem without .xes).
# you can also automatically grab all .xes:
logs = [p.stem for p in Path(raw_xes_dir).glob("*.xes")]

# ---------------- main loop ----------------
# Variants to run (from config; fallback to just STD if not provided)
variants = list(CFG_VARIANTS) if CFG_VARIANTS is not None else ["STD"]
print(f"Using variants: {variants}")

for log in logs:
    print(f"\n=== {log} ===")

    # vocab
    act_map_path = f"{base_path}/{log}_activity_to_int.p"
    if not os.path.exists(act_map_path):
        print(f"[WARN] Missing vocab for {log} at {act_map_path}. Skipping.")
        continue
    activity_to_int = pickle.load(open(act_map_path, "rb"))

    # df_log from XES
    xes_path = f"{raw_xes_dir}/{log}.xes"
    if not os.path.exists(xes_path):
        print(f"[WARN] Missing XES {xes_path}. Skipping.")
        continue
    df_log = load_df_log_from_xes(xes_path)

    for variant in variants:
        try:
            train_df, val_df, test_df = load_variant_pkls(base_path, log, variant)
        except Exception as e:
            print(f"[INFO] Skip {log} / {variant}: {e}")
            continue

        rows = []
        for run in range(1, n_runs + 1):
            print(f"Run {run}/{n_runs} — {log} [{variant}]")
            scenario_tag = f"{log}_{variant}_run{run}"
            stats = run_level3_xgboost_for_scenario(
                scenario=scenario_tag,
                activity_to_int=activity_to_int,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                df_log=df_log,
                k_last=k_last,
                top_m_patterns=top_m_patterns,
                save_dir=save_dir,
            )
            rows.append({
                "run": run,
                "val_mae": stats["val_mae"],
                "test_mae": stats["test_mae"],
                "n_features": stats["n_features"],
                "top_m_patterns": stats["top_m_patterns"],
                "unique_patterns_seen_train": stats["unique_patterns_seen_train"],
                "training_time_sec": stats["training_time_sec"],
            })

        out_csv = f"{results_dir}/LEVEL3_XGB_{log}_{variant}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
