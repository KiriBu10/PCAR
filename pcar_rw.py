# timeaware_xes_runner.py
import os, math, pickle, bisect
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# pm4py for XES parsing
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter

# import your model runner and dataset class from your existing script
# (assumes they're in the same process; otherwise change to a module import)
from cross_attention_method import run_experiments

from config_train import VARIANTS as CFG_VARIANTS
from concurrent.futures import ProcessPoolExecutor, as_completed

from framework.utils import attach_parallel_sequences_fast
import torch
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
except Exception:
    pass


NUM_WORKERS = 2 

def _resolve_split_files(stem: str, variant: str):
    """
    Returns (train_pkl, val_pkl, test_pkl) filenames for the given variant.

    Rules:
      - STD  -> train/val/test from STD_*
      - GEN  -> train/val/test from GEN_*
      - else -> train/val from STD_*, test from <VARIANT>_test_prefix.pkl
    """
    if variant == "STD":
        return (f"{stem}_STD_train_prefix.pkl",
                f"{stem}_STD_val_prefix.pkl",
                f"{stem}_STD_test_prefix.pkl")
    if variant == "GEN":
        return (f"{stem}_GEN_train_prefix.pkl",
                f"{stem}_GEN_val_prefix.pkl",
                f"{stem}_GEN_test_prefix.pkl")
    # TEST_* (or any other test-only variant)
    return (f"{stem}_STD_train_prefix.pkl",
            f"{stem}_STD_val_prefix.pkl",
            f"{stem}_{variant}_test_prefix.pkl")


# -------------------------------
# XES loader (must match preprocessing units)
# -------------------------------
def load_xes_to_df(xes_path: str) -> pd.DataFrame:
    xes_path = str(xes_path)
    log = xes_importer.apply(xes_path)
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

    case_col = "case:concept:name"
    act_col  = "concept:name"
    time_col = "time:timestamp"
    res_col  = None
    for c in ["org:resource", "Resource", "resource"]:
        if c in df.columns:
            res_col = c
            break
    if case_col not in df.columns or act_col not in df.columns or time_col not in df.columns:
        raise ValueError(f"Missing required XES columns in {xes_path}")

    proc = Path(xes_path).stem
    out = pd.DataFrame()
    out["process"] = proc
    out["case_id"] = df[case_col].astype(str)
    out["activity"] = df[act_col].astype(str)
    # IMPORTANT: same unit as your preprocessing script (HOURS)
    out["timestamp_abs"] = pd.to_datetime(df[time_col]).astype("int64") / 3_600_000_000_000
    out["resource"] = (df[res_col].astype(str) if res_col else "").fillna("")
    out["status"] = "running"
    out = out.sort_values(["case_id", "timestamp_abs"], kind="mergesort").reset_index(drop=True)
    return out

# -------------------------------
# Build a fast per-case index
# -------------------------------
def build_case_index(df_events: pd.DataFrame, activity_to_int: dict, process_to_int: dict):
    """
    Returns:
      case_meta: dict[case_id] -> {
          "process": str, "proc_int": int,
          "times_abs": np.ndarray[float],          # ascending
          "times_rel": np.ndarray[float],          # from case start
          "acts_int":  np.ndarray[int]
      }
      intervals: (case_ids_arr, starts_arr, ends_arr)
    """
    case_meta = {}
    g = df_events.groupby("case_id", sort=False)
    for cid, grp in g:
        grp = grp.sort_values("timestamp_abs")
        pr  = grp["process"].iloc[0]
        p_i = int(process_to_int.get(pr, 0))
        times_abs = grp["timestamp_abs"].to_numpy(dtype=float)
        times_rel = times_abs - times_abs[0]
        acts_int  = np.array([activity_to_int.get(a, 0) for a in grp["activity"].tolist()], dtype=np.int64)
        case_meta[cid] = dict(
            process=pr, proc_int=p_i,
            times_abs=times_abs, times_rel=times_rel, acts_int=acts_int
        )
    # intervals for fast concurrency lookup
    case_ids = np.array(list(case_meta.keys()))
    starts   = np.array([cm["times_abs"][0] for cm in case_meta.values()], dtype=float)
    ends     = np.array([cm["times_abs"][-1] for cm in case_meta.values()], dtype=float)
    return case_meta, (case_ids, starts, ends)

def concurrent_cases_at(t_abs: float, target_cid: str, intervals):
    case_ids, starts, ends = intervals
    m = (starts <= t_abs) & (ends > t_abs)
    if isinstance(target_cid, str):
        m = m & (case_ids != target_cid)
    else:
        # if target_cid is numeric-like in XES -> cast to str in our stored array
        m = m & (case_ids.astype(str) != str(target_cid))
    return case_ids[m]

# -------------------------------
# Build parallel sequences for each prefix row
# -------------------------------
# def attach_parallel_sequences(prefix_df: pd.DataFrame,
#                               case_meta: dict,
#                               intervals,
#                               max_parallel=10):
#     """
#     Adds a column 'parallel_sequences' to prefix_df.
#     Each item is a list of dicts:
#       {
#         'activity_sequence_int': List[int],
#         'timestamps_case_related': List[float],     # relative to that case
#         'process_int': int,
#         'process': str
#       }
#     Rule: a parallel case is any other case active at the prefix's absolute time.
#     We take that case's events up to t_abs (<=), not beyond.
#     """
#     rows = []
#     for _, r in tqdm(prefix_df.iterrows(), total=len(prefix_df), desc="attach_parallel_sequences"):
#         cid   = r["case_id"]
#         t_abs = r.get("prefix_abs_time", None)
#         if t_abs is None:
#             # rebuild absolute time from first rel timestamp + rel end
#             # (in your XES preprocessing, prefix_abs_time is always present)
#             raise ValueError("prefix_abs_time missing; regenerate prefixes from XES.")

#         # find concurrent cases
#         ocases = concurrent_cases_at(float(t_abs), cid, intervals)
#         par_list = []
#         for oc in ocases:
#             cm = case_meta[oc]
#             # cut at t_abs
#             idx = bisect.bisect_right(cm["times_abs"], t_abs) - 1
#             if idx < 0:  # shouldn't happen due to mask
#                 continue
#             acts = cm["acts_int"][:idx+1].tolist()
#             tt   = cm["times_rel"][:idx+1].astype(float).tolist()
#             if not acts:  # skip empties
#                 continue
#             par_list.append({
#                 "activity_sequence_int": acts,
#                 "timestamps_case_related": tt,
#                 "process_int": cm["proc_int"],
#                 "process": cm["process"],
#             })
#         # cap by recency: keep the ones whose last time is closest to t_abs
#         if len(par_list) > max_parallel:
#             par_list.sort(key=lambda d: -d["timestamps_case_related"][-1])  # most progressed first
#             par_list = par_list[:max_parallel]
#         rows.append(par_list)

#     out = prefix_df.copy()
#     out["parallel_sequences"] = rows
#     return out

# -------------------------------
# Runner over dataset_xes splits
# -------------------------------
def run_timeaware_on_xes(
    xes_name: str,
    variant: str,         # 'STD' | 'TEST_PAR' | 'TEST_NOPAR' | 'TEST_SAMERES_PAR' | 'GEN'
    dataset_dir="dataset_xes",
    raw_dir="raw_dataset",
    max_parallel=10,
    # your model hyperparams
    d_model=36, nhead=4, ff_dim=64, dropout=0.2,
    max_seq_len=50, batch_size=128, num_epochs=200, patience=10,
    n_runs=1,
):
    stem = Path(xes_name).stem
    out_csv = Path("results", f"TIMEAWARE_{stem}_{variant}.csv")
    if out_csv.exists() and out_csv.stat().st_size > 0:
        try:
            df = pd.read_csv(out_csv)
            if len(df) > 0:
                print(f"[SKIP] {stem} / {variant}: {out_csv} exists")
                return df   # or return f"SKIP:{out_csv}"
        except Exception:
            pass
    tr_p, va_p, te_p = _resolve_split_files(stem, variant)


    # load vocabs
    with open(Path(dataset_dir, f"{stem}_activity_to_int.p"), "rb") as f: activity_to_int = pickle.load(f)
    with open(Path(dataset_dir, f"{stem}_process_to_int.p"), "rb") as f: process_to_int  = pickle.load(f)

    # load splits
    train_df = pd.read_pickle(Path(dataset_dir, tr_p))
    val_df   = pd.read_pickle(Path(dataset_dir, va_p))
    test_df  = pd.read_pickle(Path(dataset_dir, te_p))

    # parse raw XES once and build case index
    df_events = load_xes_to_df(str(Path(raw_dir, f"{stem}.xes")))
    case_meta, intervals = build_case_index(df_events, activity_to_int, process_to_int)

    # attach parallel sequences
    # train_ps = attach_parallel_sequences(train_df, case_meta, intervals, max_parallel=max_parallel)
    # val_ps   = attach_parallel_sequences(val_df,   case_meta, intervals, max_parallel=max_parallel)
    # test_ps  = attach_parallel_sequences(test_df,  case_meta, intervals, max_parallel=max_parallel)
    # timeaware_xes_runner.py
    # replace:
    # train_ps = attach_parallel_sequences(train_df, case_meta, intervals, max_parallel=...)
    # with:
    train_ps = attach_parallel_sequences_fast(train_df, case_meta, intervals, max_parallel=max_parallel)
    val_ps   = attach_parallel_sequences_fast(val_df,   case_meta, intervals, max_parallel=max_parallel)
    test_ps  = attach_parallel_sequences_fast(test_df,  case_meta, intervals, max_parallel=max_parallel)

    # run your (unchanged) experiment loop
    stats = run_experiments(
        n_runs=n_runs,
        train_df_parallel_sequences=train_ps,
        val_df_parallel_sequences=val_ps,
        test_df_parallel_sequences=test_ps,
        process_to_int=process_to_int,
        save_path_prefix=f"models/timeaware_{stem}_{variant}.pt",
        d_model=d_model, nhead=nhead, ff_dim=ff_dim, dropout=dropout,
        max_seq_len=max_seq_len, max_parallel=max_parallel,
        batch_size=batch_size, num_epochs=num_epochs, patience=patience
    )
    out_csv = Path("results", f"TIMEAWARE_{stem}_{variant}.csv")
    stats.to_csv(out_csv, index=False)
    print(f"[DONE] {stem} / {variant} -> {out_csv}")
    return stats

# -------------------------------
# Example main
# -------------------------------
# if __name__ == "__main__":
#     os.makedirs("models", exist_ok=True)
#     os.makedirs("results", exist_ok=True)

#     # choose the logs and variants you want
#     logs = [p.name for p in Path("raw_dataset").glob("*.xes")]
#     variants = list(CFG_VARIANTS)

#     for log in logs:
#         for v in variants:
#             try:
#                 run_timeaware_on_xes(
#                     xes_name=log,
#                     variant=v,
#                     dataset_dir="dataset_xes",
#                     raw_dir="raw_dataset",
#                     n_runs=1,                    # set >1 to repeat
#                     d_model=36, nhead=4, ff_dim=64, dropout=0.2,
#                     max_seq_len=50, batch_size=128, num_epochs=200, patience=10,
#                     max_parallel=10
#                 )
#             except Exception as e:
#                 print(f"[SKIP] {log} / {v}: {e}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # choose the logs and variants you want
    logs = [p.name for p in Path("raw_dataset").glob("*.xes")]

    # Variants to run (from config; fallback to a sensible default)
    variants = list(CFG_VARIANTS) if CFG_VARIANTS is not None else ["STD", "GEN"]
    print(f"Using variants: {variants}")

    # How many parallel workers (processes)?
    # If NUM_WORKERS is not set, default to min(cpu_count, #tasks)
    #tasks = [(log, v) for log in logs for v in variants]  # current: all pairs
    all_tasks = [(log, v) for log in logs for v in variants]
    tasks = []
    for log, v in all_tasks:
        stem = Path(log).stem
        out_csv = Path("results", f"TIMEAWARE_{stem}_{v}.csv")
        if out_csv.exists() and out_csv.stat().st_size > 0:
            print(f"[SKIP enqueue] {stem} / {v}: {out_csv} exists")
            continue
        tasks.append((log, v))

    if not tasks:
        print("Nothing to do: all results already exist.")
        raise SystemExit(0)

    # or per single log:
    # tasks = [(logs[0], v) for v in variants]
    if not tasks:
        print("No logs/variants to run.")
        raise SystemExit(0)

    #cpu_cnt = os.cpu_count() or 1
    #max_workers = int(CFG_NUM_WORKERS) if CFG_NUM_WORKERS else min(cpu_cnt, len(tasks))
    #print(f"Launching up to {max_workers} parallel workers over {len(tasks)} tasks.")

    # Common hyperparams (adjust as needed)
    common_kwargs = dict(
        dataset_dir="dataset_xes",
        raw_dir="raw_dataset",
        n_runs=1,                    # set >1 to repeat
        d_model=36, nhead=4, ff_dim=64, dropout=0.2,
        max_seq_len=50, batch_size=128, num_epochs=200, patience=10,
        max_parallel=10
    )

    # NOTE: If training uses a single GPU, set NUM_WORKERS=1 to avoid contention.
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        fut2task = {
            ex.submit(run_timeaware_on_xes, xes_name=log, variant=v, **common_kwargs): (log, v)
            for (log, v) in tasks
        }
        for fut in as_completed(fut2task):
            log, v = fut2task[fut]
            try:
                _ = fut.result()
            except Exception as e:
                print(f"[SKIP] {log} / {v}: {e}")
            else:
                print(f"[DONE] {log} / {v}")
