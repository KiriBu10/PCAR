

# run_pgt_repoish_on_xes_splits.py
import os
import pickle
import pandas as pd
from pathlib import Path
import multiprocessing as mp

from config_train import VARIANTS as CFG_VARIANTS
from amiri2024 import run_pgt_net_repoish_for_scenario

import warnings
warnings.filterwarnings("ignore")

# IMPORTANT: if you have only one GPU, keep this at 1
NUM_PROCESSES = 8

# ---- optional XES -> event DataFrame (for df_log / workload) ----
def _try_build_df_log_from_xes(xes_path: Path):
    """
    Best-effort: build a df_log with columns ['case_id','timestamp','status','end_time'].
    Returns a (df_log or None).
    """
    try:
        from pm4py.objects.log.importer.xes import importer as xes_importer
        from pm4py.objects.conversion.log import converter as log_converter
    except Exception:
        return None

    if not xes_path.exists():
        return None
    try:
        log = xes_importer.apply(str(xes_path))
        df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
        case_col = "case:concept:name"
        time_col = "time:timestamp"
        out = pd.DataFrame()
        out["case_id"] = df[case_col].astype(str)
        out["timestamp"] = pd.to_datetime(df[time_col]).astype("int64") / 3_600_000_000_000.0  # ns->hours
        out["status"] = "running"
        out["end_time"] = pd.NA
        return out[["case_id","timestamp","status","end_time"]].sort_values(["case_id","timestamp"], kind="mergesort").reset_index(drop=True)
    except Exception:
        return None

def _load_activity_vocab(dataset_dir: Path, stem: str):
    p = dataset_dir / f"{stem}_activity_to_int.p"
    with open(p, "rb") as f:
        return pickle.load(f)

def _load_variant_split(dataset_dir: Path, stem: str, variant: str):
    """
    Returns (train_df, val_df, test_df) for any variant.
      - STD  -> train/val/test from STD_*
      - GEN  -> train/val/test from GEN_*
      - else -> train/val from STD_*, test from <VARIANT>_test_prefix.pkl
    """
    def pkl(suffix: str) -> Path:
        return dataset_dir / f"{stem}_{suffix}.pkl"

    if variant == "STD":
        tr, va, te = pkl("STD_train_prefix"), pkl("STD_val_prefix"), pkl("STD_test_prefix")
    elif variant == "GEN":
        tr, va, te = pkl("GEN_train_prefix"), pkl("GEN_val_prefix"), pkl("GEN_test_prefix")
    else:
        tr, va, te = pkl("STD_train_prefix"), pkl("STD_val_prefix"), pkl(f"{variant}_test_prefix")

    missing = [p.name for p in (tr, va, te) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files for {stem} / {variant}: {missing}")

    tr_df, va_df, te_df = pd.read_pickle(tr), pd.read_pickle(va), pd.read_pickle(te)
    if len(te_df) == 0:
        raise ValueError(f"Empty test set for {stem} / {variant}")
    return tr_df, va_df, te_df


def _build_df_log_for_workload(stem: str, dataset_dir: Path, raw_dir: Path):
    """
    Build df_log for workload computation:
      1) If raw XES exists & pm4py available -> parse it.
      2) Else: return an empty-but-well-formed DataFrame (workload -> 0).
    """
    xes_path = raw_dir / f"{stem}.xes"
    df_log = _try_build_df_log_from_xes(xes_path)
    if df_log is not None:
        return df_log
    return pd.DataFrame(columns=["case_id","timestamp","status","end_time"])


# -------- single task worker --------
def _run_one_pgt_task(stem: str, variant: str,
                      dataset_dir: str = "dataset_xes",
                      raw_dir: str = "raw_dataset",
                      save_dir: str = "models_pgt_xes",
                      results_dir: str = "results"):
    dataset_dir = Path(dataset_dir)
    raw_dir = Path(raw_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    label = f"{stem}_{variant}"
    out_csv = Path(results_dir) / f"PGT_{label}.csv"
    # if there’s already a non-empty CSV, skip the whole run
    if out_csv.exists() and out_csv.stat().st_size > 0:
        try:
            _df = pd.read_csv(out_csv)
            if len(_df) > 0:
                print(f"[SKIP] {label}: results already at {out_csv}")
                return {"stem": stem, "variant": variant, "status": "skipped", "csv": str(out_csv)}
        except Exception:
            # corrupt CSV? fall through and recompute
            pass


    try:
        activity_to_int = _load_activity_vocab(dataset_dir, stem)
        train_df, val_df, test_df = _load_variant_split(dataset_dir, stem, variant)
        df_log = _build_df_log_for_workload(stem, dataset_dir, raw_dir)

        stats = run_pgt_net_repoish_for_scenario(
            scenario=label,
            activity_to_int=activity_to_int,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            df_log=df_log,
            # model/training params (mirrors the YAML; no CV)
            d_model=64, num_heads=8, n_layers=5,
            k_lap=1, k_rw=10,
            pe_dropout=0.1, model_dropout=0.0, attn_dropout=0.5,
            batch_size=128, lr=1e-3, weight_decay=1e-5,
            max_epochs=200, patience=10,
            precompute_graphs=True,
            num_workers=0,
            use_amp=True, use_compile=False,
            save_dir=save_dir
        )

        out_csv = Path(results_dir) / f"PGT_{label}.csv"
        pd.DataFrame([{
            "stem": stem,
            "variant": variant,
            "val_mae": stats["val_mae"],
            "test_mae": stats["test_mae"],
            "training_time_sec": stats.get("training_time_sec", None),
            "model_path": stats.get("model_path", ""),
        }]).to_csv(out_csv, index=False)
        print(f"[OK] {label} -> {out_csv}")
        return {"stem": stem, "variant": variant, "status": "ok", "csv": str(out_csv)}
    except Exception as e:
        print(f"[SKIP] {label}: {e}")
        return {"stem": stem, "variant": variant, "status": f"error: {e}"}


def main():
    dataset_dir = Path("dataset_xes")
    raw_dir = Path("raw_dataset")
    save_dir = "models_pgt_xes"
    results_dir = "results"

    stems = sorted(set(
        p.name.replace("_STD_train_prefix.pkl", "")
        for p in dataset_dir.glob("*_STD_train_prefix.pkl")
    ))
    variants = list(CFG_VARIANTS) if CFG_VARIANTS is not None else ["STD", "GEN"]
    print(f"Using variants: {variants}")

    # tasks = [(stem, var, str(dataset_dir), str(raw_dir), save_dir, results_dir)
    #          for stem in stems for var in variants]
    all_tasks = [(stem, var, str(dataset_dir), str(raw_dir), save_dir, results_dir)
             for stem in stems for var in variants]

    tasks = []
    for stem, var, ds, rd, sd, rd_results in all_tasks:
        label = f"{stem}_{var}"
        out_csv = Path(results_dir) / f"PGT_{label}.csv"
        if out_csv.exists() and out_csv.stat().st_size > 0:
            try:
                _df = pd.read_csv(out_csv)
                if len(_df) > 0:
                    print(f"[SKIP enqueue] {label}: {out_csv} exists")
                    continue
            except Exception:
                pass
        tasks.append((stem, var, ds, rd, sd, rd_results))

    
    if not tasks:
        print("No tasks to run.")
        return

    with mp.Pool(processes=NUM_PROCESSES) as pool:
        summaries = pool.starmap(_run_one_pgt_task, tasks)
    
    print("\nAll tasks finished.")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    mp.freeze_support()
    main()
