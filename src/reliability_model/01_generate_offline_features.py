from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import config
from main_model_inference import (
    MyPredictionModel,
    auto_dataset_paths,
    build_reference_info,
    discover_scaffold_checkpoints,
    log,
    log_error_target,
    smooth_loss_weights,
    predict_csv_with_checkpoint,
    safe_float,
)
from data_utils import extract_features_for_error_model, records_from_dataframe, standardize_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline AE feature CSV from 5-fold checkpoints.")
    parser.add_argument("--split", choices=["scaffold", "random", "ae"], default="ae")
    parser.add_argument("--best_dir", default="Best")
    parser.add_argument("--split_dir", default="../../data/splits/deployment")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument(
        "--feature_n_jobs",
        type=int,
        default=None,
        help="Parallel workers for RDKit/offline feature extraction. Use -1 for all CPU threads.",
    )
    parser.add_argument(
        "--feature_chunk_size",
        type=int,
        default=None,
        help="Rows per chunk for solute feature extraction.",
    )
    parser.add_argument(
        "--feature_backend",
        choices=["threading", "process"],
        default=None,
        help="Parallel backend for solute feature extraction. process uses more CPU and memory.",
    )
    return parser.parse_args()


def resolve_feature_n_jobs(feature_n_jobs: int | None = None) -> int:
    value = config.AE_FEATURE_N_JOBS if feature_n_jobs is None else int(feature_n_jobs)
    if value < 0:
        return max(1, (os.cpu_count() or 1) + 1 + value)
    return max(1, value)


def resolve_feature_chunk_size(feature_chunk_size: int | None = None) -> int:
    value = config.AE_FEATURE_CHUNK_SIZE if feature_chunk_size is None else int(feature_chunk_size)
    return max(1, value)


def resolve_feature_backend(feature_backend: str | None = None) -> str:
    value = config.AE_FEATURE_BACKEND if feature_backend is None else str(feature_backend)
    if value not in {"threading", "process"}:
        raise ValueError(f"Unsupported feature backend: {value}")
    return value


def fold_number(path: Path, split: str) -> int:
    import re

    pattern = r"fold[_ -]*(\d+)"
    match = re.search(pattern, str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 9999


def discover_checkpoints(best_dir: str, split: str, expected: int = 5) -> List[str]:
    root = Path(best_dir)
    if not root.exists():
        root = Path(__file__).resolve().parent / best_dir
    root = root.resolve()
    label_by_split = {
        "scaffold": "Scaffold",
        "random": "Random",
        "ae": "fold",
    }
    label = label_by_split.get(split)
    if label is None:
        raise ValueError(f"Unsupported split={split!r}.")
    paths = sorted(root.glob(f"{label}_*/best_model.pt"), key=lambda p: (fold_number(p, split), str(p)))
    if len(paths) != expected:
        found = "\n".join(f"  - {p}" for p in paths) or "  <none>"
        raise ValueError(f"Expected exactly {expected} {label} checkpoints under {root}, found {len(paths)}.\n{found}")
    return [str(p.resolve()) for p in paths]


def ensure_random_csvs_available(split_dir: str) -> None:
    """Require the deployment table and its fixed test table."""
    base = Path(split_dir)
    needed = [base / "deployment.csv", base / "deployment_test.csv"]
    if all(path.exists() for path in needed):
        return
    missing = "\n".join(f"  - {path}" for path in needed if not path.exists())
    raise FileNotFoundError(
        "Missing required unified AE split CSV files.\n"
        "Expected deployment.csv with cv_fold plus deployment_test.csv.\n"
        f"Missing in {base.resolve()}:\n{missing}"
    )


def reference_cache_path(split: str, fold_idx: int) -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "intermediate" / f"reference_cache_{split}_fold{fold_idx}.pkl"


def load_or_build_reference_info(
    *,
    split: str,
    fold_idx: int,
    checkpoint_path: str,
    reference_csv: str,
    cv_fold: int | None = None,
) -> Dict:
    path = reference_cache_path(split, fold_idx)
    if path.exists():
        log(f"[Offline Features] reusing reference cache: {path.name}")
        with open(path, "rb") as f:
            item = pickle.load(f)
        reference_info = item.get("reference_info", item)
        reference_info["solvent_cache"] = None
        return reference_info

    log(f"[Offline Features] building reference fold {fold_idx}")
    reference_info = build_reference_info(
        reference_csv,
        cv_fold=cv_fold,
        fold_role="train" if cv_fold is not None else None,
    )
    reference_info["solvent_cache"] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    item = {
        "version": "flair_v3_reference_cache",
        "split": split,
        "fold": int(fold_idx),
        "checkpoint_path": checkpoint_path,
        "reference_csv": reference_csv,
        "reference_info": reference_info,
    }
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(item, f)
    os.replace(tmp_path, path)
    log(f"[Offline Features] saved reference cache: {path.resolve()}")
    return reference_info


def _extract_shared_solute_one(task):
    idx, rec, solute_cache = task
    features, names = extract_features_for_error_model(
        solute_mol=rec.get("solute_mol"),
        solute_cache=solute_cache,
        solvent_mol=None,
        solvent_cache=None,
        fold_variance=0.0,
        base_prediction=None,
    )
    return idx, features, names


_PROCESS_SOLUTE_CACHE = None


def _init_process_solute_cache(solute_cache):
    global _PROCESS_SOLUTE_CACHE
    _PROCESS_SOLUTE_CACHE = solute_cache


def _extract_shared_solute_one_process(task):
    idx, rec = task
    features, names = extract_features_for_error_model(
        solute_mol=rec.get("solute_mol"),
        solute_cache=_PROCESS_SOLUTE_CACHE,
        solvent_mol=None,
        solvent_cache=None,
        fold_variance=0.0,
        base_prediction=None,
    )
    return idx, features, names


def build_shared_solute_feature_frame(
    pred_df: pd.DataFrame,
    reference_info: Dict,
    feature_n_jobs: int | None = None,
    feature_chunk_size: int | None = None,
    feature_backend: str | None = None,
) -> tuple[pd.DataFrame, List[str]]:
    """Compute slow solute distribution features once per sample.

    Property-specific columns such as base_prediction and AE are appended later.
    """
    pred_df = standardize_columns(pred_df.copy())
    records = records_from_dataframe(pred_df)
    n_jobs = resolve_feature_n_jobs(feature_n_jobs)
    chunk_size = resolve_feature_chunk_size(feature_chunk_size)
    backend = resolve_feature_backend(feature_backend)
    X = []
    used_rows = []
    feature_names = None

    log(
        "[Offline Features] shared solute features: "
        f"rows={len(records)}, n_jobs={n_jobs}, chunk_size={chunk_size}, backend={backend}"
    )
    tasks = [(idx, rec, reference_info["solute_cache"]) for idx, rec in enumerate(records)]
    total = len(tasks)
    done = 0

    executor = None
    if n_jobs > 1 and total > 1:
        if backend == "process":
            executor = ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=_init_process_solute_cache,
                initargs=(reference_info["solute_cache"],),
            )
        else:
            executor = ThreadPoolExecutor(max_workers=n_jobs)

    try:
        for chunk_start in range(0, total, chunk_size):
            chunk = tasks[chunk_start : chunk_start + chunk_size]
            chunk_id = chunk_start // chunk_size + 1
            n_chunks = (total + chunk_size - 1) // chunk_size
            log(
                "[Offline Features] shared solute chunk "
                f"{chunk_id}/{n_chunks}: rows {chunk_start + 1}-{chunk_start + len(chunk)}"
            )

            if executor is None:
                iterator = map(_extract_shared_solute_one, chunk)
            elif backend == "process":
                process_tasks = [(idx, rec) for idx, rec, _ in chunk]
                map_chunk_size = max(1, len(process_tasks) // max(1, n_jobs * 4))
                iterator = executor.map(
                    _extract_shared_solute_one_process,
                    process_tasks,
                    chunksize=map_chunk_size,
                )
            else:
                iterator = executor.map(_extract_shared_solute_one, chunk)

            for idx, features, names in iterator:
                X.append(features)
                used_rows.append(idx)
                feature_names = names
                done += 1
            log(f"[Offline Features] shared solute features: {done}/{total}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if not X:
        raise ValueError("No shared solute feature rows were generated.")

    feature_df = pd.DataFrame(np.vstack(X), columns=list(feature_names or []))
    sub = pred_df.iloc[used_rows].reset_index(drop=True)
    shared = pd.concat(
        [
            pd.DataFrame({"row_index": np.asarray(used_rows, dtype=int)}),
            sub.reset_index(drop=True),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )
    return shared, list(feature_names or [])


def rows_for_property_from_shared(
    *,
    shared_df: pd.DataFrame,
    feature_names: List[str],
    prop: str,
    fold_idx: int,
    split_name: str,
    source_csv: str,
    checkpoint_path: str,
) -> pd.DataFrame:
    pred_col = f"pred_{prop}"
    true_col = f"true_{prop}" if f"true_{prop}" in shared_df.columns else prop

    sub = shared_df.copy()
    y_true = pd.to_numeric(sub[true_col], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(sub[pred_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    sub = sub.iloc[np.where(valid)[0]].reset_index(drop=True)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    abs_error = np.abs(y_pred - y_true)
    sample_weight = smooth_loss_weights(abs_error, prop)

    feature_df = sub[feature_names].reset_index(drop=True)
    feature_df["base_prediction"] = y_pred
    out = pd.DataFrame(
        {
            "fold": int(fold_idx),
            "split": split_name,
            "source_csv": source_csv,
            "checkpoint_path": checkpoint_path,
            "row_index": sub["row_index"].to_numpy(dtype=int),
            config.SMILES_COLUMN: sub.get(config.SMILES_COLUMN, pd.Series([""] * len(sub))).astype(str).to_numpy(),
            config.SOLVENT_COLUMN: sub.get(config.SOLVENT_COLUMN, pd.Series([""] * len(sub))).astype(str).to_numpy(),
            "property": prop,
            "y_true": y_true,
            "y_pred": y_pred,
            "ae": abs_error,
            "true_ae": abs_error,
            "abs_error": abs_error,
            "log_abs_error": log_error_target(abs_error, prop),
            "sample_weight": sample_weight,
            "epsilon": safe_float(config.ERROR_EPSILON_BY_PROPERTY, prop, 1e-8),
        }
    )
    return pd.concat([out, feature_df.reset_index(drop=True)], axis=1)


def rows_for_all_properties(
    *,
    pred_df: pd.DataFrame,
    reference_info: Dict,
    fold_idx: int,
    split_name: str,
    source_csv: str,
    checkpoint_path: str,
    feature_n_jobs: int | None = None,
    feature_chunk_size: int | None = None,
    feature_backend: str | None = None,
) -> List[pd.DataFrame]:
    shared_df, feature_names = build_shared_solute_feature_frame(
        pred_df,
        reference_info,
        feature_n_jobs=feature_n_jobs,
        feature_chunk_size=feature_chunk_size,
        feature_backend=feature_backend,
    )
    rows = []
    for prop in config.PROPERTIES:
        log(f"[Offline Features] property rows fold {fold_idx} {split_name} {prop}")
        rows.append(
            rows_for_property_from_shared(
                shared_df=shared_df,
                feature_names=feature_names,
                prop=prop,
                fold_idx=fold_idx,
                split_name=split_name,
                source_csv=source_csv,
                checkpoint_path=checkpoint_path,
            )
        )
    return rows


def generate_offline_features(
    *,
    split: str = "scaffold",
    best_dir: str = "Best",
    split_dir: str = ".",
    output_csv: str | None = None,
    summary_json: str | None = None,
    feature_n_jobs: int | None = None,
    feature_chunk_size: int | None = None,
    feature_backend: str | None = None,
) -> Dict:
    os.chdir(Path(__file__).resolve().parent)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if split in {"random", "ae"}:
        ensure_random_csvs_available(split_dir)

    output_feature_csv = Path(output_csv or f"results/offline_ae_features_{split}.csv")
    output_summary_json = Path(summary_json or f"results/offline_feature_generation_summary_{split}.json")
    checkpoint_paths = discover_checkpoints(best_dir, split=split, expected=5)
    resolved_feature_n_jobs = resolve_feature_n_jobs(feature_n_jobs)
    resolved_feature_chunk_size = resolve_feature_chunk_size(feature_chunk_size)
    resolved_feature_backend = resolve_feature_backend(feature_backend)
    all_feature_frames: List[pd.DataFrame] = []
    summary = {
        "version": f"flair_v3_offline_ae_features_{split}",
        "split": split,
        "feature_n_jobs": resolved_feature_n_jobs,
        "feature_chunk_size": resolved_feature_chunk_size,
        "feature_backend": resolved_feature_backend,
        "checkpoint_paths": checkpoint_paths,
        "splits": [],
        "output_feature_csv": str(output_feature_csv.resolve()),
    }

    log(f"[Offline Features] start split={split}")
    log(
        "[Offline Features] feature parallelism: "
        f"n_jobs={resolved_feature_n_jobs}, chunk_size={resolved_feature_chunk_size}, "
        f"backend={resolved_feature_backend}"
    )
    for fold_idx, checkpoint_path in enumerate(checkpoint_paths, start=1):
        log(f"[Offline Features] fold {fold_idx}/5")
        paths = auto_dataset_paths(checkpoint_path, extra_roots=[split_dir, "."])
        summary["splits"].append(
            {
                "fold": fold_idx,
                **paths,
                "reference_cache": str(reference_cache_path(split, fold_idx).resolve()),
            }
        )

        split_specs = [
            ("calibration", paths["calibration_csv"]),
            ("test", paths["ae_test_csv"]),
        ]

        fold_feature_paths = [
            results_dir / f"offline_feature_rows_{split}_fold{fold_idx}_{split_name}.csv"
            for split_name, csv_path in split_specs
            if csv_path and Path(csv_path).exists()
        ]
        if fold_feature_paths and all(path.exists() for path in fold_feature_paths):
            for path in fold_feature_paths:
                log(f"[Offline Features] reusing feature rows: {path.name}")
                all_feature_frames.append(pd.read_csv(path))
            continue

        log(f"[Offline Features] loading model fold {fold_idx}")
        model = MyPredictionModel.load(
            checkpoint_path,
            train_csv_override=paths["reference_csv"],
            cv_fold_override=paths["reference_fold"],
        )

        reference_info = load_or_build_reference_info(
            split=split,
            fold_idx=fold_idx,
            checkpoint_path=checkpoint_path,
            reference_csv=paths["reference_csv"],
            cv_fold=paths["reference_fold"],
        )

        for split_name, csv_path in split_specs:
            if not csv_path or not Path(csv_path).exists():
                continue
            feature_rows_csv = results_dir / f"offline_feature_rows_{split}_fold{fold_idx}_{split_name}.csv"
            if feature_rows_csv.exists():
                log(f"[Offline Features] reusing feature rows: {feature_rows_csv.name}")
                all_feature_frames.append(pd.read_csv(feature_rows_csv))
                continue

            log(f"[Offline Features] predicting fold {fold_idx} {split_name}: {Path(csv_path).name}")
            prediction_csv = results_dir / f"offline_predictions_fold{fold_idx}_{split_name}.csv"
            if prediction_csv.exists():
                log(f"[Offline Features] reusing predictions: {prediction_csv.name}")
                pred_df = pd.read_csv(prediction_csv)
            else:
                pred_df = predict_csv_with_checkpoint(
                    model,
                    csv_path,
                    output_csv=str(prediction_csv),
                    batch_size=256,
                    cv_fold=paths["calibration_fold"] if split_name == "calibration" else None,
                    fold_role="valid" if split_name == "calibration" and paths["calibration_fold"] is not None else None,
                )

            split_feature_frames = rows_for_all_properties(
                pred_df=pred_df,
                reference_info=reference_info,
                fold_idx=fold_idx,
                split_name=split_name,
                source_csv=csv_path,
                checkpoint_path=checkpoint_path,
                feature_n_jobs=resolved_feature_n_jobs,
                feature_chunk_size=resolved_feature_chunk_size,
                feature_backend=resolved_feature_backend,
            )
            split_feature_df = pd.concat(split_feature_frames, ignore_index=True)
            split_feature_df.to_csv(feature_rows_csv, index=False)
            all_feature_frames.append(split_feature_df)

    if not all_feature_frames:
        raise ValueError("No offline AE feature rows were generated.")

    feature_df = pd.concat(all_feature_frames, ignore_index=True)
    output_feature_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_feature_csv, index=False)
    summary["n_rows"] = int(len(feature_df))
    summary["columns"] = list(feature_df.columns)

    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"[Offline Features] saved: {output_feature_csv.resolve()}")
    return summary


if __name__ == "__main__":
    args = parse_args()
    result = generate_offline_features(
        split=args.split,
        best_dir=args.best_dir,
        split_dir=args.split_dir,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        feature_n_jobs=args.feature_n_jobs,
        feature_chunk_size=args.feature_chunk_size,
        feature_backend=args.feature_backend,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
