from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

import importlib

from main_model_inference import MyPredictionModel, auto_dataset_paths, log, predict_csv_with_checkpoint
from data_utils import (
    FEATURE_NAMES_SOLUTE,
    TANIMOTO_NEIGHBOR_FEATURE_NAMES,
    build_unique_tanimoto_reference,
    get_morgan_fingerprint,
    get_scaffold_smiles,
    mol_from_smiles,
    scaffold_novelty,
    scaffold_similarity,
    standardize_columns,
    tanimoto_neighborhood_features,
)
from rank_ensemble_features import FINAL_FEATURE_COLUMNS, NEIGHBOR_FEATURE_COLUMNS, engineer_rank_ensemble_features

offline = importlib.import_module("01_generate_offline_features")


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
RESULTS_DIR = ROOT / "results" / "intermediate"
DEFAULT_AE_PRE_NAMES = ["data/source/ae_pre_holdout_candidate_pool.csv"]
STD_RENAME = {
    "std_smiles": "smiles",
    "std_solvent": "solvent",
    "std_abs_nm": "abs",
    "std_emi_nm": "emi",
    "std_plqy": "plqy",
    "std_log_epsilon_or_epsilon": "em",
}
CORE_COLUMNS = ["smiles", "solvent", "abs", "emi", "plqy", "em"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed grouped held-out split, then fit/transform leakage-safe AE features."
    )
    parser.add_argument("--split", choices=["ae"], default="ae")
    parser.add_argument("--best_dir", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument(
        "--results_dir",
        default=str(RESULTS_DIR),
        help="Directory for split files and intermediate artifacts; useful for isolated sensitivity experiments.",
    )
    parser.add_argument("--current_test_csv", default=None)
    parser.add_argument("--ae_pre_test_csv", default=None)
    parser.add_argument(
        "--calibration_output_csv",
        default=str(RESULTS_DIR / "calibration_features_ae_pre.csv"),
    )
    parser.add_argument(
        "--test_features_output_csv",
        default=str(RESULTS_DIR / "test_features_ae_pre.csv"),
    )
    parser.add_argument(
        "--test_labels_output_csv",
        default=str(RESULTS_DIR / "test_labels_ae_pre.csv"),
    )
    parser.add_argument("--summary_json", default=str(RESULTS_DIR / "offline_feature_generation_summary_ae_pre.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--feature_n_jobs", type=int, default=8)
    parser.add_argument("--feature_backend", choices=["threading", "process"], default="threading")
    parser.add_argument("--feature_chunk_size", type=int, default=512)
    parser.add_argument(
        "--rebuild_merged_split",
        action="store_true",
        help="Rebuild the canonical solute-solvent grouped held-out split from the two source CSVs.",
    )
    parser.add_argument(
        "--reuse_intermediate_cache",
        action="store_true",
        help="Reuse prediction/feature caches. Off by default to prevent stale-input or stale-checkpoint reuse.",
    )
    return parser.parse_args()


def split_labels(test_fraction: float) -> Tuple[str, str]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test_fraction must be in (0, 1).")
    test_percent = int(round(test_fraction * 100))
    if not np.isclose(test_fraction, test_percent / 100.0):
        raise ValueError("--test_fraction must be an integer percentage, for example 0.15 or 0.20.")
    return f"calibration{100 - test_percent}", f"test{test_percent}"


def resolve_path(path: str | None, *, default: Path | None = None, required: bool = True) -> Path | None:
    if path is None:
        if default is None:
            return None
        candidate = default
    else:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
    candidate = candidate.resolve()
    if required and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def discover_ae_pre_test(path: str | None) -> Path:
    if path:
        return resolve_path(path, required=True)  # type: ignore[return-value]
    for name in DEFAULT_AE_PRE_NAMES:
        candidate = ROOT / name
        if candidate.exists():
            return candidate.resolve()
    expected = "\n".join(f"  - {ROOT / name}" for name in DEFAULT_AE_PRE_NAMES)
    raise FileNotFoundError("Missing AE PRE test CSV. Expected one of:\n" + expected)


def standardize_input_csv(path: Path, source_group: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={src: dst for src, dst in STD_RENAME.items() if src in df.columns})
    lower_to_col = {str(col).lower(): col for col in df.columns}
    rename = {}
    for col in CORE_COLUMNS:
        if col not in df.columns and col.lower() in lower_to_col:
            rename[lower_to_col[col.lower()]] = col
    if rename:
        df = df.rename(columns=rename)
    missing = [col for col in ["smiles", "solvent"] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns after standardization: {missing}")
    for prop in ["abs", "emi", "plqy", "em"]:
        if prop not in df.columns:
            df[prop] = np.nan
    out = df.copy()
    out["source_group"] = source_group
    out["source_file"] = str(path.resolve())
    out["_ae_pre_source_row"] = np.arange(len(out), dtype=int)
    return out


def file_signature(path: Path) -> Dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def canonical_structure(value: object) -> str:
    raw = "" if pd.isna(value) else str(value).strip()
    if not raw:
        return ""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(raw)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else f"RAW::{raw}"


def canonical_pair_keys(df: pd.DataFrame) -> pd.Series:
    cache: Dict[str, str] = {}

    def cached(value: object) -> str:
        raw = "" if pd.isna(value) else str(value).strip()
        if raw not in cache:
            cache[raw] = canonical_structure(raw)
        return cache[raw]

    solute = df["smiles"].map(cached)
    solvent = df["solvent"].map(cached)
    return solute + "\x1f" + solvent


def validate_merged_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    calibration_label: str,
    test_label: str,
    reference_df: pd.DataFrame | None = None,
) -> Dict[str, int | float]:
    required = {
        "smiles",
        "solvent",
        "source_group",
        "_ae_pre_source_row",
        "_ae_pre_merged_row",
        "ae_pre_split",
    }
    for name, frame, expected_split in [
        (calibration_label, train_df, calibration_label),
        (test_label, test_df, test_label),
    ]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Existing {name} file is missing required columns: {missing}")
        if frame.empty:
            raise ValueError(f"Existing {name} file is empty.")
        if set(frame["ae_pre_split"].astype(str)) != {expected_split}:
            raise ValueError(f"Existing {name} file has invalid ae_pre_split values.")
        if frame["_ae_pre_merged_row"].duplicated().any():
            raise ValueError(f"Existing {name} file has duplicate _ae_pre_merged_row values.")

    train_rows = set(train_df["_ae_pre_merged_row"].astype(int))
    test_rows = set(test_df["_ae_pre_merged_row"].astype(int))
    if train_rows & test_rows:
        raise ValueError("Existing merged split has overlapping _ae_pre_merged_row values.")

    train_source_rows = set(zip(train_df["source_group"], train_df["_ae_pre_source_row"].astype(int)))
    test_source_rows = set(zip(test_df["source_group"], test_df["_ae_pre_source_row"].astype(int)))
    if train_source_rows & test_source_rows:
        raise ValueError("Existing merged split has source rows in both calibration and test.")

    train_pairs = set(canonical_pair_keys(train_df))
    test_pairs = set(canonical_pair_keys(test_df))
    overlap = train_pairs & test_pairs
    if overlap:
        raise ValueError(
            "Existing merged split leaks canonical solute-solvent groups across calibration and test: "
            f"{len(overlap)} overlapping groups."
        )
    reference_overlap = 0
    if reference_df is not None:
        reference_pairs = set(canonical_pair_keys(standardize_columns(reference_df.copy())))
        reference_overlap = len((train_pairs | test_pairs) & reference_pairs)
        if reference_overlap:
            raise ValueError(
                "Merged reliability data overlaps TrainAndVal canonical solute-solvent groups: "
                f"{reference_overlap} groups."
            )
    total = len(train_df) + len(test_df)
    return {
        "canonical_pair_overlap": 0,
        "train_and_val_canonical_pair_overlap": int(reference_overlap),
        "unique_canonical_pairs": int(len(train_pairs | test_pairs)),
        "actual_test_fraction": float(len(test_df) / total),
    }


def write_merged_split_files(
    *,
    current_test_csv: Path,
    ae_pre_test_csv: Path,
    out_dir: Path,
    reference_csv: Path,
    seed: int,
    test_fraction: float,
) -> Tuple[Path, Path, Dict]:
    calibration_label, test_label = split_labels(test_fraction)
    current = standardize_input_csv(current_test_csv, "current_test")
    ae_pre = standardize_input_csv(ae_pre_test_csv, "ae_pre_test")
    merged = pd.concat([current, ae_pre], ignore_index=True)
    property_values = merged[["abs", "emi", "plqy", "em"]].apply(pd.to_numeric, errors="coerce")
    usable_mask = property_values.notna().any(axis=1)
    dropped_no_target = int((~usable_mask).sum())
    merged = merged.loc[usable_mask].reset_index(drop=True)
    merged["_canonical_pair_key"] = canonical_pair_keys(merged)
    reference_df = standardize_columns(pd.read_csv(reference_csv))
    reference_pairs = set(canonical_pair_keys(reference_df))
    reference_overlap_mask = merged["_canonical_pair_key"].isin(reference_pairs)
    dropped_reference_overlap = int(reference_overlap_mask.sum())
    merged = merged.loc[~reference_overlap_mask].reset_index(drop=True)
    merged["_ae_pre_merged_row"] = np.arange(len(merged), dtype=int)

    rng = np.random.default_rng(seed)
    group_sizes = merged["_canonical_pair_key"].value_counts(sort=False)
    group_names = group_sizes.index.to_numpy(dtype=object)
    rng.shuffle(group_names)
    target_test_rows = int(round(len(merged) * test_fraction))
    selected_groups = []
    selected_rows = 0
    for group in group_names:
        group_size = int(group_sizes.loc[group])
        if selected_rows < target_test_rows or abs(selected_rows + group_size - target_test_rows) < abs(
            selected_rows - target_test_rows
        ):
            selected_groups.append(group)
            selected_rows += group_size
    test_mask = merged["_canonical_pair_key"].isin(selected_groups).to_numpy(dtype=bool)

    train_df = merged.loc[~test_mask].drop(columns=["_canonical_pair_key"]).reset_index(drop=True)
    test_df = merged.loc[test_mask].drop(columns=["_canonical_pair_key"]).reset_index(drop=True)
    train_df["ae_pre_split"] = calibration_label
    test_df["ae_pre_split"] = test_label
    validation = validate_merged_split(
        train_df,
        test_df,
        calibration_label=calibration_label,
        test_label=test_label,
        reference_df=reference_df,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"AE_PRE_MERGED_{calibration_label}.csv"
    test_path = out_dir / f"AE_PRE_MERGED_{test_label}.csv"
    train_tmp = train_path.with_suffix(".csv.tmp")
    test_tmp = test_path.with_suffix(".csv.tmp")
    train_df.to_csv(train_tmp, index=False)
    test_df.to_csv(test_tmp, index=False)
    os.replace(train_tmp, train_path)
    os.replace(test_tmp, test_path)

    summary = {
        "split_provenance": "generated_from_current_sources",
        "split_strategy": "canonical_solute_solvent_group_random",
        "current_test_csv": str(current_test_csv.resolve()),
        "ae_pre_test_csv": str(ae_pre_test_csv.resolve()),
        "current_test_signature": file_signature(current_test_csv),
        "ae_pre_test_signature": file_signature(ae_pre_test_csv),
        "train_and_val_signature": file_signature(reference_csv),
        "merged_rows": int(len(merged)),
        "dropped_rows_without_any_target": dropped_no_target,
        "dropped_rows_overlapping_train_and_val": dropped_reference_overlap,
        "calibration_label": calibration_label,
        "test_label": test_label,
        "calibration_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "seed": int(seed),
        "test_fraction": float(test_fraction),
        **validation,
        "source_group_counts": {str(k): int(v) for k, v in merged["source_group"].value_counts().items()},
        "calibration_source_group_counts": {
            str(k): int(v) for k, v in train_df["source_group"].value_counts().items()
        },
        "test_source_group_counts": {str(k): int(v) for k, v in test_df["source_group"].value_counts().items()},
    }
    manifest_path = out_dir / "AE_PRE_MERGED_split_manifest.json"
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    return train_path, test_path, summary


def write_train_and_val_reference(split_dir: str, out_dir: Path) -> Path:
    split_root = Path(split_dir)
    deployment_path = split_root / "deployment.csv"
    if not deployment_path.exists():
        raise FileNotFoundError(f"Missing deployment dataset: {deployment_path}")
    return deployment_path.resolve()


def build_reference_info_once(reference_csv: Path, *, reuse_cache: bool) -> Dict:
    cache_path = ROOT / "results" / "intermediate" / "reference_cache_ae_train_and_val.pkl"
    if reuse_cache and cache_path.exists():
        import pickle

        with open(cache_path, "rb") as f:
            item = pickle.load(f)
        reference_info = item.get("reference_info", item)
        solute_cache = reference_info.get("solute_cache", {})
        if (
            item.get("version") == "reference_cache_ae_train_and_val_v3"
            and "neighbor_fps" in solute_cache
            and "neighbor_scaffolds" in solute_cache
        ):
            log(f"[AE_PRE] reusing TrainAndVal reference cache: {cache_path.name}")
            reference_info["solvent_cache"] = None
            return reference_info
        log(f"[AE_PRE] rebuilding stale TrainAndVal reference cache: {cache_path.name}")

    log(f"[AE_PRE] building fast TrainAndVal reference cache from {reference_csv.name}")
    ref_df = standardize_columns(pd.read_csv(reference_csv))
    mols = [mol_from_smiles(smi) for smi in ref_df["smiles"].astype(str).tolist()]
    valid_mols = [mol for mol in mols if mol is not None]
    if not valid_mols:
        raise ValueError(f"No valid molecules in {reference_csv}")
    neighbor_smiles_column = "_canonical_smiles" if "_canonical_smiles" in ref_df.columns else "smiles"
    neighbor_fps, neighbor_scaffolds = build_unique_tanimoto_reference(ref_df[neighbor_smiles_column])
    fps = neighbor_fps
    scaffold_set = set()
    scaffold_fps = []
    scaffold_counts = {}
    for mol in valid_mols:
        scaf = get_scaffold_smiles(mol)
        if scaf:
            scaffold_counts[scaf] = scaffold_counts.get(scaf, 0) + 1
            if scaf not in scaffold_set:
                scaffold_set.add(scaf)
                scaffold_fps.append(get_morgan_fingerprint(mol_from_smiles(scaf)))
    reference_info = {
        "reference_csv": str(reference_csv.resolve()),
        "records": [],
        "solute_cache": {
            "fast_reference": True,
            "mols": valid_mols,
            "fps": fps,
            "scaffold_set": scaffold_set,
            "scaffold_fps": scaffold_fps,
            "scaffold_counts": scaffold_counts,
            "neighbor_fps": neighbor_fps,
            "neighbor_scaffolds": neighbor_scaffolds,
        },
        "solvent_cache": None,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "version": "reference_cache_ae_train_and_val_v3",
                "reference_csv": str(reference_csv.resolve()),
                "reference_info": reference_info,
            },
            f,
        )
    return reference_info


def shared_cache_path(results_dir: Path, split_name: str) -> Path:
    return results_dir / f"offline_shared_solute_features_ae_pre_{split_name}.csv"


def load_or_build_shared_features(
    *,
    pred_df: pd.DataFrame,
    reference_info: Dict,
    split_name: str,
    results_dir: Path,
    feature_n_jobs: int,
    feature_chunk_size: int,
    feature_backend: str,
    reuse_cache: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    path = shared_cache_path(results_dir, split_name)
    if reuse_cache and path.exists():
        shared = pd.read_csv(path)
        if all(column in shared.columns for column in NEIGHBOR_FEATURE_COLUMNS):
            log(f"[AE_PRE] reusing shared solute features: {path.name}")
            feature_names = [col for col in shared.columns if col.startswith("solute_")]
            return shared, feature_names
        log(f"[AE_PRE] rebuilding stale shared solute features: {path.name}")

    cache = reference_info.get("solute_cache", {})
    if cache.get("fast_reference"):
        log(f"[AE_PRE] building fast shared solute features for {split_name}: rows={len(pred_df)}")
        df = standardize_columns(pred_df.copy())
        rows = []
        feature_cache: Dict[str, List[float]] = {}
        for idx, smiles in enumerate(df["smiles"].astype(str).tolist()):
            if smiles not in feature_cache:
                mol = mol_from_smiles(smiles)
                neighborhood = tanimoto_neighborhood_features(
                    mol,
                    cache["neighbor_fps"],
                    cache["neighbor_scaffolds"],
                )
                neighbor_values = [neighborhood[name] for name in TANIMOTO_NEIGHBOR_FEATURE_NAMES]
                if mol is None:
                    feature_cache[smiles] = [0.0, 999.0, 1.0, 0.0, 0.0, 0.0, 0.0, *neighbor_values]
                else:
                    scaffold = get_scaffold_smiles(mol)
                    scaffold_count = float(cache.get("scaffold_counts", {}).get(scaffold, 0))
                    feature_cache[smiles] = [
                        neighborhood["solute_max_tanimoto"],
                        999.0,
                        scaffold_novelty(mol, cache["scaffold_set"]),
                        scaffold_similarity(mol, cache["scaffold_fps"]),
                        0.0,
                        1.0 if scaffold_count > 0 else 0.0,
                        scaffold_count,
                        *neighbor_values,
                    ]
            rows.append([idx, *feature_cache[smiles]])
        feature_names = list(FEATURE_NAMES_SOLUTE) + [
            "solute_scaffold_in_train_and_val",
            "solute_scaffold_train_and_val_count",
        ] + list(TANIMOTO_NEIGHBOR_FEATURE_NAMES)
        shared = pd.DataFrame(rows, columns=["row_index", *feature_names])
    else:
        shared, feature_names = offline.build_shared_solute_feature_frame(
            pred_df,
            reference_info,
            feature_n_jobs=feature_n_jobs,
            feature_chunk_size=feature_chunk_size,
            feature_backend=feature_backend,
        )
    keep_cols = ["row_index"] + feature_names
    tmp_path = path.with_suffix(".csv.tmp")
    shared[keep_cols].to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
    return shared[keep_cols], feature_names


def shared_for_prediction(pred_df: pd.DataFrame, shared_features: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    pred_df = offline.standardize_columns(pred_df.copy())
    used_rows = shared_features["row_index"].to_numpy(dtype=int)
    sub = pred_df.iloc[used_rows].reset_index(drop=True)
    return pd.concat(
        [
            pd.DataFrame({"row_index": used_rows}),
            sub,
            shared_features[feature_names].reset_index(drop=True),
        ],
        axis=1,
    )


def generate_features(args: argparse.Namespace) -> Dict:
    os.chdir(SCRIPT_DIR)
    results_dir = resolve_path(args.results_dir, required=False)
    assert results_dir is not None
    train_and_val_csv = write_train_and_val_reference(args.split_dir, results_dir)
    reference_df = pd.read_csv(train_and_val_csv)
    calibration_label, test_label = split_labels(args.test_fraction)
    existing_train = results_dir / f"AE_PRE_MERGED_{calibration_label}.csv"
    existing_test = results_dir / f"AE_PRE_MERGED_{test_label}.csv"
    if existing_train.exists() != existing_test.exists():
        raise FileNotFoundError(
            f"Both {calibration_label} and {test_label} merged split files must exist together."
        )

    reuse_existing_split = existing_train.exists() and not args.rebuild_merged_split
    if reuse_existing_split:
        train_df = pd.read_csv(existing_train)
        test_df = pd.read_csv(existing_test)
        try:
            validation = validate_merged_split(
                train_df,
                test_df,
                calibration_label=calibration_label,
                test_label=test_label,
                reference_df=reference_df,
            )
        except ValueError as exc:
            log(f"[AE_PRE] existing merged split is unsafe and will be rebuilt: {exc}")
            reuse_existing_split = False

    if reuse_existing_split:
        train_csv = existing_train.resolve()
        test_csv = existing_test.resolve()
        combined = pd.concat([train_df, test_df], ignore_index=True)
        split_summary = {
            "split_provenance": "validated_existing_fixed_split",
            "split_strategy": "preexisting_canonical_pair_disjoint",
            "current_test_csv": None,
            "ae_pre_test_csv": None,
            "merged_rows": int(len(combined)),
            "dropped_rows_without_any_target": None,
            "dropped_rows_overlapping_train_and_val": None,
            "calibration_label": calibration_label,
            "test_label": test_label,
            "calibration_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "seed": None,
            "test_fraction": float(len(test_df) / len(combined)),
            **validation,
            "source_group_counts": {str(k): int(v) for k, v in combined["source_group"].value_counts().items()},
            "calibration_source_group_counts": {
                str(k): int(v) for k, v in train_df["source_group"].value_counts().items()
            },
            "test_source_group_counts": {str(k): int(v) for k, v in test_df["source_group"].value_counts().items()},
            "reused_existing_merged_split": True,
        }
        log(f"[AE_PRE] reusing validated fixed split files: {train_csv.name}, {test_csv.name}")
    else:
        current_test_csv = resolve_path(
            args.current_test_csv,
            default=ROOT / "data" / "splits" / "deployment" / "deployment_test.csv",
            required=True,
        )
        ae_pre_test_csv = discover_ae_pre_test(args.ae_pre_test_csv)
        train_csv, test_csv, split_summary = write_merged_split_files(
            current_test_csv=current_test_csv,  # type: ignore[arg-type]
            ae_pre_test_csv=ae_pre_test_csv,
            out_dir=results_dir,
            reference_csv=train_and_val_csv,
            seed=args.seed,
            test_fraction=args.test_fraction,
        )
        split_summary["reused_existing_merged_split"] = False

    calibration_output_csv = resolve_path(args.calibration_output_csv, required=False)
    test_features_output_csv = resolve_path(args.test_features_output_csv, required=False)
    test_labels_output_csv = resolve_path(args.test_labels_output_csv, required=False)
    summary_json = resolve_path(args.summary_json, required=False)
    assert calibration_output_csv is not None
    assert test_features_output_csv is not None
    assert test_labels_output_csv is not None
    assert summary_json is not None
    checkpoint_paths = offline.discover_checkpoints(args.best_dir, split=args.split, expected=5)
    reference_info = build_reference_info_once(
        train_and_val_csv,
        reuse_cache=args.reuse_intermediate_cache,
    )
    shared_feature_cache: Dict[str, Tuple[pd.DataFrame, List[str]]] = {}

    all_frames: List[pd.DataFrame] = []
    summary = {
        "version": "offline_ae_features_ae_pre_v4_sealed_test",
        "calibration_features_csv": str(calibration_output_csv.resolve()),
        "test_features_csv": str(test_features_output_csv.resolve()),
        "test_labels_csv": str(test_labels_output_csv.resolve()),
        "split": args.split,
        "reference_csv": str(train_and_val_csv.resolve()),
        "reuse_intermediate_cache": bool(args.reuse_intermediate_cache),
        **split_summary,
        "folds": [],
    }

    for fold_idx, checkpoint_path in enumerate(checkpoint_paths, start=1):
        log(f"[AE_PRE] fold {fold_idx}/5")
        paths = auto_dataset_paths(checkpoint_path, extra_roots=[args.split_dir, "."])
        model = MyPredictionModel.load(
            checkpoint_path,
            train_csv_override=paths["reference_csv"],
            cv_fold_override=paths["reference_fold"],
        )

        fold_summary = {"fold": fold_idx, "checkpoint_path": checkpoint_path, "sources": []}
        for split_name, csv_path in [("calibration", train_csv), ("test", test_csv)]:
            feature_rows_csv = results_dir / f"offline_feature_rows_ae_pre_fold{fold_idx}_{split_name}.csv"
            prediction_csv = results_dir / f"offline_predictions_ae_pre_fold{fold_idx}_{split_name}.csv"
            feature_df = None
            if args.reuse_intermediate_cache and feature_rows_csv.exists():
                cached_feature_df = pd.read_csv(feature_rows_csv)
                if all(column in cached_feature_df.columns for column in NEIGHBOR_FEATURE_COLUMNS):
                    log(f"[AE_PRE] reusing feature rows: {feature_rows_csv.name}")
                    feature_df = cached_feature_df
                else:
                    log(f"[AE_PRE] rebuilding stale feature rows: {feature_rows_csv.name}")
            if feature_df is None:
                if args.reuse_intermediate_cache and prediction_csv.exists():
                    log(f"[AE_PRE] reusing predictions: {prediction_csv.name}")
                    pred_df = pd.read_csv(prediction_csv)
                else:
                    log(f"[AE_PRE] predicting fold {fold_idx} {split_name}: {csv_path.name}")
                    pred_df = predict_csv_with_checkpoint(
                        model,
                        str(csv_path),
                        output_csv=str(prediction_csv),
                        batch_size=256,
                    )
                if split_name not in shared_feature_cache:
                    shared_feature_cache[split_name] = load_or_build_shared_features(
                        pred_df=pred_df,
                        reference_info=reference_info,
                        split_name=split_name,
                        results_dir=results_dir,
                        feature_n_jobs=args.feature_n_jobs,
                        feature_chunk_size=args.feature_chunk_size,
                        feature_backend=args.feature_backend,
                        reuse_cache=args.reuse_intermediate_cache,
                    )
                shared_features, feature_names = shared_feature_cache[split_name]
                shared_df = shared_for_prediction(pred_df, shared_features, feature_names)
                frames = []
                for prop in offline.config.PROPERTIES:
                    log(f"[AE_PRE] property rows fold {fold_idx} {split_name} {prop}")
                    frames.append(
                        offline.rows_for_property_from_shared(
                            shared_df=shared_df,
                            feature_names=feature_names,
                            prop=prop,
                            fold_idx=fold_idx,
                            split_name=split_name,
                            source_csv=str(csv_path.resolve()),
                            checkpoint_path=checkpoint_path,
                        )
                    )
                feature_df = pd.concat(frames, ignore_index=True)
                feature_tmp = feature_rows_csv.with_suffix(".csv.tmp")
                feature_df.to_csv(feature_tmp, index=False)
                os.replace(feature_tmp, feature_rows_csv)
            all_frames.append(feature_df)
            fold_summary["sources"].append(
                {
                    "split": split_name,
                    "source_csv": str(csv_path.resolve()),
                    "feature_rows_csv": str(feature_rows_csv.resolve()),
                    "feature_rows": int(len(feature_df)),
                }
            )
        summary["folds"].append(fold_summary)

    feature_df = pd.concat(all_frames, ignore_index=True)
    test_raw = feature_df[feature_df["split"] == "test"].copy()
    label_consistency = test_raw.groupby(["row_index", "property"])["y_true"].nunique(dropna=False)
    if not (label_consistency == 1).all():
        raise ValueError("Held-out Test labels are inconsistent across fold rows.")
    test_labels_df = (
        test_raw.groupby(["row_index", "property"], as_index=False, sort=False)
        .agg(y_true=("y_true", "first"))
        .sort_values(["property", "row_index"])
        .reset_index(drop=True)
    )
    
    target_columns = ["y_true", "ae", "true_ae", "abs_error", "log_abs_error", "sample_weight", "epsilon"]
    test_mask = feature_df["split"] == "test"
    feature_df.loc[test_mask, target_columns] = np.nan
    log("[AE_PRE] engineering final scaffold, fold-uncertainty, and calibration features")
    feature_df = engineer_rank_ensemble_features(feature_df, str(train_and_val_csv))
    missing_final_features = [column for column in FINAL_FEATURE_COLUMNS if column not in feature_df.columns]
    if missing_final_features:
        raise ValueError(f"Step 02 output is missing final Reliability features: {missing_final_features}")
    feature_matrix = feature_df[FINAL_FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(feature_matrix).all():
        bad_columns = [
            column
            for column in FINAL_FEATURE_COLUMNS
            if not np.isfinite(pd.to_numeric(feature_df[column], errors="coerce").to_numpy(dtype=float)).all()
        ]
        raise ValueError(f"Step 02 generated non-finite final Reliability features: {bad_columns}")
    calibration_df = feature_df[feature_df["split"] == "calibration"].copy()
    test_feature_columns = [
        "fold",
        "split",
        "source_csv",
        "checkpoint_path",
        "row_index",
        "smiles",
        "solvent",
        "property",
        *FINAL_FEATURE_COLUMNS,
    ]
    test_features_df = feature_df.loc[feature_df["split"] == "test", test_feature_columns].copy()
    forbidden_test_columns = set(target_columns) & set(test_features_df.columns)
    if forbidden_test_columns:
        raise ValueError(f"Held-out Test features contain forbidden label columns: {sorted(forbidden_test_columns)}")
    if test_features_df.duplicated(["property", "row_index", "fold"]).any():
        raise ValueError("Held-out Test features contain duplicate property/row_index/fold keys.")
    if test_labels_df.duplicated(["property", "row_index"]).any():
        raise ValueError("Held-out Test labels contain duplicate property/row_index keys.")
    test_feature_keys = set(map(tuple, test_features_df[["property", "row_index"]].drop_duplicates().to_numpy()))
    test_label_keys = set(map(tuple, test_labels_df[["property", "row_index"]].to_numpy()))
    if test_feature_keys != test_label_keys:
        raise ValueError("Held-out Test feature and label keys do not match exactly.")

    log("[AE_PRE] leakage audit passed: Test labels were sealed before fitted feature engineering")
    calibration_output_csv.parent.mkdir(parents=True, exist_ok=True)
    test_features_output_csv.parent.mkdir(parents=True, exist_ok=True)
    test_labels_output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    for frame, path in [
        (calibration_df, calibration_output_csv),
        (test_features_df, test_features_output_csv),
        (test_labels_df, test_labels_output_csv),
    ]:
        tmp_path = path.with_suffix(".csv.tmp")
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    summary["calibration_feature_rows"] = int(len(calibration_df))
    summary["test_feature_rows"] = int(len(test_features_df))
    summary["test_label_rows"] = int(len(test_labels_df))
    summary["final_reliability_feature_columns"] = list(FINAL_FEATURE_COLUMNS)
    summary["feature_reference_scope"] = {
        "tanimoto_and_scaffold_counts": "TrainAndVal.csv only",
        "historical_residual_statistics": "calibration split only",
        "calibration_training_rows": "leave-one-row_index-out",
        "test_labels_loaded_during_fitted_feature_engineering": False,
        "test_label_columns_in_test_features_csv": [],
    }
    summary_tmp = summary_json.with_suffix(".json.tmp")
    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(summary_tmp, summary_json)
    log(f"[AE_PRE] saved calibration features: {calibration_output_csv.resolve()}")
    log(f"[AE_PRE] saved sealed Test features: {test_features_output_csv.resolve()}")
    log(f"[AE_PRE] saved separate Test labels: {test_labels_output_csv.resolve()}")
    return summary


if __name__ == "__main__":
    result = generate_features(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
