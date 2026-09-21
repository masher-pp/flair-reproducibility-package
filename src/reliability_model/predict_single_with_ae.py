from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import config
from main_model_inference import (
    MyPredictionModel,
    auto_dataset_paths,
    log,
    predict_csv_with_checkpoint,
)
from data_utils import (
    build_solvent_tanimoto_reference,
    build_unique_tanimoto_reference,
    get_morgan_fingerprint,
    get_scaffold_smiles,
    max_tanimoto_to_train_set,
    mol_from_smiles,
    scaffold_novelty,
    scaffold_similarity,
    solvent_reference_features,
    standardize_columns,
    tanimoto_neighborhood_features,
)
from log_mlp_reliability import load_log_mlp_bundle
from log_mlp_reliability import predict_pre_ae as predict_log_mlp_pre_ae

import importlib

offline = importlib.import_module("01_generate_offline_features")


PROPERTY_DISPLAY_ORDER = ["abs", "emi", "plqy", "em"]


def content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Abs/Emi/Plqy/Em and numeric pre_AE for one solute/solvent pair."
    )
    parser.add_argument("--smiles", default=None, help="Solute SMILES. If omitted, the script will ask interactively.")
    parser.add_argument("--solvent", default=None, help="Solvent SMILES. If omitted, the script will ask interactively.")
    parser.add_argument("--split", choices=["random", "scaffold", "ae"], default="ae")
    parser.add_argument("--best_dir", default="../../models/main_model")
    parser.add_argument(
        "--ae_model",
        default="../../models/reliability_model/mlp_log_hpo100_reliability_model_bundle.pt",
        help="v3 Log-MLP Reliability checkpoint (.pt).",
    )
    parser.add_argument("--offline_csv", default=None, help="Offline AE feature CSV used to build kNN error cache.")
    parser.add_argument("--deploy_cache", default=None, help="Deployment cache path. Created automatically if missing.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rebuild_cache", action="store_true", help="Rebuild deployment cache even if it already exists.")
    parser.add_argument("--build_cache_only", action="store_true", help="Only build deployment cache and exit.")
    parser.add_argument("--json", action="store_true", help="Print compact JSON only.")
    return parser.parse_args()


def fill_interactive_inputs(args: argparse.Namespace) -> argparse.Namespace:
    if args.build_cache_only:
        args.smiles = args.smiles or "CCO"
        args.solvent = args.solvent or "O"
        return args

    if args.smiles is None or str(args.smiles).strip() == "":
        args.smiles = input(" solute SMILES: ").strip()
    if args.solvent is None or str(args.solvent).strip() == "":
        args.solvent = input(" solvent SMILES: ").strip()

    if not args.smiles:
        raise ValueError("solute SMILES 。")
    if not args.solvent:
        raise ValueError("solvent SMILES 。")
    return args


def resolve_path(path: str | None, default: str | None = None) -> Path:
    raw = path or default
    if raw is None:
        raise ValueError("Missing path.")
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p.resolve()


def make_query_csv(smiles: str, solvent: str) -> str:
    row = {
        config.SMILES_COLUMN: smiles,
        config.SOLVENT_COLUMN: solvent,
        "abs": 0.0,
        "emi": 0.0,
        "plqy": 0.0,
        "em": 0.0,
    }
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    pd.DataFrame([row]).to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def load_main_models(split: str, best_dir: str, batch_size: int):
    checkpoints = offline.discover_checkpoints(best_dir, split=split, expected=5)
    loaded = []
    for fold_idx, checkpoint in enumerate(checkpoints, start=1):
        paths = auto_dataset_paths(checkpoint, extra_roots=[".", Path(__file__).resolve().parent])
        log(f"[Single Predict] loading main model fold {fold_idx}/5")
        model = MyPredictionModel.load(
            checkpoint,
            train_csv_override=paths["reference_csv"],
            cv_fold_override=paths["reference_fold"],
        )
        loaded.append(
            {
                "fold": fold_idx,
                "checkpoint": checkpoint,
                "paths": paths,
                "model": model,
            }
        )
    return loaded


def predict_main_models(loaded_models: List[Dict], smiles: str, solvent: str, batch_size: int) -> pd.DataFrame:
    query_csv = make_query_csv(smiles, solvent)
    rows = []
    try:
        for item in loaded_models:
            fold = item["fold"]
            pred_df = predict_csv_with_checkpoint(item["model"], query_csv, output_csv=None, batch_size=batch_size)
            row = {"fold": fold}
            for prop in PROPERTY_DISPLAY_ORDER:
                row[prop] = float(pred_df.loc[0, f"pred_{prop}"])
            rows.append(row)
    finally:
        try:
            os.unlink(query_csv)
        except OSError:
            pass
    return pd.DataFrame(rows)


def build_latest_rank_ensemble_cache(
    offline_csv: Path,
    train_and_val_csv: Path,
    *,
    ensemble_base: bool,
) -> Dict:
    if not offline_csv.exists():
        raise FileNotFoundError(f"Missing offline feature CSV: {offline_csv}")
    if not train_and_val_csv.exists():
        raise FileNotFoundError(f"Missing TrainAndVal CSV: {train_and_val_csv}")

    ref_df = standardize_columns(pd.read_csv(train_and_val_csv))
    ref_mols = [mol_from_smiles(smi) for smi in ref_df[config.SMILES_COLUMN].astype(str).tolist()]
    valid_mols = [mol for mol in ref_mols if mol is not None]
    neighbor_smiles_column = "_canonical_smiles" if "_canonical_smiles" in ref_df.columns else config.SMILES_COLUMN
    ref_fps, ref_scaffolds = build_unique_tanimoto_reference(ref_df[neighbor_smiles_column])
    solvent_tanimoto_reference = build_solvent_tanimoto_reference(ref_df[config.SOLVENT_COLUMN].tolist())
    scaffold_set = set()
    scaffold_fps = []
    train_and_val_scaffold_counts: Dict[str, int] = {}
    for mol in valid_mols:
        scaffold = get_scaffold_smiles(mol)
        if not scaffold:
            continue
        scaffold_set.add(scaffold)
        train_and_val_scaffold_counts[scaffold] = train_and_val_scaffold_counts.get(scaffold, 0) + 1
        scaffold_fps.append(get_morgan_fingerprint(mol_from_smiles(scaffold)))

    df = pd.read_csv(offline_csv)
    df = df.copy()
    if ensemble_base:
        keys = ["property", "row_index"]
        group_sizes = df.groupby(keys, sort=False).size()
        if not (group_sizes == 5).all():
            raise ValueError(
                "Ensemble deployment cache requires five base-model rows per sample/property; "
                f"found {group_sizes.value_counts().to_dict()}"
            )
        constant_columns = ["split", "smiles", "solvent", "y_true"]
        for column in constant_columns:
            if (df.groupby(keys, sort=False)[column].nunique(dropna=False) != 1).any():
                raise ValueError(f"Column {column} is inconsistent across base-model rows.")
        grouped = df.groupby(keys, as_index=False, sort=False)
        sample = grouped[constant_columns].first()
        prediction = grouped["base_prediction"].mean().rename(columns={"base_prediction": "base_prediction"})
        df = sample.merge(prediction, on=keys, how="left", validate="one_to_one")
        df["abs_error"] = np.abs(
            df["base_prediction"].to_numpy(dtype=float) - df["y_true"].to_numpy(dtype=float)
        )
    scaffold_by_smiles: Dict[str, str] = {}

    def scaffold_for_smiles(smiles: object) -> str:
        key = "" if pd.isna(smiles) else str(smiles).strip()
        if not key:
            return ""
        if key not in scaffold_by_smiles:
            scaffold_by_smiles[key] = get_scaffold_smiles(mol_from_smiles(key))
        return scaffold_by_smiles[key]

    df["_direct_scaffold"] = df["smiles"].map(scaffold_for_smiles)
    stats_by_property = {}
    for prop in config.PROPERTIES:
        sub = df[(df["split"] == "calibration") & (df["property"] == prop)].copy()
        global_stats = {
            "mean": float(sub["abs_error"].mean()),
            "median": float(sub["abs_error"].median()),
            "max": float(sub["abs_error"].max()),
            "std": float(sub["abs_error"].std(ddof=0)),
        }
        scaffold_stats = sub.groupby("_direct_scaffold")["abs_error"].agg(
            count="count",
            mean="mean",
            median="median",
            max="max",
            std=lambda values: float(values.std(ddof=0)),
        )
        scaffold_solvent_stats = sub.groupby(["_direct_scaffold", "solvent"])["abs_error"].agg(["count", "mean"])
        stats_by_property[prop] = {
            "global": global_stats,
            "scaffold": scaffold_stats,
            "scaffold_solvent": scaffold_solvent_stats,
        }

    return {
        "version": "flair_v3_single_predict_cache",
        "base_prediction_type": (
            "mean_of_5_base_models_before_ae_prediction" if ensemble_base else "individual_base_model"
        ),
        "offline_csv": str(offline_csv),
        "offline_csv_sha256": content_sha256(offline_csv),
        "train_and_val_csv": str(train_and_val_csv),
        "train_and_val_csv_sha256": content_sha256(train_and_val_csv),
        "reference": {
            "mols": valid_mols,
            "fps": ref_fps,
            "neighbor_scaffolds": ref_scaffolds,
            "scaffold_set": scaffold_set,
            "scaffold_fps": scaffold_fps,
            "scaffold_counts": train_and_val_scaffold_counts,
            "solvent_tanimoto_reference": solvent_tanimoto_reference,
        },
        "calibration_stats_by_property": stats_by_property,
    }


def load_or_build_latest_rank_ensemble_cache(
    *,
    offline_csv: Path,
    train_and_val_csv: Path,
    deploy_cache_path: Path,
    rebuild: bool,
    ensemble_base: bool,
) -> Dict:
    expected_base_type = (
        "mean_of_5_base_models_before_ae_prediction" if ensemble_base else "individual_base_model"
    )
    if deploy_cache_path.exists() and not rebuild:
        log(f"[Deploy Cache] loading latest rank-ensemble cache: {deploy_cache_path}")
        with open(deploy_cache_path, "rb") as f:
            cache = pickle.load(f)
        cache_matches_inputs = (
            cache.get("version") == "flair_v3_single_predict_cache"
            and cache.get("base_prediction_type") == expected_base_type
            and cache.get("offline_csv_sha256") == content_sha256(offline_csv)
            and cache.get("train_and_val_csv_sha256") == content_sha256(train_and_val_csv)
        )
        if cache_matches_inputs:
            return cache
        log("[Deploy Cache] cache version or source content changed; rebuilding")

    cache = build_latest_rank_ensemble_cache(
        offline_csv,
        train_and_val_csv,
        ensemble_base=ensemble_base,
    )
    deploy_cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = deploy_cache_path.with_suffix(deploy_cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f)
    os.replace(tmp_path, deploy_cache_path)
    log(f"[Deploy Cache] saved latest rank-ensemble cache: {deploy_cache_path}")
    return cache


def latest_rank_ensemble_feature_row(
    *,
    smiles: str,
    solvent: str,
    prop: str,
    base_value: float,
    fold_values: np.ndarray,
    cache: Dict,
) -> Dict[str, float]:
    reference = cache["reference"]
    mol = mol_from_smiles(smiles)
    scaffold = get_scaffold_smiles(mol)
    scaffold_count = float(reference["scaffold_counts"].get(scaffold, 0))
    row = {
        "solute_max_tanimoto": max_tanimoto_to_train_set(mol, reference["mols"], reference["fps"]) if mol is not None else 0.0,
        "solute_scaffold_novel": scaffold_novelty(mol, reference["scaffold_set"]) if mol is not None else 1.0,
        "solute_scaffold_similarity": scaffold_similarity(mol, reference["scaffold_fps"]) if mol is not None else 0.0,
        "base_prediction": float(base_value),
        "solute_scaffold_in_train_and_val": 1.0 if scaffold_count > 0 else 0.0,
        "solute_scaffold_train_and_val_count": scaffold_count,
        "solute_scaffold_train_and_val_count_log1p": float(np.log1p(scaffold_count)),
        "solute_scaffold_train_and_val_count_sqrt": float(np.sqrt(scaffold_count)),
        "solute_scaffold_rare_in_train_and_val": 1.0 if 0 < scaffold_count <= 2 else 0.0,
        "solute_scaffold_common_in_train_and_val": 1.0 if scaffold_count >= 10 else 0.0,
        "base_prediction_fold_std": float(np.std(fold_values, ddof=0)),
        "base_prediction_fold_range": float(np.max(fold_values) - np.min(fold_values)),
    }
    row.update(
        tanimoto_neighborhood_features(
            mol,
            reference["fps"],
            reference["neighbor_scaffolds"],
        )
    )
    row.update(solvent_reference_features(solvent, reference["solvent_tanimoto_reference"]))
    stats = cache["calibration_stats_by_property"][prop]
    global_stats = stats["global"]
    scaffold_stats = stats["scaffold"]
    scaffold_solvent_stats = stats["scaffold_solvent"]
    if scaffold in scaffold_stats.index:
        row["calibration_same_scaffold_count"] = float(scaffold_stats.loc[scaffold, "count"])
        row["calibration_same_scaffold_mean_abs_error"] = float(scaffold_stats.loc[scaffold, "mean"])
        row["calibration_same_scaffold_median_abs_error"] = float(scaffold_stats.loc[scaffold, "median"])
        row["calibration_same_scaffold_max_abs_error"] = float(scaffold_stats.loc[scaffold, "max"])
        std_error = float(scaffold_stats.loc[scaffold, "std"])
        row["calibration_same_scaffold_std_abs_error"] = std_error if np.isfinite(std_error) else float(global_stats["std"])
    else:
        row["calibration_same_scaffold_count"] = 0.0
        row["calibration_same_scaffold_mean_abs_error"] = float(global_stats["mean"])
        row["calibration_same_scaffold_median_abs_error"] = float(global_stats["median"])
        row["calibration_same_scaffold_max_abs_error"] = float(global_stats["max"])
        row["calibration_same_scaffold_std_abs_error"] = float(global_stats["std"])

    solvent_key = (scaffold, solvent)
    if solvent_key in scaffold_solvent_stats.index:
        row["calibration_same_scaffold_solvent_count"] = float(scaffold_solvent_stats.loc[solvent_key, "count"])
        row["calibration_same_scaffold_solvent_mean_abs_error"] = float(scaffold_solvent_stats.loc[solvent_key, "mean"])
    else:
        row["calibration_same_scaffold_solvent_count"] = 0.0
        row["calibration_same_scaffold_solvent_mean_abs_error"] = float(global_stats["mean"])
    return row


def predict_single(args: argparse.Namespace) -> Dict:
    root = Path(__file__).resolve().parent
    ae_model_path = resolve_path(args.ae_model)
    deploy_cache_path = resolve_path(
        args.deploy_cache,
        "../../models/reliability_model/prediction_deploy_cache_solvent32.pkl",
    )
    if not ae_model_path.exists():
        raise FileNotFoundError(f"Missing v3 Reliability model: {ae_model_path}")
    if ae_model_path.suffix.lower() != ".pt":
        raise ValueError("FLAIR v3 accepts only the Log-MLP .pt Reliability checkpoint.")

    ae_bundle = load_log_mlp_bundle(ae_model_path)
    base_prediction_type = str(ae_bundle.get("base_prediction_type", ""))
    expected_base_type = "mean_of_5_base_models_before_ae_prediction"
    if base_prediction_type != expected_base_type:
        raise ValueError(
            "The v3 Reliability checkpoint must use the mean of five Base predictions."
        )

    offline_default = (
        ae_bundle.get("calibration_features_csv")
        or "../../results/intermediate/calibration_features_ae_pre.csv"
    )
    offline_csv = resolve_path(args.offline_csv, offline_default)
    train_and_val_csv = root.parents[1] / "data/splits/deployment/deployment.csv"
    deploy_cache = load_or_build_latest_rank_ensemble_cache(
        offline_csv=offline_csv,
        train_and_val_csv=train_and_val_csv,
        deploy_cache_path=deploy_cache_path,
        rebuild=args.rebuild_cache,
        ensemble_base=True,
    )

    if args.build_cache_only:
        return {
            "status": "cache_built",
            "model_family": str(ae_bundle["model_family"]),
            "deploy_cache_path": str(deploy_cache_path),
            "base_prediction_type": base_prediction_type,
        }

    loaded_main = load_main_models(args.split, args.best_dir, args.batch_size)
    main_pred_df = predict_main_models(
        loaded_main,
        args.smiles,
        args.solvent,
        args.batch_size,
    )
    output = {"smiles": args.smiles, "solvent": args.solvent, "properties": {}}
    for prop in PROPERTY_DISPLAY_ORDER:
        fold_values = main_pred_df[prop].to_numpy(dtype=float)
        pred_mean = float(np.mean(fold_values))
        row = latest_rank_ensemble_feature_row(
            smiles=args.smiles,
            solvent=args.solvent,
            prop=prop,
            base_value=pred_mean,
            fold_values=fold_values,
            cache=deploy_cache,
        )
        feature_df = pd.DataFrame([row]).reindex(
            columns=ae_bundle["feature_columns"],
            fill_value=0.0,
        )
        predicted_ae = float(predict_log_mlp_pre_ae(ae_bundle, prop, feature_df)[0])
        output["properties"][prop] = {
            "prediction": pred_mean,
            "pre_AE": predicted_ae,
        }
    return output


def print_human_readable(result: Dict) -> None:
    print("\n=== Single Molecule Prediction ===")
    print(f"solute : {result['smiles']}")
    print(f"solvent: {result['solvent']}")
    print(f"\n{'Property':<8} {'Prediction':>14} {'pre_AE':>14}")
    for prop in PROPERTY_DISPLAY_ORDER:
        item = result["properties"][prop]
        print(
            f"{prop:<8} "
            f"{item['prediction']:>14.6g} "
            f"{item['pre_AE']:>14.6g}"
        )


if __name__ == "__main__":
    parsed = fill_interactive_inputs(parse_args())
    result = predict_single(parsed)
    if parsed.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human_readable(result)
