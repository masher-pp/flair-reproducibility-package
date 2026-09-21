from __future__ import annotations

"""Leakage-safe HPO for non-neural Reliability model families.

The script intentionally keeps the Log-MLP-HPO100 data, features, target,
weights, grouped folds, and selection metric unchanged.  It adds RF,
ExtraTrees, HistGradientBoosting, and XGBoost searches and records validation
R2/Spearman/MAE/RMSE for every trial and every fold.  Test15 labels are not
loaded until all selected models have been fitted and their label-free
predictions have been frozen.
"""

import argparse
import concurrent.futures
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pickle
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RELIABILITY_DIR = PACKAGE_ROOT / "src/reliability_model"
HPO_ROOT = PACKAGE_ROOT / "results/hpo/ml_model_families_hpo100_20260825"
FINAL_ROOT = PACKAGE_ROOT / "results/final/ml_model_families_hpo100_20260825"
MODEL_ROOT = PACKAGE_ROOT / "models/reliability_model/ml_model_families_hpo100_20260825"
PLAN_VERSION = "ml_model_families_log_ae_grouped_cv_hpo100_v1"
DEFAULT_FAMILIES = ["rf", "extratrees", "histgb", "xgb"]

os.environ["MAIN_MODEL_ROOT"] = str(PACKAGE_ROOT / "src/main_model")
os.environ["MODEL_CODE_ROOT"] = str(PACKAGE_ROOT / "src/main_model")
sys.path.insert(0, str(RELIABILITY_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STEP03 = load_module("step03_ml_family_hpo", RELIABILITY_DIR / "03_train_rf_leaf1_sqrt_oob.py")
STEP02 = importlib.import_module("02_generate_ae_pre_merged_features")
PROPERTIES = list(STEP03.PROPERTIES)
EPSILON = dict(STEP03.EPSILON_BY_PROPERTY)
FEATURE_COLUMNS = list(STEP03.SOLVENT_AUGMENTED_MODEL_FEATURE_COLUMNS)
PROPERTY_INDEX = {prop: index for index, prop in enumerate(PROPERTIES)}

_CV_FRAMES: list[pd.DataFrame] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable leakage-safe HPO for RF, ExtraTrees, HistGradientBoosting, "
            "and XGBoost Reliability models."
        )
    )
    parser.add_argument("--families", nargs="+", choices=DEFAULT_FAMILIES, default=DEFAULT_FAMILIES)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--hpo-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def inverse_log(values: np.ndarray, prop: str) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), -50.0, 50.0)
    return np.maximum(np.exp(values) - float(EPSILON[prop]), 0.0)


def metric_values(true_ae: np.ndarray, predicted_ae: np.ndarray) -> dict[str, float]:
    true_ae = np.asarray(true_ae, dtype=float)
    predicted_ae = np.asarray(predicted_ae, dtype=float)
    return {
        "r2": float(r2_score(true_ae, predicted_ae)),
        "spearman": float(STEP03.spearman_manual(predicted_ae, true_ae)),
        "mae": float(mean_absolute_error(true_ae, predicted_ae)),
        "rmse": float(math.sqrt(mean_squared_error(true_ae, predicted_ae))),
    }


def full_metric_values(
    true_ae: np.ndarray,
    predicted_ae: np.ndarray,
    true_log: np.ndarray,
    predicted_log: np.ndarray,
) -> dict[str, float]:
    values = metric_values(true_ae, predicted_ae)
    values["log_r2"] = float(r2_score(true_log, predicted_log))
    return values


def weighted(rows: list[dict[str, Any]], key: str, weight_key: str = "n") -> float:
    weights = np.asarray([row[weight_key] for row in rows], dtype=float)
    values = np.asarray([row[key] for row in rows], dtype=float)
    valid = np.isfinite(weights) & np.isfinite(values) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def prepare_base_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    calibration_path = PACKAGE_ROOT / "results/intermediate/calibration_features_ae_pre.csv"
    test_path = PACKAGE_ROOT / "results/intermediate/test_features_ae_pre.csv"
    reference_path = PACKAGE_ROOT / "data/splits/deployment/deployment.csv"
    calibration, _ = STEP03.load_calibration_features(str(calibration_path), feature_columns=FEATURE_COLUMNS)
    test = STEP03.load_test_features(str(test_path))
    calibration = STEP03.aggregate_base_predictions(calibration, include_targets=True)
    test = STEP03.aggregate_base_predictions(test, include_targets=False)
    calibration, test, solvent_summary = STEP03.add_solvent_reference_feature_columns(
        calibration, test, str(reference_path)
    )
    scaffold_cache: dict[str, str] = {}

    def scaffold(smiles: object) -> str:
        key = "" if pd.isna(smiles) else str(smiles).strip()
        if key not in scaffold_cache:
            scaffold_cache[key] = (
                STEP03.get_scaffold_smiles(STEP03.mol_from_smiles(key)) if key else ""
            )
        return scaffold_cache[key]

    calibration["_direct_scaffold"] = calibration["smiles"].map(scaffold)
    test["_direct_scaffold"] = test["smiles"].map(scaffold)
    return calibration, test, solvent_summary


def load_final_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    calibration_path = PACKAGE_ROOT / "results/intermediate/calibration_features_ae_pre.csv"
    test_path = PACKAGE_ROOT / "results/intermediate/test_features_ae_pre.csv"
    reference_path = PACKAGE_ROOT / "data/splits/deployment/deployment.csv"
    calibration, _ = STEP03.load_calibration_features(str(calibration_path), feature_columns=FEATURE_COLUMNS)
    test = STEP03.load_test_features(str(test_path))
    calibration = STEP03.aggregate_base_predictions(calibration, include_targets=True)
    test = STEP03.aggregate_base_predictions(test, include_targets=False)
    calibration, test = STEP03.rebuild_ensemble_history_features(calibration, test)
    calibration, test, solvent_summary = STEP03.add_solvent_reference_feature_columns(
        calibration, test, str(reference_path)
    )
    return calibration, test, solvent_summary


def prepare_cv_cache(calibration: pd.DataFrame, reset: bool = False) -> list[pd.DataFrame]:
    cache_dir = HPO_ROOT / "cv_cache"
    paths = [cache_dir / f"fold_{index}.pkl" for index in range(1, 6)]
    manifest_path = cache_dir / "manifest.json"
    source_path = PACKAGE_ROOT / "results/intermediate/calibration_features_ae_pre.csv"
    expected = {
        "plan_version": PLAN_VERSION,
        "method": "5-fold canonical-solute-solvent GroupKFold within Calibration85",
        "splitter": "GroupKFold(n_splits=5, shuffle=False)",
        "calibration_sha256": sha256(source_path),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "historical_features_fit_on_cv_training_only": True,
        "training_weights_recomputed_on_cv_training_only": True,
    }
    if not reset and manifest_path.exists() and all(path.exists() for path in paths):
        if json.loads(manifest_path.read_text(encoding="utf-8")) == expected:
            print("Reusing leakage-safe CV cache", flush=True)
            return [pd.read_pickle(path) for path in paths]
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples = calibration[["row_index", "smiles", "solvent"]].drop_duplicates().copy()
    if samples["row_index"].duplicated().any():
        raise ValueError("row_index maps to more than one solute-solvent pair")
    samples["pair_group"] = STEP02.canonical_pair_keys(samples)
    splitter = GroupKFold(n_splits=5)
    keep_columns = list(
        dict.fromkeys(
            [
                "split",
                "property",
                "row_index",
                "smiles",
                "solvent",
                "y_true",
                "base_prediction",
                "abs_error",
                "sample_weight",
                *FEATURE_COLUMNS,
            ]
        )
    )
    frames: list[pd.DataFrame] = []
    for cv_fold, (_, validation_positions) in enumerate(
        splitter.split(samples, groups=samples["pair_group"]), start=1
    ):
        validation_ids = set(samples.iloc[validation_positions]["row_index"].astype(int))
        work = calibration.copy()
        is_validation = work["row_index"].astype(int).isin(validation_ids)
        work["split"] = np.where(is_validation, "test", "calibration")
        work.loc[is_validation, "abs_error"] = np.nan
        work = STEP03.add_calibration_scaffold_error_features(work)
        work.loc[is_validation, "split"] = "validation"
        for prop in PROPERTIES:
            train_mask = (work["split"] == "calibration") & (work["property"] == prop)
            train_errors = work.loc[train_mask, "abs_error"].to_numpy(dtype=float)
            work.loc[train_mask, "sample_weight"] = STEP03.smooth_loss_weights(train_errors, prop)
        frame = work[keep_columns].copy()
        if not np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite feature in CV fold {cv_fold}")
        if frame.loc[frame["split"] == "calibration", "sample_weight"].isna().any():
            raise ValueError(f"Missing training weights in CV fold {cv_fold}")
        frame.to_pickle(paths[cv_fold - 1])
        frames.append(frame)
        print(f"Prepared CV fold {cv_fold}/5", flush=True)
    atomic_json(expected, manifest_path)
    return frames


def loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def sample_config(family: str, rng: np.random.Generator, smoke: bool) -> dict[str, Any]:
    if family in {"rf", "extratrees"}:
        return {
            "n_estimators": 20 if smoke else int(rng.choice([80, 120, 200, 300, 500])),
            "max_depth": None if rng.random() < 0.20 else int(rng.choice([6, 10, 14, 20, 30])),
            "min_samples_leaf": int(rng.choice([1, 2, 4, 6, 10, 15])),
            "min_samples_split": int(rng.choice([2, 4, 8, 12, 20])),
            "max_features": str(rng.choice(["sqrt", "0.25", "0.5", "0.75", "1.0"])),
            "max_samples": None if rng.random() < 0.20 else float(rng.choice([0.60, 0.75, 0.90, 1.0])),
            "criterion": "squared_error",
        }
    if family == "histgb":
        return {
            "loss": str(rng.choice(["squared_error", "absolute_error"])),
            "learning_rate": loguniform(rng, 0.015, 0.25),
            "max_iter": 20 if smoke else int(rng.choice([100, 200, 300, 500])),
            "max_leaf_nodes": int(rng.choice([15, 31, 63, 127])),
            "max_depth": None if rng.random() < 0.25 else int(rng.choice([4, 6, 8, 12])),
            "min_samples_leaf": int(rng.choice([5, 10, 20, 30, 50])),
            "l2_regularization": loguniform(rng, 1e-8, 10.0),
            "max_bins": int(rng.choice([63, 127, 255])),
        }
    if family == "xgb":
        return {
            "n_estimators": 20 if smoke else int(rng.choice([100, 200, 300, 500, 800])),
            "learning_rate": loguniform(rng, 0.01, 0.20),
            "max_depth": int(rng.choice([2, 3, 4, 5, 6, 8, 10])),
            "min_child_weight": loguniform(rng, 0.5, 20.0),
            "subsample": float(rng.uniform(0.55, 1.0)),
            "colsample_bytree": float(rng.uniform(0.45, 1.0)),
            "reg_alpha": loguniform(rng, 1e-8, 2.0),
            "reg_lambda": loguniform(rng, 1e-3, 20.0),
            "gamma": float(rng.uniform(0.0, 2.0)),
        }
    raise KeyError(family)


def sample_configs(family: str, n_trials: int, smoke: bool) -> list[dict[str, Any]]:
    seed_by_family = {"rf": 202608251, "extratrees": 202608252, "histgb": 202608253, "xgb": 202608254}
    rng = np.random.default_rng(seed_by_family[family])
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(configs) < n_trials:
        config = sample_config(family, rng, smoke)
        key = stable_json(config)
        if key in seen:
            continue
        seen.add(key)
        configs.append(config)
    return configs


def normalize_tree_params(config: dict[str, Any]) -> dict[str, Any]:
    params = dict(config)
    value = params.get("max_features")
    if isinstance(value, str) and value != "sqrt":
        params["max_features"] = float(value)
    return params


def make_model(family: str, config: dict[str, Any], seed: int):
    if family == "rf":
        return RandomForestRegressor(
            **normalize_tree_params(config),
            bootstrap=True,
            n_jobs=1,
            random_state=seed,
        )
    if family == "extratrees":
        return ExtraTreesRegressor(
            **normalize_tree_params(config),
            bootstrap=True,
            n_jobs=1,
            random_state=seed,
        )
    if family == "histgb":
        return HistGradientBoostingRegressor(
            **config,
            early_stopping=False,
            random_state=seed,
        )
    if family == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for family=xgb. Install xgboost==2.1.4 or add its "
                "isolated vendor directory to PYTHONPATH."
            ) from exc
        return XGBRegressor(
            **config,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )
    raise KeyError(family)


def init_worker(cache_dir: str) -> None:
    global _CV_FRAMES
    _CV_FRAMES = [pd.read_pickle(Path(cache_dir) / f"fold_{index}.pkl") for index in range(1, 6)]


def evaluate_config(family: str, trial_id: int, config: dict[str, Any]) -> dict[str, Any]:
    if _CV_FRAMES is None:
        raise RuntimeError("Worker CV cache not initialized")
    started = time.time()
    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        prop: {"pred_ae": [], "true_ae": [], "pred_log": [], "true_log": []}
        for prop in PROPERTIES
    }
    fold_rows: list[dict[str, Any]] = []
    for cv_fold, frame in enumerate(_CV_FRAMES, start=1):
        for prop in PROPERTIES:
            train = frame[(frame["split"] == "calibration") & (frame["property"] == prop)]
            validation = frame[(frame["split"] == "validation") & (frame["property"] == prop)]
            x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
            y_train = np.log(train["abs_error"].to_numpy(dtype=float) + float(EPSILON[prop]))
            x_validation = validation[FEATURE_COLUMNS].to_numpy(dtype=float)
            true_ae = np.abs(
                validation["base_prediction"].to_numpy(dtype=float)
                - validation["y_true"].to_numpy(dtype=float)
            )
            true_log = np.log(true_ae + float(EPSILON[prop]))
            model_seed = 42 + cv_fold * 100 + PROPERTY_INDEX[prop]
            model = make_model(family, config, model_seed)
            model.fit(x_train, y_train, sample_weight=train["sample_weight"].to_numpy(dtype=float))
            predicted_log = np.asarray(model.predict(x_validation), dtype=float)
            predicted_ae = inverse_log(predicted_log, prop)
            values = full_metric_values(true_ae, predicted_ae, true_log, predicted_log)
            fold_rows.append(
                {
                    "family": family,
                    "trial_id": trial_id,
                    "cv_fold": cv_fold,
                    "property": prop,
                    "n": int(len(validation)),
                    **values,
                }
            )
            pooled[prop]["pred_ae"].append(predicted_ae)
            pooled[prop]["true_ae"].append(true_ae)
            pooled[prop]["pred_log"].append(predicted_log)
            pooled[prop]["true_log"].append(true_log)
    result: dict[str, Any] = {
        "family": family,
        "trial_id": trial_id,
        "config_id": hashlib.sha256(stable_json(config).encode("utf-8")).hexdigest()[:16],
        "params_json": stable_json(config),
    }
    property_rows: list[dict[str, Any]] = []
    for prop in PROPERTIES:
        true_ae = np.concatenate(pooled[prop]["true_ae"])
        predicted_ae = np.concatenate(pooled[prop]["pred_ae"])
        true_log = np.concatenate(pooled[prop]["true_log"])
        predicted_log = np.concatenate(pooled[prop]["pred_log"])
        values = full_metric_values(true_ae, predicted_ae, true_log, predicted_log)
        row = {"property": prop, "n": int(len(true_ae)), **values}
        property_rows.append(row)
        for key, value in values.items():
            result[f"{prop}_val_{key}"] = value
    for key in ["r2", "spearman", "mae", "rmse", "log_r2"]:
        result[f"weighted_val_{key}"] = weighted(property_rows, key)
        result[f"macro_val_{key}"] = float(np.mean([row[key] for row in property_rows]))
    fold_family_rows: list[dict[str, Any]] = []
    for cv_fold in range(1, 6):
        rows = [row for row in fold_rows if row["cv_fold"] == cv_fold]
        fold_family_rows.append(
            {
                "cv_fold": cv_fold,
                "weighted_r2": weighted(rows, "r2"),
                "weighted_spearman": weighted(rows, "spearman"),
            }
        )
    result["fold_weighted_r2_mean"] = float(np.mean([row["weighted_r2"] for row in fold_family_rows]))
    result["fold_weighted_r2_std"] = float(np.std([row["weighted_r2"] for row in fold_family_rows]))
    result["fold_weighted_spearman_mean"] = float(
        np.mean([row["weighted_spearman"] for row in fold_family_rows])
    )
    result["fold_weighted_spearman_std"] = float(
        np.std([row["weighted_spearman"] for row in fold_family_rows])
    )
    result["runtime_seconds"] = float(time.time() - started)
    return {"summary": result, "fold_metrics": fold_rows}


def family_manifest(family: str, configs: list[dict[str, Any]], n_trials: int) -> dict[str, Any]:
    source_path = PACKAGE_ROOT / "results/intermediate/calibration_features_ae_pre.csv"
    return {
        "plan_version": PLAN_VERSION,
        "family": family,
        "n_random_trials": n_trials,
        "random_search_seed": {"rf": 202608251, "extratrees": 202608252, "histgb": 202608253, "xgb": 202608254}[family],
        "candidate_sha256": hashlib.sha256(stable_json(configs).encode("utf-8")).hexdigest(),
        "calibration_sha256": sha256(source_path),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "cv_method": "5-fold canonical-solute-solvent GroupKFold within Calibration85",
        "selection_metric": "sample-count-weighted pooled OOF Spearman; R2 winner also recorded",
        "test15_used_for_hpo_selection": False,
        "python": sys.version,
        "platform": platform.platform(),
        "sklearn_version": importlib.import_module("sklearn").__version__,
    }


def run_family(family: str, n_trials: int, workers: int, reset: bool, smoke: bool) -> dict[str, Any]:
    family_dir = HPO_ROOT / family
    if reset and family_dir.exists():
        shutil.rmtree(family_dir)
    family_dir.mkdir(parents=True, exist_ok=True)
    configs = sample_configs(family, n_trials, smoke)
    manifest = family_manifest(family, configs, n_trials)
    manifest_path = family_dir / "manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ["plan_version", "family", "n_random_trials", "candidate_sha256", "calibration_sha256"]:
            if existing_manifest.get(key) != manifest.get(key):
                raise ValueError(f"Existing {family} output is incompatible at {key}; use --reset")
    else:
        atomic_json(manifest, manifest_path)
    trial_path = family_dir / "trials.csv"
    fold_path = family_dir / "fold_metrics.csv"
    existing = pd.read_csv(trial_path) if trial_path.exists() else pd.DataFrame()
    fold_existing = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()
    completed_ids = set(existing["trial_id"].astype(int)) if not existing.empty else set()
    pending = [(index + 1, config) for index, config in enumerate(configs) if index + 1 not in completed_ids]
    rows = existing.to_dict("records") if not existing.empty else []
    fold_rows = fold_existing.to_dict("records") if not fold_existing.empty else []
    if pending:
        cache_dir = HPO_ROOT / "cv_cache"
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, workers), initializer=init_worker, initargs=(str(cache_dir),)
        ) as executor:
            futures = {
                executor.submit(evaluate_config, family, trial_id, config): trial_id
                for trial_id, config in pending
            }
            for future in concurrent.futures.as_completed(futures):
                payload = future.result()
                summary = payload["summary"]
                rows.append(summary)
                fold_rows.extend(payload["fold_metrics"])
                atomic_csv(pd.DataFrame(rows).sort_values("trial_id"), trial_path)
                atomic_csv(
                    pd.DataFrame(fold_rows).sort_values(["trial_id", "cv_fold", "property"]), fold_path
                )
                print(
                    f"[{family}] Trial {int(summary['trial_id']):03d}/{n_trials}: "
                    f"weighted val Spearman={summary['weighted_val_spearman']:.6f}; "
                    f"R2={summary['weighted_val_r2']:.6f}; runtime={summary['runtime_seconds']:.1f}s",
                    flush=True,
                )
    trials = pd.DataFrame(rows)
    if len(trials) != n_trials or set(trials["trial_id"].astype(int)) != set(range(1, n_trials + 1)):
        raise RuntimeError(f"Expected {n_trials} unique {family} trials, found {len(trials)}")
    trials["spearman_rank"] = trials["weighted_val_spearman"].rank(method="min", ascending=False).astype(int)
    trials["r2_rank"] = trials["weighted_val_r2"].rank(method="min", ascending=False).astype(int)
    trials = trials.sort_values(["spearman_rank", "trial_id"]).reset_index(drop=True)
    atomic_csv(trials, trial_path)
    spearman_winner = trials.sort_values(["weighted_val_spearman", "trial_id"], ascending=[False, True]).iloc[0]
    r2_winner = trials.sort_values(["weighted_val_r2", "trial_id"], ascending=[False, True]).iloc[0]
    selection = {
        "family": family,
        "selection_rule": "highest pooled 5-fold sample-count-weighted validation Spearman; ties use lower trial_id",
        "test15_used_for_hpo_selection": False,
        "n_random_trials": n_trials,
        "best_by_spearman": {
            "trial_id": int(spearman_winner["trial_id"]),
            "config_id": str(spearman_winner["config_id"]),
            "params": json.loads(str(spearman_winner["params_json"])),
            "weighted_val_spearman": float(spearman_winner["weighted_val_spearman"]),
            "weighted_val_r2": float(spearman_winner["weighted_val_r2"]),
            "weighted_val_mae": float(spearman_winner["weighted_val_mae"]),
            "weighted_val_rmse": float(spearman_winner["weighted_val_rmse"]),
        },
        "best_by_r2": {
            "trial_id": int(r2_winner["trial_id"]),
            "config_id": str(r2_winner["config_id"]),
            "params": json.loads(str(r2_winner["params_json"])),
            "weighted_val_spearman": float(r2_winner["weighted_val_spearman"]),
            "weighted_val_r2": float(r2_winner["weighted_val_r2"]),
            "weighted_val_mae": float(r2_winner["weighted_val_mae"]),
            "weighted_val_rmse": float(r2_winner["weighted_val_rmse"]),
        },
    }
    atomic_json(selection, family_dir / "best_params.json")
    return selection


def train_selected_and_freeze_predictions(
    family: str,
    selection: dict[str, Any],
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    solvent_summary: dict[str, Any],
) -> tuple[Path, Path]:
    params = dict(selection["best_by_spearman"]["params"])
    models: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for prop in PROPERTIES:
        train_prop = calibration[calibration["property"] == prop]
        test_prop = test[test["property"] == prop]
        target = np.log(train_prop["abs_error"].to_numpy(dtype=float) + float(EPSILON[prop]))
        model = make_model(family, params, 42 + PROPERTY_INDEX[prop])
        model.fit(
            train_prop[FEATURE_COLUMNS].to_numpy(dtype=float),
            target,
            sample_weight=train_prop["sample_weight"].to_numpy(dtype=float),
        )
        models[prop] = model
        predicted_log = np.asarray(model.predict(test_prop[FEATURE_COLUMNS].to_numpy(dtype=float)), dtype=float)
        frame = test_prop[["property", "row_index", "fold"]].copy()
        frame["predicted_log_ae"] = predicted_log
        frame["predicted_ae"] = inverse_log(predicted_log, prop)
        prediction_frames.append(frame)
    final_dir = FINAL_ROOT / family
    final_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = final_dir / "test_predictions_label_free.csv"
    atomic_csv(pd.concat(prediction_frames, ignore_index=True), prediction_path)
    member_name = f"{family}_log_hpo100_32f"
    bundle = {
        "version": f"ae_pre_ml_family_{family}_log_hpo100_32f_grouped_cv_v1",
        "model_name": member_name,
        "model_family": family,
        "base_prediction_type": "mean_of_5_base_models_before_ae_prediction",
        "optimization_target": "log(AE + epsilon_by_property)",
        "target_transform": "log(AE + epsilon_by_property); inverse=max(exp(output)-epsilon,0)",
        "loss_weighting": "same sample weights as Log-MLP-HPO100",
        "uses_calibration_feature_cache": True,
        "calibration_features_csv": "results/intermediate/calibration_features_ae_pre.csv",
        "test_features_csv": "results/intermediate/test_features_ae_pre.csv",
        "feature_columns": FEATURE_COLUMNS,
        "members": [member_name],
        "models": {member_name: {"models": models, "kind": family, "params": params}},
        "properties": PROPERTIES,
        "epsilon_by_property": EPSILON,
        "selection": selection,
        "solvent_reference_summary": solvent_summary,
    }
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_ROOT / f"{family}_log_hpo100_bundle.pkl"
    tmp = model_path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as handle:
        pickle.dump(bundle, handle)
    os.replace(tmp, model_path)
    return prediction_path, model_path


def evaluate_frozen_predictions(
    family: str,
    prediction_path: Path,
    model_path: Path,
    test: pd.DataFrame,
    selection: dict[str, Any],
) -> dict[str, Any]:
    labels = STEP03.load_test_labels(str(PACKAGE_ROOT / "results/intermediate/test_labels_ae_pre.csv"))
    predictions = pd.read_csv(prediction_path)
    evaluated = test.merge(
        predictions, on=["property", "row_index", "fold"], how="left", validate="one_to_one"
    ).merge(labels, on=["property", "row_index"], how="left", validate="many_to_one")
    if evaluated[["predicted_ae", "predicted_log_ae", "y_true"]].isna().any().any():
        raise ValueError(f"Missing value after Test15 join for {family}")
    metric_rows: list[dict[str, Any]] = []
    for prop in PROPERTIES:
        subset = evaluated[evaluated["property"] == prop]
        true_ae = np.abs(
            subset["base_prediction"].to_numpy(dtype=float) - subset["y_true"].to_numpy(dtype=float)
        )
        predicted_ae = subset["predicted_ae"].to_numpy(dtype=float)
        true_log = np.log(true_ae + float(EPSILON[prop]))
        predicted_log = subset["predicted_log_ae"].to_numpy(dtype=float)
        metric_rows.append(
            {
                "family": family,
                "property": prop,
                "n": int(len(subset)),
                **full_metric_values(true_ae, predicted_ae, true_log, predicted_log),
            }
        )
    weighted_row = {
        "family": family,
        "property": "weighted_all",
        "n": int(sum(row["n"] for row in metric_rows)),
    }
    for key in ["r2", "spearman", "mae", "rmse", "log_r2"]:
        weighted_row[key] = weighted(metric_rows, key)
    metric_rows.append(weighted_row)
    metrics = pd.DataFrame(metric_rows)
    final_dir = FINAL_ROOT / family
    atomic_csv(metrics, final_dir / "test15_metrics.csv")
    atomic_csv(evaluated, final_dir / "test15_evaluated_rows.csv")
    weighted_test = metrics[metrics["property"] == "weighted_all"].iloc[0]
    validation = {
        "status": "passed",
        "family": family,
        "n_random_trials": int(selection["n_random_trials"]),
        "cv_method": "5-fold canonical-solute-solvent GroupKFold within Calibration85",
        "selection_metric": "sample-count-weighted pooled OOF Spearman",
        "test15_used_for_hpo_selection": False,
        "test_labels_loaded_after_prediction_freeze": True,
        "n_features": len(FEATURE_COLUMNS),
        "selected_trial_id": int(selection["best_by_spearman"]["trial_id"]),
        "selected_params": selection["best_by_spearman"]["params"],
        "weighted_cv_spearman": float(selection["best_by_spearman"]["weighted_val_spearman"]),
        "weighted_cv_r2": float(selection["best_by_spearman"]["weighted_val_r2"]),
        "weighted_test15_spearman": float(weighted_test["spearman"]),
        "weighted_test15_r2": float(weighted_test["r2"]),
        "weighted_test15_mae": float(weighted_test["mae"]),
        "weighted_test15_rmse": float(weighted_test["rmse"]),
        "model_path": str(model_path.relative_to(PACKAGE_ROOT)),
        "model_sha256": sha256(model_path),
        "label_free_prediction_sha256": sha256(prediction_path),
    }
    atomic_json(validation, final_dir / "validation.json")
    return validation


def write_cross_family_summary(validations: list[dict[str, Any]]) -> None:
    rows = []
    for item in validations:
        rows.append(
            {
                "family": item["family"],
                "selected_trial_id": item["selected_trial_id"],
                "weighted_cv_spearman": item["weighted_cv_spearman"],
                "weighted_cv_r2": item["weighted_cv_r2"],
                "weighted_test15_spearman": item["weighted_test15_spearman"],
                "weighted_test15_r2": item["weighted_test15_r2"],
                "weighted_test15_mae": item["weighted_test15_mae"],
                "weighted_test15_rmse": item["weighted_test15_rmse"],
                "model_path": item["model_path"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("weighted_cv_spearman", ascending=False)
    atomic_csv(summary, FINAL_ROOT / "model_family_summary.csv")
    comparison = summary.copy()
    mlp_validation_path = PACKAGE_ROOT / "results/final/mlp_log_hpo100_20260824/validation.json"
    mlp_rows_path = PACKAGE_ROOT / "results/final/mlp_log_hpo100_20260824/test15_evaluated_rows.csv"
    if mlp_validation_path.exists() and mlp_rows_path.exists():
        mlp_validation = json.loads(mlp_validation_path.read_text(encoding="utf-8"))
        mlp_rows = pd.read_csv(mlp_rows_path)
        mlp_property_metrics: list[dict[str, Any]] = []
        for prop in PROPERTIES:
            subset = mlp_rows[mlp_rows["property"] == prop]
            true_ae = np.abs(
                subset["base_prediction"].to_numpy(dtype=float)
                - subset["y_true"].to_numpy(dtype=float)
            )
            predicted_ae = subset["predicted_ae"].to_numpy(dtype=float)
            mlp_property_metrics.append(
                {"n": int(len(subset)), **metric_values(true_ae, predicted_ae)}
            )
        mlp_row = {
            "family": "current_mlp_log_hpo100",
            "selected_trial_id": int(mlp_validation["best_trial_id"]),
            "weighted_cv_spearman": float(mlp_validation["best_weighted_cv_spearman"]),
            # The original MLP trial table did not persist OOF predictions or R2.
            "weighted_cv_r2": np.nan,
            "weighted_test15_spearman": float(mlp_validation["weighted_test15_spearman"]),
            "weighted_test15_r2": weighted(mlp_property_metrics, "r2"),
            "weighted_test15_mae": weighted(mlp_property_metrics, "mae"),
            "weighted_test15_rmse": weighted(mlp_property_metrics, "rmse"),
            "model_path": str(mlp_validation["checkpoint"]),
        }
        comparison = pd.concat([comparison, pd.DataFrame([mlp_row])], ignore_index=True)
    comparison = comparison.sort_values("weighted_test15_spearman", ascending=False)
    atomic_csv(comparison, FINAL_ROOT / "comparison_with_current_mlp.csv")
    readme = [
        "# Reliability ML model-family HPO100",
        "",
        "The run keeps the v3.1 Log-MLP Reliability data, 32 features, sample weights,",
        "property-specific log(AE + epsilon) targets, and leakage-safe grouped five-fold CV.",
        "Each family uses 100 deterministic random candidates. Selection uses validation",
        "Spearman to remain comparable to Log-MLP-HPO100; validation R2 and the R2 winner",
        "are also saved. Test15 is opened only after each selected model and label-free",
        "prediction file are frozen.",
        "",
        "Families: Random Forest (rf), Extra Trees (extratrees), Histogram Gradient",
        "Boosting (histgb), and XGBoost (xgb).",
        "",
        "Important files:",
        "",
        "- `results/hpo/ml_model_families_hpo100_20260825/<family>/trials.csv`",
        "- `results/hpo/ml_model_families_hpo100_20260825/<family>/fold_metrics.csv`",
        "- `results/hpo/ml_model_families_hpo100_20260825/<family>/best_params.json`",
        "- `results/final/ml_model_families_hpo100_20260825/model_family_summary.csv`",
        "- `results/final/ml_model_families_hpo100_20260825/comparison_with_current_mlp.csv`",
        "- `models/reliability_model/ml_model_families_hpo100_20260825/`",
        "",
        "R2, MAE, and RMSE are computed in AE space. `log_r2` is a separate diagnostic",
        "in the property-specific transformed target space. Weighted cross-property",
        "values are sample-count-weighted means of property-specific metrics; they are",
        "not a pooled regression across incompatible physical units.",
    ]
    path = FINAL_ROOT / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(readme) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def validate_inputs(families: list[str], n_trials: int) -> dict[str, Any]:
    required = [
        PACKAGE_ROOT / "results/intermediate/calibration_features_ae_pre.csv",
        PACKAGE_ROOT / "results/intermediate/test_features_ae_pre.csv",
        PACKAGE_ROOT / "results/intermediate/test_labels_ae_pre.csv",
        PACKAGE_ROOT / "data/splits/deployment/deployment.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing HPO inputs:\n" + "\n".join(missing))
    if "xgb" in families:
        xgboost = importlib.import_module("xgboost")
        xgb_version = xgboost.__version__
    else:
        xgb_version = None
    calibration, _, _ = prepare_base_frames()
    return {
        "status": "passed",
        "package_root": str(PACKAGE_ROOT),
        "families": families,
        "n_random_trials_per_family": n_trials,
        "calibration_rows": int(len(calibration)),
        "n_features": len(FEATURE_COLUMNS),
        "xgboost_version": xgb_version,
        "test15_used_for_hpo_selection": False,
    }


def main() -> None:
    args = parse_args()
    families = list(dict.fromkeys(args.families))
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.smoke_test:
        args.n_trials = min(args.n_trials, 2)
        args.workers = min(args.workers, 2)
    check = validate_inputs(families, args.n_trials)
    if args.check:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return
    calibration, _, _ = prepare_base_frames()
    prepare_cv_cache(calibration, reset=args.reset)
    selections: dict[str, dict[str, Any]] = {}
    for family in families:
        selections[family] = run_family(
            family, args.n_trials, args.workers, args.reset, args.smoke_test
        )
        print(
            f"[{family}] selected trial {selections[family]['best_by_spearman']['trial_id']}: "
            f"weighted CV Spearman={selections[family]['best_by_spearman']['weighted_val_spearman']:.6f}; "
            f"R2={selections[family]['best_by_spearman']['weighted_val_r2']:.6f}",
            flush=True,
        )
    if args.hpo_only or args.smoke_test:
        return
    calibration_final, test_final, solvent_summary = load_final_frames()
    frozen: dict[str, tuple[Path, Path]] = {}
    for family in families:
        frozen[family] = train_selected_and_freeze_predictions(
            family, selections[family], calibration_final, test_final, solvent_summary
        )
        print(f"[{family}] final model and label-free Test15 predictions frozen", flush=True)
    validations = []
    for family in families:
        prediction_path, model_path = frozen[family]
        validation = evaluate_frozen_predictions(
            family, prediction_path, model_path, test_final, selections[family]
        )
        validations.append(validation)
        print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    write_cross_family_summary(validations)


if __name__ == "__main__":
    main()
