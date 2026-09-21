from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from data_utils import get_scaffold_smiles, mol_from_smiles
from rank_ensemble_features import PROPERTIES, add_calibration_scaffold_error_features


step03 = importlib.import_module("03_train_rf_leaf1_sqrt_oob")

SCRIPT_DIR = Path(__file__).resolve().parent
PLAN_VERSION = "rf_24f_staircase_v2"
MODEL_FEATURE_COLUMNS = list(step03.MODEL_FEATURE_COLUMNS)
EPSILON_BY_PROPERTY = dict(step03.EPSILON_BY_PROPERTY)
BASE_PARAMS: dict[str, Any] = {
    "n_estimators": 150,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "max_depth": None,
    "max_leaf_nodes": None,
    "max_samples": None,
    "criterion": "squared_error",
}
SEARCH_GRIDS: dict[str, list[Any]] = {
    "min_samples_leaf": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
    "max_features": [0.10, 0.15, "sqrt", 0.25, 0.35, 0.50, 0.70, 1.00],
    "max_depth": [6, 10, 14, 20, 28, 40, 60, None],
    "max_leaf_nodes": [64, 128, 256, 512, 1024, 2048, None],
    "min_samples_split": [2, 4, 8, 16, 32, 64],
    "max_samples": [0.40, 0.55, 0.70, 0.85, None],
    "criterion": ["squared_error", "friedman_mse"],
}
STABILITY_SEEDS = [(13, 17), (42, 42), (97, 89)]
LOG_PATH: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline, resumable staircase HPO for the 24-feature RF model.")
    parser.add_argument("--calibration_csv", default="../../results/intermediate/calibration_features_ae_pre.csv")
    parser.add_argument("--out_dir", default="../../results/hpo/staircase")
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--stop_after_stage", type=int, choices=range(0, 7), default=6)
    parser.add_argument("--reset", action="store_true", help="Delete only the selected HPO output directory before running.")
    parser.add_argument("--check", action="store_true", help="Validate the offline environment and Calibration input only.")
    parser.add_argument("--smoke_test", action="store_true", help="Run one tiny 2-fold trial in results/hpo_smoke.")
    return parser.parse_args()


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if LOG_PATH is not None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_gzip_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_id(stage: int, params: dict, cv_spec: dict, rf_seed: int) -> str:
    payload = {"stage": stage, "params": params, "cv": cv_spec, "rf_seed": rf_seed}
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:16]


def resolve_from_script(path: str) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else SCRIPT_DIR / value).resolve()


def load_and_validate_calibration(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df, features = step03.load_calibration_features(str(path))
    if list(features) != MODEL_FEATURE_COLUMNS:
        raise ValueError("The HPO feature order differs from the current Step 03 model feature order.")
    forbidden_splits = set(df["split"].astype(str)) - {"calibration"}
    if forbidden_splits:
        raise ValueError(f"HPO accepts Calibration rows only, found splits: {sorted(forbidden_splits)}")
    if len(MODEL_FEATURE_COLUMNS) != 24:
        raise ValueError(f"Expected the selected 24-feature scheme, found {len(MODEL_FEATURE_COLUMNS)} features.")
    return df


def add_direct_scaffold(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    cache: dict[str, str] = {}

    def scaffold(smiles: object) -> str:
        key = "" if pd.isna(smiles) else str(smiles).strip()
        if key not in cache:
            cache[key] = get_scaffold_smiles(mol_from_smiles(key)) if key else ""
        return cache[key]

    result["_direct_scaffold"] = result["smiles"].map(scaffold)
    return result


def canonical_pair_keys(df: pd.DataFrame) -> pd.Series:
    """Canonical solute-solvent grouping copied from Step 02 without importing PyTorch."""
    cache: dict[str, str] = {}

    def canonical(value: object) -> str:
        raw = "" if pd.isna(value) else str(value).strip()
        if raw not in cache:
            if not raw:
                cache[raw] = ""
            else:
                with rdBase.BlockLogs():
                    mol = Chem.MolFromSmiles(raw)
                cache[raw] = Chem.MolToSmiles(mol, canonical=True) if mol is not None else f"RAW::{raw}"
        return cache[raw]

    return df["smiles"].map(canonical) + "\x1f" + df["solvent"].map(canonical)


def manifest_payload(calibration_path: Path) -> dict:
    code_paths = [
        Path(__file__).resolve(),
        SCRIPT_DIR / "02_generate_ae_pre_merged_features.py",
        SCRIPT_DIR / "03_train_rf_leaf1_sqrt_oob.py",
        SCRIPT_DIR / "rank_ensemble_features.py",
    ]
    return {
        "plan_version": PLAN_VERSION,
        "calibration_csv": str(calibration_path),
        "calibration_sha256": file_sha256(calibration_path),
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "base_params": BASE_PARAMS,
        "search_grids": SEARCH_GRIDS,
        "code_sha256": {path.name: file_sha256(path) for path in code_paths},
        "sklearn_version": sklearn_version,
        "test_data_used": False,
    }


def initialize_run(out_dir: Path, calibration_path: Path, reset: bool) -> dict:
    if reset and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    current = manifest_payload(calibration_path)
    manifest_path = out_dir / "hpo_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        keys = [
            "plan_version",
            "calibration_sha256",
            "feature_columns",
            "base_params",
            "search_grids",
            "code_sha256",
        ]
        mismatches = [key for key in keys if existing.get(key) != current.get(key)]
        if mismatches:
            raise ValueError(
                "Existing HPO results are incompatible with the current input/code "
                f"({mismatches}). Run again with --reset to start a clean search."
            )
    else:
        atomic_json(current, manifest_path)
    return current


def cv_cache_key(manifest: dict, cv_spec: dict) -> str:
    payload = {
        "plan_version": PLAN_VERSION,
        "calibration_sha256": manifest["calibration_sha256"],
        "features": MODEL_FEATURE_COLUMNS,
        "cv": cv_spec,
        "code_sha256": manifest["code_sha256"],
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:16]


def build_or_load_cv_frames(
    df_with_scaffold: pd.DataFrame,
    out_dir: Path,
    manifest: dict,
    cv_spec: dict,
) -> list[pd.DataFrame]:
    key = cv_cache_key(manifest, cv_spec)
    cache_dir = out_dir / "cv_feature_cache" / key
    cache_manifest_path = cache_dir / "cache_manifest.json"
    expected = {
        "cache_key": key,
        "cv_spec": cv_spec,
        "calibration_sha256": manifest["calibration_sha256"],
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "code_sha256": manifest["code_sha256"],
        "historical_features_fit_on_cv_training_only": True,
    }
    fold_paths = [cache_dir / f"fold_{fold}.pkl" for fold in range(1, int(cv_spec["n_splits"]) + 1)]
    if cache_manifest_path.exists() and all(path.exists() for path in fold_paths):
        found = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if found == expected:
            log(f"Reusing leakage-safe CV feature cache {key}")
            return [pd.read_pickle(path) for path in fold_paths]
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    samples = df_with_scaffold[["row_index", "smiles", "solvent"]].drop_duplicates().copy()
    if samples["row_index"].duplicated().any():
        raise ValueError("A row_index maps to more than one solute-solvent pair.")
    samples["pair_group"] = canonical_pair_keys(samples)
    splitter = GroupKFold(
        n_splits=int(cv_spec["n_splits"]),
        shuffle=bool(cv_spec["shuffle"]),
        random_state=cv_spec["split_seed"] if cv_spec["shuffle"] else None,
    )
    keep_columns = list(
        dict.fromkeys(
            [
                "split",
                "property",
                "row_index",
                "fold",
                "y_true",
                "base_prediction",
                "log_abs_error",
                "sample_weight",
                "calibration_same_scaffold_count",
                *MODEL_FEATURE_COLUMNS,
            ]
        )
    )
    frames = []
    for cv_fold, (_, validation_positions) in enumerate(
        splitter.split(samples, groups=samples["pair_group"]),
        start=1,
    ):
        validation_ids = set(samples.iloc[validation_positions]["row_index"].astype(int))
        work = df_with_scaffold.copy()
        is_validation = work["row_index"].astype(int).isin(validation_ids)
        work["split"] = np.where(is_validation, "validation", "calibration")
        
        work.loc[is_validation, "split"] = "test"
        work.loc[is_validation, "abs_error"] = np.nan
        work = add_calibration_scaffold_error_features(work)
        work.loc[work["split"] == "test", "split"] = "validation"
        frame = work[keep_columns].copy()
        if not np.isfinite(frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)).all():
            raise ValueError(f"CV fold {cv_fold} contains non-finite model features.")
        if set(frame["split"].astype(str)) != {"calibration", "validation"}:
            raise ValueError(f"CV fold {cv_fold} has invalid roles.")
        tmp = fold_paths[cv_fold - 1].with_suffix(".pkl.tmp")
        frame.to_pickle(tmp)
        os.replace(tmp, fold_paths[cv_fold - 1])
        frames.append(frame)
        log(
            f"Prepared CV cache {key} fold {cv_fold}/{cv_spec['n_splits']}: "
            f"validation_samples={len(validation_ids)}"
        )
    atomic_json(expected, cache_manifest_path)
    return frames


def spearman(a, b) -> float:
    values = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return float("nan")
    ar = values["a"].rank(method="average").to_numpy(dtype=float)
    br = values["b"].rank(method="average").to_numpy(dtype=float)
    ar -= ar.mean()
    br -= br.mean()
    denominator = math.sqrt(float(np.sum(ar * ar) * np.sum(br * br)))
    return float(np.sum(ar * br) / denominator) if denominator else float("nan")


def inverse_log_error(values, prop: str) -> np.ndarray:
    return np.maximum(np.exp(np.asarray(values, dtype=float)) - EPSILON_BY_PROPERTY[prop], 0.0)


def forest_tree_bytes(model: RandomForestRegressor) -> int:
    total = 0
    for estimator in model.estimators_:
        state = estimator.tree_.__getstate__()
        total += sum(value.nbytes for value in state.values() if isinstance(value, np.ndarray))
    return int(total)


def weighted(values: pd.DataFrame, column: str, weight: str = "n") -> float:
    valid = values[[column, weight]].replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[valid[weight] > 0]
    return float(np.average(valid[column], weights=valid[weight]))


def evaluate_config(
    params: dict,
    cv_frames: list[pd.DataFrame],
    rf_seed: int,
    n_jobs: int,
    keep_oof: bool,
) -> tuple[dict, pd.DataFrame | None]:
    started = time.time()
    prediction_frames = []
    size_by_cv_fold: list[int] = []
    for cv_fold, frame in enumerate(cv_frames, start=1):
        fold_size = 0
        for prop in PROPERTIES:
            train = frame[(frame["split"] == "calibration") & (frame["property"] == prop)]
            validation = frame[(frame["split"] == "validation") & (frame["property"] == prop)]
            model = RandomForestRegressor(
                **params,
                bootstrap=True,
                oob_score=False,
                random_state=rf_seed,
                n_jobs=n_jobs,
            )
            model.fit(
                train[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float),
                train["log_abs_error"].to_numpy(dtype=float),
                sample_weight=train["sample_weight"].to_numpy(dtype=float),
            )
            fold_size += forest_tree_bytes(model)
            pred_ae = inverse_log_error(
                model.predict(validation[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)),
                prop,
            )
            pred = validation[
                ["property", "row_index", "fold", "y_true", "base_prediction", "calibration_same_scaffold_count"]
            ].copy()
            pred["predicted_ae"] = pred_ae
            pred["cv_fold"] = cv_fold
            prediction_frames.append(pred)
        size_by_cv_fold.append(fold_size)

    fold_rows = pd.concat(prediction_frames, ignore_index=True)
    sample_rows = (
        fold_rows.groupby(["property", "row_index"], as_index=False, sort=False)
        .agg(
            y_true=("y_true", "first"),
            base_prediction=("base_prediction", "mean"),
            predicted_ae=("predicted_ae", "mean"),
            same_scaffold_count=("calibration_same_scaffold_count", "first"),
            cv_fold=("cv_fold", "first"),
            n_fold_rows=("fold", "size"),
        )
    )
    row_counts = set(sample_rows["n_fold_rows"].astype(int))
    if len(row_counts) != 1 or not row_counts.issubset({1, 5}):
        raise ValueError(
            "OOF evaluation must contain consistently one ensemble row or five base-model rows "
            f"per sample/property; found {sorted(row_counts)}."
        )
    sample_rows["true_ae"] = np.abs(sample_rows["base_prediction"] - sample_rows["y_true"])

    property_rows = []
    for prop in PROPERTIES:
        sub = sample_rows[sample_rows["property"] == prop]
        zero = sub["same_scaffold_count"] == 0
        property_rows.append(
            {
                "property": prop,
                "n": int(len(sub)),
                "spearman": spearman(sub["predicted_ae"], sub["true_ae"]),
                "mae": float(np.mean(np.abs(sub["predicted_ae"] - sub["true_ae"]))),
                "n_zero": int(zero.sum()),
                "spearman_zero": spearman(sub.loc[zero, "predicted_ae"], sub.loc[zero, "true_ae"]),
                "n_nonzero": int((~zero).sum()),
                "spearman_nonzero": spearman(sub.loc[~zero, "predicted_ae"], sub.loc[~zero, "true_ae"]),
            }
        )
    prop_metrics = pd.DataFrame(property_rows)

    fold_scores = []
    for cv_fold in sorted(sample_rows["cv_fold"].unique()):
        per_property = []
        for prop in PROPERTIES:
            sub = sample_rows[(sample_rows["cv_fold"] == cv_fold) & (sample_rows["property"] == prop)]
            per_property.append({"n": len(sub), "spearman": spearman(sub["predicted_ae"], sub["true_ae"])})
        fold_scores.append(weighted(pd.DataFrame(per_property), "spearman"))

    result: dict[str, Any] = {
        "weighted_spearman": weighted(prop_metrics, "spearman"),
        "macro_spearman": float(prop_metrics["spearman"].mean()),
        "worst_task_spearman": float(prop_metrics["spearman"].min()),
        "same_scaffold_0_spearman": weighted(prop_metrics, "spearman_zero", "n_zero"),
        "same_scaffold_nonzero_spearman": weighted(prop_metrics, "spearman_nonzero", "n_nonzero"),
        "fold_weighted_spearman_mean": float(np.mean(fold_scores)),
        "fold_weighted_spearman_std": float(np.std(fold_scores, ddof=0)),
        "estimated_model_size_mb": float(np.mean(size_by_cv_fold) / (1024**2)),
        "runtime_seconds": float(time.time() - started),
    }
    for row in property_rows:
        prop = str(row["property"])
        result[f"{prop}_n"] = int(row["n"])
        result[f"{prop}_spearman"] = float(row["spearman"])
        result[f"{prop}_mae"] = float(row["mae"])
    oof = None
    if keep_oof:
        oof = sample_rows[
            [
                "property",
                "row_index",
                "cv_fold",
                "predicted_ae",
                "true_ae",
                "same_scaffold_count",
            ]
        ].copy()
    return result, oof


def dedupe_trials(trials: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for trial in trials:
        key = stable_json(trial["params"])
        if key in seen:
            continue
        seen.add(key)
        out.append(trial)
    return out


def run_stage(
    *,
    stage: int,
    name: str,
    trials: list[dict],
    cv_spec: dict,
    rf_seed: int,
    df_with_scaffold: pd.DataFrame,
    out_dir: Path,
    manifest: dict,
    n_jobs: int,
) -> pd.DataFrame:
    stage_path = out_dir / f"stage_{stage:02d}_{name}.csv"
    existing = pd.read_csv(stage_path) if stage_path.exists() else pd.DataFrame()
    if not existing.empty and {"status", "config_id"}.issubset(existing.columns):
        complete = existing[existing["status"] == "complete"].copy()
        if stage >= 3:
            if "oof_predictions_csv" not in complete.columns:
                complete = complete.iloc[0:0]
            else:
                complete = complete[
                    complete["oof_predictions_csv"].fillna("").map(lambda value: Path(str(value)).exists())
                ]
        complete_ids = set(complete["config_id"].astype(str))
    else:
        complete_ids = set()
    cv_frames = build_or_load_cv_frames(df_with_scaffold, out_dir, manifest, cv_spec)
    rows = existing.to_dict(orient="records") if not existing.empty else []

    for position, trial in enumerate(dedupe_trials(trials), start=1):
        cid = config_id(stage, trial["params"], cv_spec, rf_seed)
        if cid in complete_ids:
            log(f"Stage {stage} {position}/{len(trials)}: resume skip {cid}")
            continue
        config_path = out_dir / "configs" / f"stage_{stage:02d}" / f"{cid}.json"
        atomic_json(
            {
                "plan_version": PLAN_VERSION,
                "stage": stage,
                "config_id": cid,
                "label": trial["label"],
                "params": trial["params"],
                "cv_spec": cv_spec,
                "rf_seed": rf_seed,
                "test_data_used": False,
            },
            config_path,
        )
        log(f"Stage {stage} {position}/{len(trials)}: running {cid} {trial['label']}")
        metrics, oof = evaluate_config(trial["params"], cv_frames, rf_seed, n_jobs, keep_oof=stage >= 3)
        oof_path = ""
        if oof is not None:
            oof_file = out_dir / "oof_predictions" / f"stage_{stage:02d}" / f"{cid}.csv.gz"
            atomic_gzip_csv(oof, oof_file)
            oof_path = str(oof_file.resolve())
        row = {
            "status": "complete",
            "stage": stage,
            "config_id": cid,
            "label": trial["label"],
            "parent_config_id": trial.get("parent_config_id", ""),
            "changed_parameter": trial.get("changed_parameter", ""),
            "changed_value_json": stable_json(trial.get("changed_value")),
            "params_json": stable_json(trial["params"]),
            "n_splits": cv_spec["n_splits"],
            "shuffle": cv_spec["shuffle"],
            "split_seed": cv_spec["split_seed"],
            "rf_seed": rf_seed,
            "oof_predictions_csv": oof_path,
            **metrics,
        }
        rows.append(row)
        atomic_csv(pd.DataFrame(rows), stage_path)
        task_summary_path = write_task_spearman_summary(rows, stage_path)
        complete_ids.add(cid)
        log(
            f"Stage {stage} RESULT {position}/{len(dedupe_trials(trials))}: {cid}; "
            f"Abs={metrics['abs_spearman']:.6f}; Emi={metrics['emi_spearman']:.6f}; "
            f"Plqy={metrics['plqy_spearman']:.6f}; Em={metrics['em_spearman']:.6f}; "
            f"weighted={metrics['weighted_spearman']:.6f}; "
            f"params={stable_json(trial['params'])}; task_summary={task_summary_path.name}"
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"Stage {stage} produced no completed trials.")
    write_task_spearman_summary(rows, stage_path)
    return result[result["status"] == "complete"].copy()


def params_from_row(row: pd.Series) -> dict:
    return json.loads(str(row["params_json"]))


def write_task_spearman_summary(rows: list[dict], stage_path: Path) -> Path:
    summary_path = stage_path.with_name(f"{stage_path.stem}_task_spearman.csv")
    frame = pd.DataFrame(rows)
    parameter_columns = [
        "n_estimators",
        "min_samples_leaf",
        "min_samples_split",
        "max_features",
        "max_depth",
        "max_leaf_nodes",
        "max_samples",
        "criterion",
    ]
    if "params_json" not in frame.columns:
        raise ValueError("Cannot write task Spearman summary without params_json.")
    parsed_params = frame["params_json"].map(json.loads)
    for parameter in parameter_columns:
        frame[parameter] = parsed_params.map(lambda values: values.get(parameter))
    columns = [
        "status",
        "stage",
        "config_id",
        "label",
        *parameter_columns,
        "weighted_spearman",
        "plqy_spearman",
        "emi_spearman",
        "em_spearman",
        "abs_spearman",
    ]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot write task Spearman summary; missing columns: {missing}")
    atomic_csv(frame[columns], summary_path)
    return summary_path


def add_guard_columns(df: pd.DataFrame, baseline_label: str = "baseline") -> pd.DataFrame:
    result = df.copy()
    baseline_rows = result[result["label"] == baseline_label]
    if baseline_rows.empty:
        raise ValueError(f"Stage is missing {baseline_label!r} for guard comparisons.")
    baseline = baseline_rows.sort_values("weighted_spearman", ascending=False).iloc[0]
    task_ok = np.ones(len(result), dtype=bool)
    for prop in PROPERTIES:
        task_ok &= result[f"{prop}_spearman"].to_numpy(dtype=float) >= float(baseline[f"{prop}_spearman"]) - 0.02
    scaffold_ok = (
        result["same_scaffold_0_spearman"].to_numpy(dtype=float)
        >= float(baseline["same_scaffold_0_spearman"]) - 0.015
    )
    result["passes_task_guard"] = task_ok
    result["passes_scaffold_guard"] = scaffold_ok
    result["passes_all_guards"] = task_ok & scaffold_ok
    return result


def baseline_trial(n_estimators: int, label: str = "baseline") -> dict:
    params = dict(BASE_PARAMS)
    params["n_estimators"] = n_estimators
    return {"label": label, "params": params}


def stage1_trials() -> list[dict]:
    trials = [baseline_trial(150)]
    for parameter, values in SEARCH_GRIDS.items():
        for value in values:
            params = dict(BASE_PARAMS)
            params[parameter] = value
            trials.append(
                {
                    "label": f"one_factor:{parameter}={value}",
                    "params": params,
                    "changed_parameter": parameter,
                    "changed_value": value,
                }
            )
    return dedupe_trials(trials)


def one_factor_scores(stage1: pd.DataFrame, parameter: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for value in SEARCH_GRIDS[parameter]:
        candidate = dict(BASE_PARAMS)
        candidate[parameter] = value
        key = stable_json(candidate)
        matches = stage1[stage1["params_json"] == key]
        if not matches.empty:
            scores[stable_json(value)] = float(matches.iloc[0]["weighted_spearman"])
    return scores


def top_values(stage1: pd.DataFrame, parameter: str, count: int) -> list[Any]:
    scores = one_factor_scores(stage1, parameter)
    return sorted(
        SEARCH_GRIDS[parameter],
        key=lambda value: scores.get(stable_json(value), -np.inf),
        reverse=True,
    )[:count]


def stage2_trials(stage1: pd.DataFrame) -> list[dict]:
    best = {parameter: top_values(stage1, parameter, 1)[0] for parameter in SEARCH_GRIDS}
    core1 = dict(BASE_PARAMS)
    core1.update(best)
    core1["n_estimators"] = 200
    core2 = dict(BASE_PARAMS)
    for parameter in SEARCH_GRIDS:
        values = top_values(stage1, parameter, 2)
        core2[parameter] = values[min(1, len(values) - 1)]
    core2["n_estimators"] = 200

    trials = [baseline_trial(200)]
    for leaf in top_values(stage1, "min_samples_leaf", 4):
        for max_features in top_values(stage1, "max_features", 3):
            params = dict(core1)
            params.update(min_samples_leaf=leaf, max_features=max_features)
            trials.append({"label": "H1_leaf_x_features", "params": params})
    for depth in top_values(stage1, "max_depth", 3):
        for max_leaf_nodes in top_values(stage1, "max_leaf_nodes", 3):
            params = dict(core1)
            params.update(max_depth=depth, max_leaf_nodes=max_leaf_nodes)
            trials.append({"label": "H2_depth_x_leaf_nodes", "params": params})
    for core in [core1, core2]:
        for max_samples in top_values(stage1, "max_samples", 3):
            for min_split in top_values(stage1, "min_samples_split", 2):
                params = dict(core)
                params.update(max_samples=max_samples, min_samples_split=min_split)
                trials.append({"label": "H3_bootstrap_x_split", "params": params})

    presets = [
        (1, 0.15, None, None),
        (2, "sqrt", 40, None),
        (4, 0.25, 28, 2048),
        (6, 0.35, 20, 1024),
        (8, 0.50, 20, 512),
        (12, 0.50, 14, 512),
        (16, 0.70, 14, 256),
        (24, 1.00, 10, 128),
    ]
    for leaf, max_features, depth, leaf_nodes in presets:
        params = dict(core1)
        params.update(
            min_samples_leaf=leaf,
            max_features=max_features,
            max_depth=depth,
            max_leaf_nodes=leaf_nodes,
        )
        trials.append({"label": "H4_balanced_preset", "params": params})
    return dedupe_trials(trials)


def top_rows(df: pd.DataFrame, count: int, exclude_baseline: bool = False) -> pd.DataFrame:
    guarded = add_guard_columns(df)
    eligible = guarded[guarded["passes_all_guards"]].copy()
    if exclude_baseline:
        eligible = eligible[eligible["label"] != "baseline"]
    return eligible.sort_values("weighted_spearman", ascending=False).head(count)


def promoted_trials(df: pd.DataFrame, count: int, n_estimators: int, label: str) -> list[dict]:
    trials = [baseline_trial(n_estimators)]
    for _, row in top_rows(df, count, exclude_baseline=True).iterrows():
        params = params_from_row(row)
        params["n_estimators"] = n_estimators
        trials.append(
            {
                "label": label,
                "params": params,
                "parent_config_id": row["config_id"],
            }
        )
    return dedupe_trials(trials)


def adjacent_best_value(stage1: pd.DataFrame, parameter: str, current: Any) -> Any | None:
    values = SEARCH_GRIDS[parameter]
    if current not in values:
        return None
    index = values.index(current)
    adjacent = []
    if index > 0:
        adjacent.append(values[index - 1])
    if index + 1 < len(values):
        adjacent.append(values[index + 1])
    scores = one_factor_scores(stage1, parameter)
    if not adjacent:
        return None
    return max(adjacent, key=lambda value: scores.get(stable_json(value), -np.inf))


def stage4_trials(stage3: pd.DataFrame, stage1: pd.DataFrame) -> list[dict]:
    trials = [baseline_trial(300)]
    refine_parameters = [
        "min_samples_leaf",
        "max_features",
        "max_depth",
        "max_leaf_nodes",
        "max_samples",
        "min_samples_split",
    ]
    for _, row in top_rows(stage3, 4, exclude_baseline=True).iterrows():
        base = params_from_row(row)
        base["n_estimators"] = 300
        trials.append({"label": "local_parent", "params": base, "parent_config_id": row["config_id"]})
        for parameter in refine_parameters:
            value = adjacent_best_value(stage1, parameter, base.get(parameter))
            if value is None:
                continue
            params = dict(base)
            params[parameter] = value
            trials.append(
                {
                    "label": f"local:{parameter}",
                    "params": params,
                    "parent_config_id": row["config_id"],
                }
            )
    return dedupe_trials(trials)


def stage5_trials(stage4: pd.DataFrame) -> list[dict]:
    trials = [baseline_trial(150)]
    for _, row in top_rows(stage4, 5, exclude_baseline=True).iterrows():
        base = params_from_row(row)
        for n_estimators in [100, 150, 250, 400]:
            params = dict(base)
            params["n_estimators"] = n_estimators
            trials.append(
                {
                    "label": f"tree_count:{n_estimators}",
                    "params": params,
                    "parent_config_id": row["config_id"],
                }
            )
    return dedupe_trials(trials)


def save_guarded_stage(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    guarded = add_guard_columns(df)
    atomic_csv(guarded, path)
    return guarded


def run_stability(
    stage5: pd.DataFrame,
    df_with_scaffold: pd.DataFrame,
    out_dir: Path,
    manifest: dict,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict]:
    finalists = []
    for _, row in top_rows(stage5, 3, exclude_baseline=True).iterrows():
        finalists.append(
            {
                "label": "finalist",
                "params": params_from_row(row),
                "parent_config_id": row["config_id"],
            }
        )
    baseline = baseline_trial(150, label="stability_baseline")
    baseline["parent_config_id"] = "baseline_24f"
    finalists.append(baseline)

    all_rows = []
    for split_seed, rf_seed in STABILITY_SEEDS:
        cv_spec = {"n_splits": 5, "shuffle": True, "split_seed": split_seed}
        renamed = []
        for trial in finalists:
            item = dict(trial)
            item["label"] = f"{trial['label']}:split{split_seed}:rf{rf_seed}"
            renamed.append(item)
        result = run_stage(
            stage=6,
            name=f"stability_seed_{split_seed}_{rf_seed}",
            trials=renamed,
            cv_spec=cv_spec,
            rf_seed=rf_seed,
            df_with_scaffold=df_with_scaffold,
            out_dir=out_dir,
            manifest=manifest,
            n_jobs=n_jobs,
        )
        all_rows.append(result)
    stability = pd.concat(all_rows, ignore_index=True)
    atomic_csv(stability, out_dir / "stage_06_stability.csv")

    aggregate_rows = []
    for parent_id, group in stability.groupby("parent_config_id", sort=False):
        row = {
            "parent_config_id": parent_id,
            "params_json": group.iloc[0]["params_json"],
            "n_seed_pairs": len(group),
            "weighted_spearman_mean": float(group["weighted_spearman"].mean()),
            "weighted_spearman_std": float(group["weighted_spearman"].std(ddof=0)),
            "same_scaffold_0_spearman_mean": float(group["same_scaffold_0_spearman"].mean()),
            "same_scaffold_0_spearman_std": float(group["same_scaffold_0_spearman"].std(ddof=0)),
            "estimated_model_size_mb_mean": float(group["estimated_model_size_mb"].mean()),
        }
        row["robust_score"] = row["weighted_spearman_mean"] - 0.5 * row["weighted_spearman_std"]
        for prop in PROPERTIES:
            row[f"{prop}_spearman_mean"] = float(group[f"{prop}_spearman"].mean())
            row[f"{prop}_spearman_min"] = float(group[f"{prop}_spearman"].min())
            row[f"{prop}_mae_mean"] = float(group[f"{prop}_mae"].mean())
        aggregate_rows.append(row)
    leaderboard = pd.DataFrame(aggregate_rows).sort_values("robust_score", ascending=False).reset_index(drop=True)
    baseline_row = leaderboard[leaderboard["parent_config_id"] == "baseline_24f"].iloc[0]
    candidates = leaderboard[leaderboard["parent_config_id"] != "baseline_24f"].copy()
    candidates["passes_improvement"] = (
        candidates["weighted_spearman_mean"] >= float(baseline_row["weighted_spearman_mean"]) + 0.003
    )
    task_guard = np.ones(len(candidates), dtype=bool)
    for prop in PROPERTIES:
        task_guard &= (
            candidates[f"{prop}_spearman_min"].to_numpy(dtype=float)
            >= float(baseline_row[f"{prop}_spearman_min"]) - 0.02
        )
    candidates["passes_task_guard"] = task_guard
    candidates["passes_scaffold_guard"] = (
        candidates["same_scaffold_0_spearman_mean"]
        >= float(baseline_row["same_scaffold_0_spearman_mean"]) - 0.015
    )
    candidates["passes_size_guard"] = candidates["estimated_model_size_mb_mean"] <= 2048.0
    eligible = candidates[
        candidates["passes_improvement"]
        & candidates["passes_task_guard"]
        & candidates["passes_scaffold_guard"]
        & candidates["passes_size_guard"]
    ]
    if eligible.empty:
        selected = baseline_row
        reason = "No finalist passed the pre-registered improvement/task/size guards; retained baseline."
    else:
        best_robust = float(eligible["robust_score"].max())
        statistically_tied = eligible[eligible["robust_score"] >= best_robust - 0.002]
        selected = statistically_tied.sort_values(
            ["estimated_model_size_mb_mean", "robust_score"],
            ascending=[True, False],
        ).iloc[0]
        reason = "Selected the smallest model within 0.002 robust_score of the best eligible finalist."
    guard_columns = candidates[
        [
            "parent_config_id",
            "passes_improvement",
            "passes_task_guard",
            "passes_scaffold_guard",
            "passes_size_guard",
        ]
    ]
    leaderboard = leaderboard.merge(guard_columns, on="parent_config_id", how="left")
    baseline_mask = leaderboard["parent_config_id"] == "baseline_24f"
    leaderboard.loc[
        baseline_mask,
        ["passes_improvement", "passes_task_guard", "passes_scaffold_guard", "passes_size_guard"],
    ] = True
    atomic_csv(leaderboard, out_dir / "hpo_leaderboard.csv")
    selection = {
        "plan_version": PLAN_VERSION,
        "selected_parent_config_id": selected["parent_config_id"],
        "selected_params": json.loads(str(selected["params_json"])),
        "selected_robust_score": float(selected["robust_score"]),
        "selected_weighted_spearman_mean": float(selected["weighted_spearman_mean"]),
        "baseline_weighted_spearman_mean": float(baseline_row["weighted_spearman_mean"]),
        "reason": reason,
        "test_data_used": False,
        "ready_for_manual_review": True,
    }
    atomic_json(selection, out_dir / "hpo_final_selection.json")
    return leaderboard, selection


def main() -> None:
    global LOG_PATH
    args = parse_args()
    os.chdir(SCRIPT_DIR)
    calibration_path = resolve_from_script(args.calibration_csv)
    out_dir = resolve_from_script(args.out_dir)
    if args.smoke_test:
        out_dir = (SCRIPT_DIR / "results" / "hpo_smoke").resolve()
    LOG_PATH = None if args.check else out_dir / "hpo_run.log"

    df = load_and_validate_calibration(calibration_path)
    if args.check:
        log(
            f"CHECK PASSED: calibration_rows={len(df)}, features={len(MODEL_FEATURE_COLUMNS)}, "
            "Test inputs are not accepted by this program."
        )
        return
    manifest = initialize_run(out_dir, calibration_path, args.reset)
    df_with_scaffold = add_direct_scaffold(df)

    if args.smoke_test:
        trial = baseline_trial(10, label="smoke_baseline")
        trial["params"].update(max_depth=4, max_leaf_nodes=32)
        result = run_stage(
            stage=0,
            name="smoke",
            trials=[trial],
            cv_spec={"n_splits": 2, "shuffle": False, "split_seed": None},
            rf_seed=42,
            df_with_scaffold=df_with_scaffold,
            out_dir=out_dir,
            manifest=manifest,
            n_jobs=args.n_jobs,
        )
        log(f"SMOKE TEST PASSED: weighted_spearman={result.iloc[0]['weighted_spearman']:.6f}")
        return

    stage0 = run_stage(
        stage=0,
        name="baseline",
        trials=[baseline_trial(150)],
        cv_spec={"n_splits": 5, "shuffle": False, "split_seed": None},
        rf_seed=42,
        df_with_scaffold=df_with_scaffold,
        out_dir=out_dir,
        manifest=manifest,
        n_jobs=args.n_jobs,
    )
    baseline_score = float(stage0.iloc[0]["weighted_spearman"])
    if abs(baseline_score - 0.584462) > 0.002:
        raise ValueError(
            f"Stage 0 baseline {baseline_score:.6f} does not reproduce 0.584462 within tolerance."
        )
    if args.stop_after_stage == 0:
        return

    stage1_path = out_dir / "stage_01_one_factor.csv"
    stage1 = run_stage(
        stage=1,
        name="one_factor",
        trials=stage1_trials(),
        cv_spec={"n_splits": 3, "shuffle": True, "split_seed": 42},
        rf_seed=42,
        df_with_scaffold=df_with_scaffold,
        out_dir=out_dir,
        manifest=manifest,
        n_jobs=args.n_jobs,
    )
    stage1 = save_guarded_stage(stage1, stage1_path)
    if args.stop_after_stage == 1:
        return

    stage2_path = out_dir / "stage_02_interactions.csv"
    stage2 = run_stage(
        stage=2,
        name="interactions",
        trials=stage2_trials(stage1),
        cv_spec={"n_splits": 3, "shuffle": True, "split_seed": 97},
        rf_seed=42,
        df_with_scaffold=df_with_scaffold,
        out_dir=out_dir,
        manifest=manifest,
        n_jobs=args.n_jobs,
    )
    stage2 = save_guarded_stage(stage2, stage2_path)
    if args.stop_after_stage == 2:
        return

    stage3_path = out_dir / "stage_03_full_5fold.csv"
    stage3 = run_stage(
        stage=3,
        name="full_5fold",
        trials=promoted_trials(stage2, 12, 300, "stage2_top12"),
        cv_spec={"n_splits": 5, "shuffle": True, "split_seed": 42},
        rf_seed=42,
        df_with_scaffold=df_with_scaffold,
        out_dir=out_dir,
        manifest=manifest,
        n_jobs=args.n_jobs,
    )
    stage3 = save_guarded_stage(stage3, stage3_path)
    if args.stop_after_stage == 3:
        return

    stage4_path = out_dir / "stage_04_local_refine.csv"
    stage4 = run_stage(
        stage=4,
        name="local_refine",
        trials=stage4_trials(stage3, stage1),
        cv_spec={"n_splits": 5, "shuffle": True, "split_seed": 97},
        rf_seed=42,
        df_with_scaffold=df_with_scaffold,
        out_dir=out_dir,
        manifest=manifest,
        n_jobs=args.n_jobs,
    )
    stage4 = save_guarded_stage(stage4, stage4_path)
    if args.stop_after_stage == 4:
        return

    stage5_path = out_dir / "stage_05_tree_count_size.csv"
    stage5 = run_stage(
        stage=5,
        name="tree_count_size",
        trials=stage5_trials(stage4),
        cv_spec={"n_splits": 5, "shuffle": True, "split_seed": 42},
        rf_seed=42,
        df_with_scaffold=df_with_scaffold,
        out_dir=out_dir,
        manifest=manifest,
        n_jobs=args.n_jobs,
    )
    stage5 = save_guarded_stage(stage5, stage5_path)
    if args.stop_after_stage == 5:
        return

    leaderboard, selection = run_stability(stage5, df_with_scaffold, out_dir, manifest, args.n_jobs)
    log(f"HPO COMPLETE: selected={selection['selected_parent_config_id']}")
    log(f"Leaderboard: {out_dir / 'hpo_leaderboard.csv'}")
    log(f"Selection: {out_dir / 'hpo_final_selection.json'}")


if __name__ == "__main__":
    main()
