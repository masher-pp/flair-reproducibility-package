from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHONWARNINGS", "ignore")
_CPU_COUNT = str(os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", _CPU_COUNT)
os.environ.setdefault("MKL_NUM_THREADS", _CPU_COUNT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _CPU_COUNT)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _CPU_COUNT)

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from rdkit import RDLogger, rdBase

    RDLogger.DisableLog("rdApp.*")
    rdBase.DisableLog("rdApp.*")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import config
from data_utils import (
    build_reference_cache,
    extract_features_for_error_model,
    records_from_dataframe,
    standardize_columns,
)
from model_interface import MyPredictionModel


def log(message: str) -> None:
    print(message, flush=True)


def torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_cfg(checkpoint_path: str | Path) -> Dict:
    ckpt = torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"{checkpoint_path} is not a dict checkpoint.")
    raw_cfg = ckpt.get("cfg", {})
    if hasattr(raw_cfg, "__dict__"):
        raw_cfg = vars(raw_cfg)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"{checkpoint_path} does not contain a usable cfg dict.")
    return dict(raw_cfg)


def resolve_path(
    raw_path: Optional[str],
    *,
    checkpoint_path: str | Path,
    extra_roots: Sequence[str | Path] = (),
    required: bool,
    description: str,
) -> Optional[str]:
    if raw_path is None or str(raw_path).strip() == "" or str(raw_path).lower() in {"none", "nan", "null"}:
        if required:
            raise FileNotFoundError(f"Missing {description}.")
        return None

    raw = os.path.expandvars(os.path.expanduser(str(raw_path).strip().strip('"').strip("'")))
    p = Path(raw)
    roots = [
        *[Path(x) for x in extra_roots if x is not None and str(x).strip()],
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(checkpoint_path).expanduser().resolve().parent,
        *Path(checkpoint_path).expanduser().resolve().parents,
    ]
    candidates = [p] if p.is_absolute() else [root / p for root in roots] + [p]
    seen = set()
    unique = []
    for item in candidates:
        key = str(item)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    for item in unique:
        if item.exists():
            return str(item.resolve())
    legacy_name = p.name.lower()
    replacement_name: Optional[str] = None
    if legacy_name in {"development.csv", "flair_db.csv"} or re.fullmatch(
        r"(?:train|validation)_fold_\d+\.csv", p.name, flags=re.IGNORECASE
    ):
        replacement_name = "deployment.csv"
    elif legacy_name == "holdout_test.csv":
        replacement_name = "deployment_test.csv"

    if replacement_name is not None:
        fallback_candidates = [item.parent / replacement_name for item in unique]
        fallback_candidates.extend(item.parent / "deployment" / replacement_name for item in unique)
        fallback_candidates.extend(root / "data" / "splits" / replacement_name for root in roots)
        fallback_candidates.extend(
            root / "data" / "splits" / "deployment" / replacement_name for root in roots
        )
        for item in fallback_candidates:
            if item.exists():
                return str(item.resolve())
    if required:
        tried = "\n".join(f"  - {x}" for x in unique)
        raise FileNotFoundError(f"Cannot find {description}: {raw}\nTried:\n{tried}")
    return str(unique[0])


def auto_dataset_paths(
    checkpoint_path: str,
    calibration_csv: Optional[str] = None,
    reference_csv: Optional[str] = None,
    ae_test_csv: Optional[str] = None,
    extra_roots: Sequence[str | Path] = (),
) -> Dict[str, object]:
    cfg = checkpoint_cfg(checkpoint_path)
    fold = cfg.get("CV_FOLD")
    if fold is None:
        for value in (cfg.get("TRAIN_CSV"), cfg.get("VAL_CSV"), checkpoint_path):
            match = re.search(r"fold[_ -]*0*(\d+)", str(value or ""), flags=re.IGNORECASE)
            if match:
                fold = int(match.group(1))
                break
    reference = reference_csv or cfg.get("TRAIN_CSV")
    calibration = calibration_csv or cfg.get("VAL_CSV") or cfg.get("TEST_CSV")
    test = ae_test_csv or cfg.get("TEST_CSV")
    resolved_reference = resolve_path(reference, checkpoint_path=checkpoint_path, extra_roots=extra_roots, required=True, description="reference/train CSV")
    resolved_calibration = resolve_path(calibration, checkpoint_path=checkpoint_path, extra_roots=extra_roots, required=True, description="calibration CSV")
    resolved_test = resolve_path(test, checkpoint_path=checkpoint_path, extra_roots=extra_roots, required=False, description="AE test CSV")
    fold_number = int(fold) if fold is not None else None
    return {
        "reference_csv": resolved_reference,
        "calibration_csv": resolved_calibration,
        "ae_test_csv": resolved_test,
        "cv_fold": fold_number,
        "reference_fold": fold_number if resolved_reference and Path(resolved_reference).name == "deployment.csv" else None,
        "calibration_fold": fold_number if resolved_calibration and Path(resolved_calibration).name == "deployment.csv" else None,
    }


def read_dataset_view(csv_path: str, cv_fold: Optional[int] = None, fold_role: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if cv_fold is None:
        return df
    role = str(fold_role or "").strip().lower()
    if "cv_fold" not in df.columns:
        raise ValueError(f"CSV {csv_path} is missing the cv_fold column required for fold selection.")
    if role not in {"train", "valid", "validation"}:
        raise ValueError(f"fold_role must be 'train' or 'valid', got {fold_role!r}.")
    values = pd.to_numeric(df["cv_fold"], errors="raise").astype(int)
    mask = values.ne(int(cv_fold)) if role == "train" else values.eq(int(cv_fold))
    out = df.loc[mask].copy().reset_index(drop=True)
    if "split" in out.columns:
        out["split"] = "train" if role == "train" else "valid"
    return out


def resolve_best_dir(best_dir: Optional[str] = "Best") -> Path:
    candidates: List[Path] = []
    if best_dir:
        raw = Path(os.path.expanduser(os.path.expandvars(str(best_dir))))
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.extend([
                Path.cwd() / raw,
                Path(__file__).resolve().parent / raw,
                Path.cwd().parent / raw,
                Path(__file__).resolve().parent.parent / raw,
            ])
    else:
        candidates.extend([
            Path.cwd() / "Best",
            Path(__file__).resolve().parent / "Best",
            Path.cwd().parent / "Best",
            Path(__file__).resolve().parent.parent / "Best",
        ])

    seen = set()
    for cand in candidates:
        try:
            resolved = cand.expanduser().resolve()
        except Exception:
            resolved = cand.expanduser().absolute()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_dir():
            return resolved

    tried = "\n".join(f"  - {x}" for x in candidates)
    raise FileNotFoundError(f"Cannot find Best directory. Tried:\n{tried}")


def scaffold_fold_number(path: Path) -> int:
    match = re.search(r"fold[_ -]*(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 9999


def discover_scaffold_checkpoints(best_dir: Optional[str] = "Best", expected: int = 5) -> List[str]:
    root = resolve_best_dir(best_dir)
    paths = sorted(
        root.glob("fold_*/best_model.pt"),
        key=lambda p: (scaffold_fold_number(p), str(p)),
    )
    if len(paths) != expected:
        found = "\n".join(f"  - {p}" for p in paths) or "  <none>"
        raise ValueError(
            f"Expected exactly {expected} Scaffold checkpoints under {root}, found {len(paths)}.\n{found}"
        )
    return [str(p.resolve()) for p in paths]


def target_columns_for_model(model: MyPredictionModel) -> List[str]:
    names = [str(x).lower() for x in getattr(model, "target_names", config.MAIN_MODEL_TARGET_ORDER)]
    return names or list(config.MAIN_MODEL_TARGET_ORDER)


def set_system_eval(system) -> None:
    system.mol_encoder.eval()
    system.sol_encoder.eval()
    if getattr(system, "use_fusion_self_attention", False):
        system.fusion_encoder.eval()
    system.mlp.eval()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    delta = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(delta ** 2)))


def configured_feature_jobs() -> int:
    n_jobs = int(getattr(config, "AE_FEATURE_N_JOBS", -1))
    if n_jobs == 0:
        return 1
    return n_jobs


def manifest_for_model(
    model: MyPredictionModel,
    csv_path: str,
    cv_fold: Optional[int] = None,
    fold_role: Optional[str] = None,
) -> pd.DataFrame:
    dl = model.dl
    cfg = model.cfg
    return dl.build_manifest_df(
        csv_path=csv_path,
        solvent_mode=getattr(cfg, "SOLVENT_MODE", "morgan"),
        drop_invalid_solvent_in_graph=bool(getattr(cfg, "DROP_INVALID_SOLVENT_IN_GRAPH", True)),
        cv_fold=cv_fold,
        fold_role=fold_role,
    )


def loader_for_manifest(model: MyPredictionModel, manifest_df: pd.DataFrame, batch_size: int):
    dl = model.dl
    cfg = model.cfg
    ds = dl.FluorSolventDataset(
        df=manifest_df,
        solvent_mode=getattr(cfg, "SOLVENT_MODE", "morgan"),
        drop_invalid_solvent_in_graph=bool(getattr(cfg, "DROP_INVALID_SOLVENT_IN_GRAPH", True)),
        target_mean=model.target_mean,
        target_std=model.target_std,
        cache_graphs=bool(getattr(cfg, "CACHE_GRAPHS", False)),
        cgsd_csv_path=getattr(cfg, "CGSD_CSV_PATH", None),
        morgan_kwargs={
            "radius": int(getattr(cfg, "MORGAN_RADIUS", 4)),
            "n_bits": int(getattr(cfg, "MORGAN_N_BITS", 256)),
            "use_chirality": bool(getattr(cfg, "MORGAN_USE_CHIRALITY", False)),
        },
        rdkit_kwargs={
            "descriptor_names": getattr(cfg, "RDKIT_DESCRIPTOR_NAMES", None),
            "nan_value": float(getattr(cfg, "RDKIT_NAN_VALUE", 0.0)),
            "inf_value": float(getattr(cfg, "RDKIT_INF_VALUE", 0.0)),
        },
    )
    return model.PyGDataLoader(ds, batch_size=batch_size, shuffle=False, follow_batch=["solvent_x"])


@torch.no_grad()
def predict_csv_with_checkpoint(
    model: MyPredictionModel,
    csv_path: str,
    output_csv: Optional[str] = None,
    batch_size: int = 256,
    cv_fold: Optional[int] = None,
    fold_role: Optional[str] = None,
) -> pd.DataFrame:
    manifest_df = manifest_for_model(model, csv_path, cv_fold=cv_fold, fold_role=fold_role)
    loader = loader_for_manifest(model, manifest_df, batch_size=batch_size)

    system = model.system
    set_system_eval(system)
    pred_scaled_rows: List[np.ndarray] = []
    total_batches = len(loader)
    log(f"[Predict] {Path(csv_path).name}: rows={len(manifest_df)}, batches={total_batches}, device={system.device}")

    for batch_idx, batch in enumerate(loader, start=1):
        batch = model.tr.move_batch_to_device(batch, system.device)
        x_vec = model.tr.build_x_vec(batch, system)
        y_scaled = system.mlp(x_vec)
        pred_scaled_rows.append(y_scaled.detach().cpu().numpy())
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0:
            log(f"[Predict] {Path(csv_path).name}: batch {batch_idx}/{total_batches}")

    if pred_scaled_rows:
        pred_scaled = np.vstack(pred_scaled_rows).astype(np.float32)
    else:
        pred_scaled = np.zeros((0, len(config.MAIN_MODEL_TARGET_ORDER)), dtype=np.float32)

    target_mean = np.asarray(model.target_mean, dtype=np.float32).reshape(1, -1)
    target_std = np.asarray(model.target_std, dtype=np.float32).reshape(1, -1)
    pred_raw = pred_scaled * target_std + target_mean

    out = manifest_df.copy()
    target_names = target_columns_for_model(model)
    for i, name in enumerate(target_names):
        if i >= pred_raw.shape[1]:
            break
        out[f"pred_{name}"] = pred_raw[:, i]
        if name in out.columns:
            out[f"true_{name}"] = pd.to_numeric(out[name], errors="coerce")
            out[f"ae_{name}"] = np.abs(out[f"pred_{name}"].to_numpy(dtype=float) - out[f"true_{name}"].to_numpy(dtype=float))

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
    return out


def safe_float(mapping: dict, key: str, default: float) -> float:
    try:
        return float(mapping.get(key, default))
    except Exception:
        return float(default)


def smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smooth_loss_weights(abs_error: np.ndarray, prop: str) -> np.ndarray:
    floor = safe_float(config.ERROR_LOSS_FLOOR_BY_PROPERTY, prop, 0.0)
    width = safe_float(config.ERROR_LOSS_SMOOTH_WIDTH_BY_PROPERTY, prop, max(floor, 1e-12))
    width = max(width, 1e-12)
    return smoothstep01((np.asarray(abs_error, dtype=float) - floor) / width)


def log_error_target(abs_error: np.ndarray, prop: str) -> np.ndarray:
    eps = safe_float(config.ERROR_EPSILON_BY_PROPERTY, prop, 1e-8)
    return np.log(np.asarray(abs_error, dtype=float) + eps)


def inverse_log_error(log_error: np.ndarray, prop: str) -> np.ndarray:
    eps = safe_float(config.ERROR_EPSILON_BY_PROPERTY, prop, 1e-8)
    return np.maximum(np.exp(np.asarray(log_error, dtype=float)) - eps, 0.0)


def build_reference_info(
    reference_csv: str,
    cv_fold: Optional[int] = None,
    fold_role: Optional[str] = None,
) -> Dict:
    log(f"[Reference] building reference features from {Path(reference_csv).name}")
    ref_df = read_dataset_view(reference_csv, cv_fold=cv_fold, fold_role=fold_role)
    return build_reference_info_from_df(ref_df, source=str(Path(reference_csv).resolve()))


def build_reference_info_from_df(ref_df: pd.DataFrame, source: str) -> Dict:
    ref_df = standardize_columns(ref_df)
    records = records_from_dataframe(ref_df)
    solute_mols = [r["solute_mol"] for r in records if r.get("solute_mol") is not None]
    solvent_mols = [r["solvent_mol"] for r in records if r.get("solvent_mol") is not None]
    if not solute_mols:
        raise ValueError(f"No valid solute molecules in reference data: {source}")
    log(f"[Reference] valid solute={len(solute_mols)}, valid solvent={len(solvent_mols)}")
    solute_cache = build_reference_cache(solute_mols, n_components=config.DESCRIPTOR_PCA_COMPONENTS)
    solvent_cache = {"mode": "raw_ecfp"} if config.USE_SOLVENT_FEATURES else None
    log("[Reference] reference feature cache ready")
    return {
        "reference_csv": source,
        "records": records,
        "solute_cache": solute_cache,
        "solvent_cache": solvent_cache,
    }


def features_from_prediction_df(pred_df: pd.DataFrame, reference_info: Dict, prop: str) -> Tuple[np.ndarray, List[str], List[int]]:
    pred_df = standardize_columns(pred_df.copy())
    records = records_from_dataframe(pred_df)
    tasks = []

    for idx, rec in enumerate(records):
        pred_col = f"pred_{prop}"
        if pred_col not in pred_df.columns:
            continue
        pred_val = pd.to_numeric(pd.Series([pred_df.iloc[idx][pred_col]]), errors="coerce").iloc[0]
        if not np.isfinite(pred_val):
            continue
        tasks.append((idx, rec, float(pred_val)))

    if not tasks:
        raise ValueError(f"No usable features for {prop}.")

    n_jobs = configured_feature_jobs()
    backend = str(getattr(config, "AE_FEATURE_BACKEND", "threading"))
    log(f"[Features] {prop}: rows={len(tasks)}, n_jobs={n_jobs}, backend={backend}")

    def _one_feature(row_idx: int, rec: Dict, pred_val: float):
        features, names = extract_features_for_error_model(
            solute_mol=rec.get("solute_mol"),
            solute_cache=reference_info["solute_cache"],
            solvent_mol=rec.get("solvent_mol"),
            solvent_cache=reference_info.get("solvent_cache"),
            fold_variance=0.0,
            base_prediction=pred_val,
        )
        return row_idx, features, names

    if n_jobs == 1 or len(tasks) < 2:
        rows = [_one_feature(row_idx, rec, pred_val) for row_idx, rec, pred_val in tasks]
    else:
        parallel_kwargs = {"n_jobs": n_jobs, "backend": backend}
        if backend in {"threading", "threads"}:
            parallel_kwargs["prefer"] = "threads"
        rows = Parallel(**parallel_kwargs)(
            delayed(_one_feature)(row_idx, rec, pred_val) for row_idx, rec, pred_val in tasks
        )

    used_rows = [row_idx for row_idx, _, _ in rows]
    X = [features for _, features, _ in rows]
    feature_names = rows[0][2] if rows else []
    return np.vstack(X), list(feature_names or []), used_rows


def fit_one_ae_model(pred_df: pd.DataFrame, reference_info: Dict, prop: str) -> Tuple[RandomForestRegressor, Dict]:
    pred_col = f"pred_{prop}"
    true_col = f"true_{prop}" if f"true_{prop}" in pred_df.columns else prop
    if pred_col not in pred_df.columns or true_col not in pred_df.columns:
        raise ValueError(f"Missing {pred_col} or {true_col} for AE training.")

    X_all, feature_names, used_rows = features_from_prediction_df(pred_df, reference_info, prop)
    sub = pred_df.iloc[used_rows].copy()
    y_true = pd.to_numeric(sub[true_col], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(sub[pred_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    X = X_all[valid]
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    return fit_one_ae_model_from_arrays(X, y_true, y_pred, feature_names, prop)


def fit_one_ae_model_from_arrays(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: Sequence[str],
    prop: str,
) -> Tuple[RandomForestRegressor, Dict]:
    if len(y_true) < config.ERROR_MODEL_MIN_SAMPLES:
        raise ValueError(f"{prop} has only {len(y_true)} valid calibration samples.")

    abs_error = np.abs(y_pred - y_true)
    y_log = log_error_target(abs_error, prop)
    weights = smooth_loss_weights(abs_error, prop)
    if np.sum(weights) <= 0:
        weights = np.ones_like(abs_error, dtype=float)

    rf = RandomForestRegressor(**config.ERROR_RF_PARAMS)
    rf.fit(X, y_log, sample_weight=weights)
    pred_log = rf.predict(X)
    pred_ae = inverse_log_error(pred_log, prop)

    info = {
        "property": prop,
        "n_samples": int(len(y_true)),
        "n_weight_positive": int(np.sum(weights > 0)),
        "main_model_pre_true_mae": float(mean_absolute_error(y_true, y_pred)),
        "ae_model_train_mae": float(mean_absolute_error(abs_error, pred_ae)),
        "ae_model_train_rmse": rmse(abs_error, pred_ae),
        "log_error_train_mae": float(mean_absolute_error(y_log, pred_log)),
        "feature_names": list(feature_names),
    }
    return rf, info


def evaluate_ae_models(pred_df: pd.DataFrame, reference_info: Dict, models: Dict[str, RandomForestRegressor]) -> Dict[str, Dict]:
    metrics: Dict[str, Dict] = {}
    for prop, model in models.items():
        pred_col = f"pred_{prop}"
        true_col = f"true_{prop}" if f"true_{prop}" in pred_df.columns else prop
        if pred_col not in pred_df.columns or true_col not in pred_df.columns:
            continue
        X_all, _, used_rows = features_from_prediction_df(pred_df, reference_info, prop)
        sub = pred_df.iloc[used_rows].copy()
        y_true = pd.to_numeric(sub[true_col], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(sub[pred_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(valid):
            continue
        abs_error = np.abs(y_pred[valid] - y_true[valid])
        pred_ae = inverse_log_error(model.predict(X_all[valid]), prop)
        metrics[prop] = {
            "n_samples": int(len(abs_error)),
            "main_model_pre_true_mae": float(mean_absolute_error(y_true[valid], y_pred[valid])),
            "ae_model_mae": float(mean_absolute_error(abs_error, pred_ae)),
            "ae_model_rmse": rmse(abs_error, pred_ae),
            "test_log_error_mae": float(mean_absolute_error(log_error_target(abs_error, prop), np.log(pred_ae + safe_float(config.ERROR_EPSILON_BY_PROPERTY, prop, 1e-8)))),
        }
    return metrics


def train_ae_from_checkpoint(
    checkpoint_path: str,
    calibration_csv: Optional[str],
    reference_csv: Optional[str],
    ae_test_csv: Optional[str],
    output_dir: str,
    batch_size: int,
) -> Dict:
    output = Path(output_dir)
    results_dir = output / config.RESULTS_DIR
    model_dir = output / config.MODEL_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    paths = auto_dataset_paths(
        checkpoint_path,
        calibration_csv=calibration_csv,
        reference_csv=reference_csv,
        ae_test_csv=ae_test_csv,
    )
    model = MyPredictionModel.load(
        checkpoint_path,
        train_csv_override=paths["reference_csv"],
        cv_fold_override=paths["reference_fold"],
    )
    reference_info = build_reference_info(
        paths["reference_csv"],
        cv_fold=paths["reference_fold"],
        fold_role="train" if paths["reference_fold"] is not None else None,
    )

    calibration_pred_path = results_dir / "calibration_predictions.csv"
    calibration_pred_df = predict_csv_with_checkpoint(
        model,
        paths["calibration_csv"],
        output_csv=str(calibration_pred_path),
        batch_size=batch_size,
        cv_fold=paths["calibration_fold"],
        fold_role="valid" if paths["calibration_fold"] is not None else None,
    )

    ae_models: Dict[str, RandomForestRegressor] = {}
    per_property: Dict[str, Dict] = {}
    for prop in config.PROPERTIES:
        ae_model, info = fit_one_ae_model(calibration_pred_df, reference_info, prop)
        ae_models[prop] = ae_model
        per_property[prop] = info

    test_metrics = None
    test_pred_path = None
    if paths.get("ae_test_csv") and Path(paths["ae_test_csv"]).exists():
        test_pred_path = results_dir / "ae_test_predictions.csv"
        test_pred_df = predict_csv_with_checkpoint(
            model,
            paths["ae_test_csv"],
            output_csv=str(test_pred_path),
            batch_size=batch_size,
        )
        test_metrics = evaluate_ae_models(test_pred_df, reference_info, ae_models)

    bundle = {
        "version": "ae_from_single_checkpoint_v1",
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "paths": paths,
        "models": ae_models,
        "reference_info": reference_info,
        "per_property": per_property,
        "test_metrics": test_metrics,
        "properties": list(config.PROPERTIES),
        "epsilon_by_property": dict(config.ERROR_EPSILON_BY_PROPERTY),
    }
    model_path = model_dir / "ae_from_pt.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    summary = {
        "ae_model_path": str(model_path.resolve()),
        "calibration_predictions": str(calibration_pred_path.resolve()),
        "ae_test_predictions": str(test_pred_path.resolve()) if test_pred_path else None,
        "paths": paths,
        "per_property": per_property,
        "test_metrics": test_metrics,
    }
    with open(results_dir / "ae_from_pt_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame(per_property.values()).to_csv(results_dir / "ae_from_pt_summary.csv", index=False)
    return summary


def combine_reference_csvs(reference_csvs: Sequence[tuple[str, Optional[int]]]) -> pd.DataFrame:
    frames = []
    for path, fold in reference_csvs:
        df = read_dataset_view(path, cv_fold=fold, fold_role="train" if fold is not None else None)
        frames.append(standardize_columns(df))
    if not frames:
        raise ValueError("No reference CSVs to combine.")
    combined = pd.concat(frames, ignore_index=True)
    subset = [c for c in [config.SMILES_COLUMN, config.SOLVENT_COLUMN, *config.PROPERTIES] if c in combined.columns]
    if subset:
        combined = combined.drop_duplicates(subset=subset).reset_index(drop=True)
    return combined


def collect_feature_rows_from_fold(
    pred_df: pd.DataFrame,
    reference_info: Dict,
    prop: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    pred_col = f"pred_{prop}"
    true_col = f"true_{prop}" if f"true_{prop}" in pred_df.columns else prop
    X_all, feature_names, used_rows = features_from_prediction_df(pred_df, reference_info, prop)
    sub = pred_df.iloc[used_rows].copy()
    y_true = pd.to_numeric(sub[true_col], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(sub[pred_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return X_all[valid], y_true[valid], y_pred[valid], feature_names


def train_ae_from_scaffold_best(
    best_dir: Optional[str],
    split_dir: Optional[str],
    output_dir: str,
    batch_size: int,
) -> Dict:
    checkpoint_paths = discover_scaffold_checkpoints(best_dir=best_dir, expected=5)
    best_root = resolve_best_dir(best_dir)
    log("[Start] AE-from-pt Scaffold training")
    log(f"[Start] Best dir: {best_root}")
    log(
        "[Start] CPU threads: "
        f"OMP={os.environ.get('OMP_NUM_THREADS')}, "
        f"MKL={os.environ.get('MKL_NUM_THREADS')}, "
        f"feature_n_jobs={getattr(config, 'AE_FEATURE_N_JOBS', 1)}"
    )
    for path in checkpoint_paths:
        log(f"[Start] checkpoint: {path}")
    extra_roots: List[str | Path] = [
        Path(__file__).resolve().parent,
        best_root,
        best_root.parent,
    ]
    if split_dir:
        extra_roots.insert(0, split_dir)

    output = Path(output_dir)
    results_dir = output / config.RESULTS_DIR
    model_dir = output / config.MODEL_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_bank = {
        prop: {"X": [], "y_true": [], "y_pred": [], "feature_names": None}
        for prop in config.PROPERTIES
    }
    fold_paths = []
    reference_csvs = []
    calibration_frames = []
    test_metric_rows = []

    for fold_idx, checkpoint_path in enumerate(checkpoint_paths, start=1):
        log(f"[Fold {fold_idx}/5] resolving CSV paths")
        paths = auto_dataset_paths(
            checkpoint_path,
            calibration_csv=None,
            reference_csv=None,
            ae_test_csv=None,
            extra_roots=extra_roots,
        )
        fold_paths.append(paths)
        reference_csvs.append((paths["reference_csv"], paths["reference_fold"]))
        log(f"[Fold {fold_idx}/5] train={paths['reference_csv']}")
        log(f"[Fold {fold_idx}/5] val={paths['calibration_csv']}")
        log(f"[Fold {fold_idx}/5] test={paths['ae_test_csv']}")

        log(f"[Fold {fold_idx}/5] loading main checkpoint")
        model = MyPredictionModel.load(
            checkpoint_path,
            train_csv_override=paths["reference_csv"],
            cv_fold_override=paths["reference_fold"],
        )
        log(f"[Fold {fold_idx}/5] main checkpoint loaded")
        reference_info = build_reference_info(
            paths["reference_csv"],
            cv_fold=paths["reference_fold"],
            fold_role="train" if paths["reference_fold"] is not None else None,
        )

        calibration_pred_path = results_dir / f"calibration_predictions_scaffold_{fold_idx}.csv"
        log(f"[Fold {fold_idx}/5] predicting calibration CSV")
        calibration_pred_df = predict_csv_with_checkpoint(
            model,
            paths["calibration_csv"],
            output_csv=str(calibration_pred_path),
            batch_size=batch_size,
            cv_fold=paths["calibration_fold"],
            fold_role="valid" if paths["calibration_fold"] is not None else None,
        )
        calibration_pred_df.insert(0, "fold", fold_idx)
        calibration_pred_df.insert(1, "checkpoint_path", checkpoint_path)
        calibration_frames.append(calibration_pred_df)

        for prop in config.PROPERTIES:
            log(f"[Fold {fold_idx}/5] collecting AE features: {prop}")
            X, y_true, y_pred, feature_names = collect_feature_rows_from_fold(
                calibration_pred_df,
                reference_info,
                prop,
            )
            feature_bank[prop]["X"].append(X)
            feature_bank[prop]["y_true"].append(y_true)
            feature_bank[prop]["y_pred"].append(y_pred)
            feature_bank[prop]["feature_names"] = feature_names

    ae_models: Dict[str, RandomForestRegressor] = {}
    per_property: Dict[str, Dict] = {}
    for prop in config.PROPERTIES:
        log(f"[AE Fit] fitting AE model: {prop}")
        X = np.vstack(feature_bank[prop]["X"])
        y_true = np.concatenate(feature_bank[prop]["y_true"])
        y_pred = np.concatenate(feature_bank[prop]["y_pred"])
        feature_names = feature_bank[prop]["feature_names"] or []
        ae_model, info = fit_one_ae_model_from_arrays(X, y_true, y_pred, feature_names, prop)
        ae_models[prop] = ae_model
        per_property[prop] = info
        log(f"[AE Fit] {prop} done: n={info['n_samples']}, MAE={info['ae_model_train_mae']:.4f}")

    for fold_idx, (checkpoint_path, paths) in enumerate(zip(checkpoint_paths, fold_paths), start=1):
        test_csv = paths.get("ae_test_csv")
        if not test_csv or not Path(test_csv).exists():
            continue
        log(f"[Fold {fold_idx}/5] evaluating AE on test CSV")
        model = MyPredictionModel.load(
            checkpoint_path,
            train_csv_override=paths["reference_csv"],
            cv_fold_override=paths["reference_fold"],
        )
        reference_info = build_reference_info(
            paths["reference_csv"],
            cv_fold=paths["reference_fold"],
            fold_role="train" if paths["reference_fold"] is not None else None,
        )
        test_pred_path = results_dir / f"ae_test_predictions_scaffold_{fold_idx}.csv"
        test_pred_df = predict_csv_with_checkpoint(
            model,
            test_csv,
            output_csv=str(test_pred_path),
            batch_size=batch_size,
        )
        metrics = evaluate_ae_models(test_pred_df, reference_info, ae_models)
        for prop, row in metrics.items():
            item = {"fold": fold_idx, "property": prop, **row}
            test_metric_rows.append(item)

    calibration_all = pd.concat(calibration_frames, ignore_index=True)
    calibration_all.to_csv(results_dir / "calibration_predictions_scaffold_all.csv", index=False)

    log("[Deploy] building combined deployment reference")
    combined_reference_df = combine_reference_csvs(reference_csvs)
    deployment_reference_info = build_reference_info_from_df(
        combined_reference_df,
        source="combined Scaffold train CSVs",
    )

    test_metrics = None
    if test_metric_rows:
        test_metrics_df = pd.DataFrame(test_metric_rows)
        test_metrics_df.to_csv(results_dir / "ae_test_metrics_scaffold.csv", index=False)
        test_metrics = {
            prop: test_metrics_df[test_metrics_df["property"] == prop].drop(columns=["property"]).to_dict("records")
            for prop in sorted(test_metrics_df["property"].unique())
        }

    bundle = {
        "version": "ae_from_scaffold_checkpoints_v1",
        "best_dir": str(best_root.resolve()),
        "checkpoint_paths": checkpoint_paths,
        "fold_paths": fold_paths,
        "models": ae_models,
        "reference_info": deployment_reference_info,
        "per_property": per_property,
        "test_metrics": test_metrics,
        "properties": list(config.PROPERTIES),
        "epsilon_by_property": dict(config.ERROR_EPSILON_BY_PROPERTY),
    }
    model_path = model_dir / "ae_from_pt_scaffold.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    summary = {
        "ae_model_path": str(model_path.resolve()),
        "best_dir": str(best_root.resolve()),
        "checkpoint_paths": checkpoint_paths,
        "fold_paths": fold_paths,
        "calibration_predictions": str((results_dir / "calibration_predictions_scaffold_all.csv").resolve()),
        "per_property": per_property,
        "test_metrics": test_metrics,
    }
    with open(results_dir / "ae_from_pt_scaffold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame(per_property.values()).to_csv(results_dir / "ae_from_pt_scaffold_summary.csv", index=False)
    log(f"[Done] AE model saved to {model_path.resolve()}")
    return summary


def load_ae_bundle(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_single_with_ae(checkpoint_path: str, ae_model_path: str, smiles: str, solvent: Optional[str]) -> Dict:
    bundle = load_ae_bundle(ae_model_path)
    if bundle.get("version") == "ae_from_scaffold_checkpoints_v1":
        fold_preds = []
        checkpoint_paths = bundle.get("checkpoint_paths", [])
        fold_paths = bundle.get("fold_paths", [])
        for idx, path in enumerate(checkpoint_paths):
            ref_csv = None
            if idx < len(fold_paths):
                ref_csv = fold_paths[idx].get("reference_csv")
            main_model = MyPredictionModel.load(path, train_csv_override=ref_csv)
            fold_preds.append(main_model.predict(smiles=smiles, solvent=solvent, solute_mol=None, solvent_mol=None))
        main_pred = {}
        main_pred_std = {}
        for prop in config.PROPERTIES:
            vals = np.asarray([p.get(prop, np.nan) for p in fold_preds], dtype=float)
            vals = vals[np.isfinite(vals)]
            main_pred[prop] = float(np.mean(vals)) if vals.size else float("nan")
            main_pred_std[prop] = float(np.std(vals)) if vals.size else float("nan")
    else:
        main_model = MyPredictionModel.load(checkpoint_path, train_csv_override=bundle.get("paths", {}).get("reference_csv"))
        main_pred = main_model.predict(smiles=smiles, solvent=solvent, solute_mol=None, solvent_mol=None)
        main_pred_std = None

    row = {config.SMILES_COLUMN: smiles, config.SOLVENT_COLUMN: "" if solvent is None else solvent}
    for prop, value in main_pred.items():
        row[f"pred_{prop}"] = value
    pred_df = pd.DataFrame([row])

    ae_pred = {}
    for prop, model in bundle["models"].items():
        X, _, _ = features_from_prediction_df(pred_df, bundle["reference_info"], prop)
        ae_pred[prop] = float(inverse_log_error(model.predict(X), prop)[0])
    result = {"predictions": main_pred, "predicted_ae": ae_pred}
    if main_pred_std is not None:
        result["prediction_std_across_5_scaffold_models"] = main_pred_std
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit/predict AE model from main-model .pt checkpoint(s); no OOF files are required. "
            "Running without arguments is equivalent to: fit_scaffold --best_dir ../../models/main_model "
            "--split_dir ../../data/splits/deployment"
        )
    )
    sub = parser.add_subparsers(dest="mode")

    fit = sub.add_parser("fit")
    fit.add_argument("--checkpoint", required=True, help="Path to best_model.pt or compatible checkpoint.")
    fit.add_argument("--calibration_csv", default=None, help="Defaults to VAL_CSV stored in the checkpoint.")
    fit.add_argument("--reference_csv", default=None, help="Defaults to TRAIN_CSV stored in the checkpoint.")
    fit.add_argument("--ae_test_csv", default=None, help="Optional independent CSV for AE-model evaluation.")
    fit.add_argument("--output_dir", default=".")
    fit.add_argument("--batch_size", type=int, default=256)

    fit_scaffold = sub.add_parser("fit_scaffold")
    fit_scaffold.add_argument("--best_dir", default="../../models/main_model", help="Folder containing fold_01..fold_05/best_model.pt.")
    fit_scaffold.add_argument(
        "--split_dir",
        default="../../data/splits/deployment",
        help="Folder containing deployment.csv and deployment_test.csv.",
    )
    fit_scaffold.add_argument("--output_dir", default=".")
    fit_scaffold.add_argument("--batch_size", type=int, default=256)

    pred = sub.add_parser("predict")
    pred.add_argument("--checkpoint", default=None, help="Only required for a single-checkpoint AE model. Scaffold AE bundles store all 5 paths.")
    pred.add_argument("--ae_model", default=os.path.join(config.MODEL_DIR, "ae_from_pt_scaffold.pkl"))
    pred.add_argument("--smiles", required=True)
    pred.add_argument("--solvent", default=None)

    args = parser.parse_args()
    if args.mode is None:
        args.mode = "fit_scaffold"
        args.best_dir = "../../models/main_model"
        args.split_dir = "../../data/splits/deployment"
        args.output_dir = "."
        args.batch_size = 256
    return args


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    args = parse_args()
    if args.mode == "fit":
        summary = train_ae_from_checkpoint(
            checkpoint_path=args.checkpoint,
            calibration_csv=args.calibration_csv,
            reference_csv=args.reference_csv,
            ae_test_csv=args.ae_test_csv,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.mode == "fit_scaffold":
        summary = train_ae_from_scaffold_best(
            best_dir=args.best_dir,
            split_dir=args.split_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.mode == "predict":
        if args.checkpoint is None:
            bundle = load_ae_bundle(args.ae_model)
            if bundle.get("version") != "ae_from_scaffold_checkpoints_v1":
                raise ValueError("--checkpoint is required when --ae_model is not a Scaffold 5-checkpoint bundle.")
            args.checkpoint = bundle.get("checkpoint_paths", [None])[0]
        result = predict_single_with_ae(
            checkpoint_path=args.checkpoint,
            ae_model_path=args.ae_model,
            smiles=args.smiles,
            solvent=args.solvent,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
