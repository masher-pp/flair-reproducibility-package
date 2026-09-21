from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
import config
from data_utils import (
    build_reference_cache,
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
    stored_name = p.name.lower()
    replacement_name: Optional[str] = None
    if stored_name in {"development.csv", "flair_db.csv"} or re.fullmatch(
        r"(?:train|validation)_fold_\d+\.csv", p.name, flags=re.IGNORECASE
    ):
        replacement_name = "deployment.csv"
    elif stored_name == "holdout_test.csv":
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


def smoothstep01(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def smooth_loss_weights(abs_error: np.ndarray, prop: str) -> np.ndarray:
    floor = safe_float(config.ERROR_LOSS_FLOOR_BY_PROPERTY, prop, 0.0)
    width = safe_float(
        config.ERROR_LOSS_SMOOTH_WIDTH_BY_PROPERTY,
        prop,
        max(floor, 1e-12),
    )
    width = max(width, 1e-12)
    return smoothstep01((np.asarray(abs_error, dtype=float) - floor) / width)


def log_error_target(abs_error: np.ndarray, prop: str) -> np.ndarray:
    epsilon = safe_float(config.ERROR_EPSILON_BY_PROPERTY, prop, 1e-8)
    return np.log(np.asarray(abs_error, dtype=float) + epsilon)


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

