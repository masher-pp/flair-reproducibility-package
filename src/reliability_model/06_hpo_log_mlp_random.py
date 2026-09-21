from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
HPO_DIR = PACKAGE_ROOT / "results/hpo/mlp_log_hpo100_20260824"
FINAL_DIR = PACKAGE_ROOT / "results/final/mlp_log_hpo100_20260824"
RELIABILITY_DIR = PACKAGE_ROOT / "src/reliability_model"
BASELINE_METRICS = HPO_DIR / "baseline_log_mlp_test15_metrics.csv"

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


FEATURES = load_module("reliability_features", RELIABILITY_DIR / "reliability_features.py")
STEP02 = importlib.import_module("02_generate_ae_pre_merged_features")
PROPERTIES = list(FEATURES.PROPERTIES)
EPSILON = dict(FEATURES.EPSILON_BY_PROPERTY)
FEATURE_COLUMNS = list(FEATURES.SOLVENT_AUGMENTED_MODEL_FEATURE_COLUMNS)
PROPERTY_INDEX = {prop: index for index, prop in enumerate(PROPERTIES)}

BASELINE_CONFIG = {
    "hidden1": 128,
    "hidden2": 64,
    "activation": "relu",
    "dropout": 0.0,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 500,
}

_CV_FRAMES: list[pd.DataFrame] | None = None
_SMOKE_MODE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe 100-trial random-search HPO for the Log-L1 MLP Reliability model."
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--hpo-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


@contextmanager
def suppress_native_training_warnings():
    """Silence the x86 MKL-on-Apple-Silicon warning emitted for every matmul."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with open(os.devnull, "w") as sink:
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def activation(name: str) -> torch.nn.Module:
    if name == "relu":
        return torch.nn.ReLU()
    if name == "leaky_relu":
        return torch.nn.LeakyReLU(negative_slope=0.05)
    if name == "silu":
        return torch.nn.SiLU()
    if name == "gelu":
        return torch.nn.GELU()
    raise KeyError(name)


def build_network(n_features: int, config: dict[str, Any]) -> torch.nn.Module:
    layers: list[torch.nn.Module] = [
        torch.nn.Linear(n_features, int(config["hidden1"])),
        activation(str(config["activation"])),
    ]
    if float(config["dropout"]) > 0:
        layers.append(torch.nn.Dropout(float(config["dropout"])))
    layers.extend(
        [
            torch.nn.Linear(int(config["hidden1"]), int(config["hidden2"])),
            activation(str(config["activation"])),
        ]
    )
    if float(config["dropout"]) > 0:
        layers.append(torch.nn.Dropout(float(config["dropout"])))
    layers.append(torch.nn.Linear(int(config["hidden2"]), 1))
    return torch.nn.Sequential(*layers)


class FittedLogMLP:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.y_center: float | None = None
        self.y_scale: float | None = None
        self.network: torch.nn.Module | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int) -> "FittedLogMLP":
        seed_everything(seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        self.x_mean = x.mean(axis=0)
        self.x_scale = x.std(axis=0)
        self.x_scale[self.x_scale < 1e-8] = 1.0
        self.y_center = float(np.median(y))
        q25, q75 = np.percentile(y, [25, 75])
        self.y_scale = float(max(q75 - q25, np.std(y), 1e-6))
        xs = (x - self.x_mean) / self.x_scale
        ys = (y - self.y_center) / self.y_scale
        self.network = build_network(x.shape[1], self.config)
        if self.config["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=float(self.config["learning_rate"]),
                weight_decay=float(self.config["weight_decay"]),
            )
        else:
            optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=float(self.config["learning_rate"]),
                weight_decay=float(self.config["weight_decay"]),
            )
        xt = torch.from_numpy(xs)
        yt = torch.from_numpy(ys).reshape(-1, 1)
        wt = torch.from_numpy(weights).reshape(-1, 1)
        denominator = torch.clamp(wt.sum(), min=1e-12)
        self.network.train()
        with suppress_native_training_warnings():
            for _ in range(int(self.config["epochs"])):
                optimizer.zero_grad(set_to_none=True)
                prediction = self.network(xt)
                loss = (wt * torch.abs(prediction - yt)).sum() / denominator
                loss.backward()
                optimizer.step()
            self.network.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.network is None or self.x_mean is None or self.x_scale is None:
            raise RuntimeError("Model is not fitted")
        xs = (np.asarray(x, dtype=np.float32) - self.x_mean) / self.x_scale
        with torch.no_grad():
            scaled = self.network(torch.from_numpy(xs)).numpy().reshape(-1)
        return scaled * float(self.y_scale) + float(self.y_center)

    def state_payload(self) -> dict[str, Any]:
        if self.network is None:
            raise RuntimeError("Model is not fitted")
        return {
            "config": self.config,
            "x_mean": self.x_mean,
            "x_scale": self.x_scale,
            "y_center": self.y_center,
            "y_scale": self.y_scale,
            "state_dict": {key: value.detach().cpu() for key, value in self.network.state_dict().items()},
        }


def inverse_log(values: np.ndarray, prop: str) -> np.ndarray:
    return np.maximum(np.exp(np.asarray(values, dtype=float)) - float(EPSILON[prop]), 0.0)


def prepare_base_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    calibration_path = PACKAGE_ROOT / "results/intermediate/calibration_features_ae_pre.csv"
    test_path = PACKAGE_ROOT / "results/intermediate/test_features_ae_pre.csv"
    reference_path = PACKAGE_ROOT / "data/splits/deployment/deployment.csv"
    calibration, _ = FEATURES.load_calibration_features(
        str(calibration_path), feature_columns=FEATURE_COLUMNS
    )
    test = FEATURES.load_test_features(str(test_path))
    calibration = FEATURES.aggregate_base_predictions(calibration, include_targets=True)
    test = FEATURES.aggregate_base_predictions(test, include_targets=False)
    calibration, test, solvent_summary = FEATURES.add_solvent_reference_feature_columns(
        calibration, test, str(reference_path)
    )
    scaffold_cache: dict[str, str] = {}

    def scaffold(smiles: object) -> str:
        key = "" if pd.isna(smiles) else str(smiles).strip()
        if key not in scaffold_cache:
            scaffold_cache[key] = FEATURES.get_scaffold_smiles(FEATURES.mol_from_smiles(key)) if key else ""
        return scaffold_cache[key]

    calibration["_direct_scaffold"] = calibration["smiles"].map(scaffold)
    test["_direct_scaffold"] = test["smiles"].map(scaffold)
    return calibration, test, solvent_summary


def load_prepared_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    calibration_path = PACKAGE_ROOT / "results/intermediate/calibration_features_ae_pre.csv"
    test_path = PACKAGE_ROOT / "results/intermediate/test_features_ae_pre.csv"
    reference_path = PACKAGE_ROOT / "data/splits/deployment/deployment.csv"
    calibration, feature_columns = FEATURES.load_calibration_features(
        str(calibration_path), feature_columns=FEATURE_COLUMNS
    )
    test = FEATURES.load_test_features(str(test_path))
    calibration = FEATURES.aggregate_base_predictions(calibration, include_targets=True)
    test = FEATURES.aggregate_base_predictions(test, include_targets=False)
    calibration, test = FEATURES.rebuild_ensemble_history_features(calibration, test)
    calibration, test, solvent_summary = FEATURES.add_solvent_reference_feature_columns(
        calibration, test, str(reference_path)
    )
    return calibration, test, feature_columns, solvent_summary


def prepare_cv_cache(calibration: pd.DataFrame, reset: bool = False) -> list[pd.DataFrame]:
    cache_dir = HPO_DIR / "cv_cache"
    paths = [cache_dir / f"fold_{index}.pkl" for index in range(1, 6)]
    manifest_path = cache_dir / "manifest.json"
    source_path = PACKAGE_ROOT / "results/intermediate/calibration_features_ae_pre.csv"
    expected = {
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
    frames: list[pd.DataFrame] = []
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
    for cv_fold, (_, validation_positions) in enumerate(
        splitter.split(samples, groups=samples["pair_group"]), start=1
    ):
        validation_ids = set(samples.iloc[validation_positions]["row_index"].astype(int))
        work = calibration.copy()
        is_validation = work["row_index"].astype(int).isin(validation_ids)
        work["split"] = np.where(is_validation, "test", "calibration")
        work.loc[is_validation, "abs_error"] = np.nan
        work = FEATURES.add_calibration_scaffold_error_features(work)
        work.loc[is_validation, "split"] = "validation"
        for prop in PROPERTIES:
            train_mask = (work["split"] == "calibration") & (work["property"] == prop)
            train_errors = work.loc[train_mask, "abs_error"].to_numpy(dtype=float)
            work.loc[train_mask, "sample_weight"] = FEATURES.smooth_loss_weights(train_errors, prop)
        frame = work[keep_columns].copy()
        if not np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite feature in CV fold {cv_fold}")
        if frame.loc[frame["split"] == "calibration", "sample_weight"].isna().any():
            raise ValueError(f"Missing training weights in CV fold {cv_fold}")
        frame.to_pickle(paths[cv_fold - 1])
        frames.append(frame)
        print(
            f"Prepared CV fold {cv_fold}/5: validation row_index count={len(validation_ids)}",
            flush=True,
        )
    atomic_json(expected, manifest_path)
    return frames


def loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def sample_configs(n_trials: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260824)
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(configs) < n_trials:
        hidden1 = int(rng.choice([64, 96, 128, 160, 192, 256]))
        hidden2_choices = [value for value in [32, 48, 64, 96, 128] if value <= hidden1]
        config = {
            "hidden1": hidden1,
            "hidden2": int(rng.choice(hidden2_choices)),
            "activation": str(rng.choice(["relu", "leaky_relu", "silu", "gelu"])),
            "dropout": float(rng.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])),
            "optimizer": str(rng.choice(["adam", "adamw"])),
            "learning_rate": loguniform(rng, 1e-4, 5e-3),
            "weight_decay": loguniform(rng, 1e-7, 2e-3),
            "epochs": 3 if _SMOKE_MODE else int(rng.choice([150, 250, 350, 500, 650])),
        }
        key = stable_json(config)
        if key in seen:
            continue
        seen.add(key)
        configs.append(config)
    return configs


def init_worker(cache_dir: str) -> None:
    global _CV_FRAMES
    torch.set_num_threads(1)
    _CV_FRAMES = [pd.read_pickle(Path(cache_dir) / f"fold_{index}.pkl") for index in range(1, 6)]


def evaluate_config(trial_id: int, config: dict[str, Any]) -> dict[str, Any]:
    if _CV_FRAMES is None:
        raise RuntimeError("Worker CV cache not initialized")
    started = time.time()
    predictions: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {prop: [] for prop in PROPERTIES}
    fold_weighted_scores: list[float] = []
    for cv_fold, frame in enumerate(_CV_FRAMES, start=1):
        fold_metric_rows = []
        for prop in PROPERTIES:
            train = frame[(frame["split"] == "calibration") & (frame["property"] == prop)]
            validation = frame[(frame["split"] == "validation") & (frame["property"] == prop)]
            x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
            y_train = np.log(train["abs_error"].to_numpy(dtype=float) + float(EPSILON[prop]))
            weights = train["sample_weight"].to_numpy(dtype=float)
            model_seed = 42 + cv_fold * 100 + PROPERTY_INDEX[prop]
            model = FittedLogMLP(config).fit(x_train, y_train, weights, model_seed)
            predicted_ae = inverse_log(
                model.predict(validation[FEATURE_COLUMNS].to_numpy(dtype=float)), prop
            )
            true_ae = np.abs(
                validation["base_prediction"].to_numpy(dtype=float)
                - validation["y_true"].to_numpy(dtype=float)
            )
            predictions[prop].append((predicted_ae, true_ae))
            fold_metric_rows.append(
                {"n": len(validation), "spearman": FEATURES.spearman_manual(predicted_ae, true_ae)}
            )
        fold_metrics = pd.DataFrame(fold_metric_rows)
        fold_weighted_scores.append(
            float(np.average(fold_metrics["spearman"], weights=fold_metrics["n"]))
        )
    result: dict[str, Any] = {
        "trial_id": trial_id,
        "config_id": hashlib.sha256(stable_json(config).encode("utf-8")).hexdigest()[:16],
        **config,
    }
    prop_rows = []
    for prop in PROPERTIES:
        pred = np.concatenate([item[0] for item in predictions[prop]])
        true = np.concatenate([item[1] for item in predictions[prop]])
        score = FEATURES.spearman_manual(pred, true)
        result[f"{prop}_val_spearman"] = score
        prop_rows.append({"property": prop, "n": len(true), "spearman": score})
    metric_frame = pd.DataFrame(prop_rows)
    result["weighted_val_spearman"] = float(
        np.average(metric_frame["spearman"], weights=metric_frame["n"])
    )
    result["macro_val_spearman"] = float(metric_frame["spearman"].mean())
    result["worst_property_val_spearman"] = float(metric_frame["spearman"].min())
    result["fold_weighted_spearman_mean"] = float(np.mean(fold_weighted_scores))
    result["fold_weighted_spearman_std"] = float(np.std(fold_weighted_scores, ddof=0))
    result["runtime_seconds"] = float(time.time() - started)
    return result


def run_search(n_trials: int, workers: int, reset: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    HPO_DIR.mkdir(parents=True, exist_ok=True)
    trial_path = HPO_DIR / "trials.csv"
    configs = sample_configs(n_trials)
    existing = pd.DataFrame()
    if trial_path.exists() and not reset:
        existing = pd.read_csv(trial_path)
    completed_ids = set(existing["trial_id"].astype(int)) if not existing.empty else set()
    pending = [(index + 1, config) for index, config in enumerate(configs) if index + 1 not in completed_ids]
    baseline_path = HPO_DIR / "baseline_cv.json"
    cache_dir = HPO_DIR / "cv_cache"
    if reset or not baseline_path.exists():
        init_worker(str(cache_dir))
        baseline_result = evaluate_config(0, BASELINE_CONFIG)
        atomic_json(baseline_result, baseline_path)
        print(
            f"Baseline CV weighted Spearman={baseline_result['weighted_val_spearman']:.6f}", flush=True
        )
    else:
        baseline_result = json.loads(baseline_path.read_text(encoding="utf-8"))
    rows = existing.to_dict("records") if not existing.empty else []
    if pending:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, workers), initializer=init_worker, initargs=(str(cache_dir),)
        ) as executor:
            futures = {
                executor.submit(evaluate_config, trial_id, config): trial_id
                for trial_id, config in pending
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                rows.append(result)
                current = pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)
                atomic_csv(current, trial_path)
                print(
                    f"Trial {int(result['trial_id']):03d}/{n_trials}: "
                    f"weighted val Spearman={result['weighted_val_spearman']:.6f}; "
                    f"runtime={result['runtime_seconds']:.1f}s",
                    flush=True,
                )
    trials = pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)
    if len(trials) != n_trials or set(trials["trial_id"].astype(int)) != set(range(1, n_trials + 1)):
        raise RuntimeError(f"Expected {n_trials} completed unique trials, found {len(trials)}")
    trials["rank"] = trials["weighted_val_spearman"].rank(method="min", ascending=False).astype(int)
    trials = trials.sort_values(["rank", "trial_id"]).reset_index(drop=True)
    atomic_csv(trials, trial_path)
    winner = trials.iloc[0]
    best_config = {key: winner[key] for key in BASELINE_CONFIG}
    best_config["hidden1"] = int(best_config["hidden1"])
    best_config["hidden2"] = int(best_config["hidden2"])
    best_config["epochs"] = int(best_config["epochs"])
    best_config["dropout"] = float(best_config["dropout"])
    best_config["learning_rate"] = float(best_config["learning_rate"])
    best_config["weight_decay"] = float(best_config["weight_decay"])
    selection = {
        "selection_rule": "highest pooled 5-fold weighted validation Spearman; ties use lower trial_id",
        "test15_used_for_hpo_selection": False,
        "n_random_trials": n_trials,
        "random_search_seed": 20260824,
        "model_seed_rule": "42 + cv_fold*100 + property_index",
        "best_trial_id": int(winner["trial_id"]),
        "best_config_id": str(winner["config_id"]),
        "best_params": best_config,
        "best_weighted_val_spearman": float(winner["weighted_val_spearman"]),
        "baseline_params": BASELINE_CONFIG,
        "baseline_weighted_val_spearman": float(baseline_result["weighted_val_spearman"]),
        "delta_val_spearman": float(
            winner["weighted_val_spearman"] - baseline_result["weighted_val_spearman"]
        ),
    }
    atomic_json(selection, HPO_DIR / "best_params.json")
    return trials, selection


def train_final_and_evaluate(
    config: dict[str, Any], selection: dict[str, Any], solvent_summary: dict[str, Any]
) -> dict[str, Any]:
    calibration, test_features, feature_columns, _ = load_prepared_data()
    models: dict[str, FittedLogMLP] = {}
    prediction_frames = []
    for prop in PROPERTIES:
        train = calibration[calibration["property"] == prop]
        test = test_features[test_features["property"] == prop]
        y = np.log(train["abs_error"].to_numpy(dtype=float) + float(EPSILON[prop]))
        model = FittedLogMLP(config).fit(
            train[feature_columns].to_numpy(dtype=float),
            y,
            train["sample_weight"].to_numpy(dtype=float),
            42 + PROPERTY_INDEX[prop],
        )
        models[prop] = model
        pred = inverse_log(model.predict(test[feature_columns].to_numpy(dtype=float)), prop)
        frame = test[["property", "row_index", "fold"]].copy()
        frame["predicted_ae"] = pred
        prediction_frames.append(frame)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    prediction_path = FINAL_DIR / "test_predictions_label_free.csv"
    atomic_csv(predictions, prediction_path)
    checkpoint = {
        "format_version": 1,
        "model_family": "mlp_log_hpo100",
        "loss": "sample-weighted L1 in property-specific log(AE + epsilon) space",
        "target_transform": "log(AE + epsilon_by_property); inverse=max(exp(output)-epsilon,0)",
        "epsilon_by_property": EPSILON,
        "feature_columns": feature_columns,
        "selected_params": config,
        "selection": selection,
        "models": {prop: model.state_payload() for prop, model in models.items()},
    }
    checkpoint_path = PACKAGE_ROOT / "models/reliability_model/mlp_log_hpo100_reliability_model_bundle.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    # Sealed labels are opened only after candidate selection, final fitting, and prediction freezing.
    labels = FEATURES.load_test_labels(
        str(PACKAGE_ROOT / "results/intermediate/test_labels_ae_pre.csv")
    )
    evaluated = test_features.merge(
        predictions, on=["property", "row_index", "fold"], how="left", validate="one_to_one"
    ).merge(labels, on=["property", "row_index"], how="left", validate="many_to_one")
    if evaluated[["predicted_ae", "y_true"]].isna().any().any():
        raise ValueError("Missing value after Test15 evaluation join")
    metric_rows = []
    for prop in PROPERTIES:
        subset = evaluated[evaluated["property"] == prop]
        metric_rows.append(
            FEATURES.deployment_metric_row(
                variant="mlp_log_hpo100",
                prop=prop,
                test_df=subset,
                predicted_ae=subset["predicted_ae"].to_numpy(dtype=float),
            )
        )
    metrics = FEATURES.add_weighted_row(pd.DataFrame(metric_rows), "mlp_log_hpo100")
    atomic_csv(metrics, FINAL_DIR / "test15_by_property.csv")
    atomic_csv(evaluated, FINAL_DIR / "test15_evaluated_rows.csv")

    baseline = pd.read_csv(BASELINE_METRICS)[["property", "n", "spearman", "ae_mae"]].rename(
        columns={"spearman": "baseline_spearman", "ae_mae": "baseline_ae_mae"}
    )
    comparison = baseline.merge(
        metrics[["property", "n", "spearman", "ae_mae"]].rename(
            columns={"spearman": "hpo_spearman", "ae_mae": "hpo_ae_mae"}
        ),
        on=["property", "n"],
        validate="one_to_one",
    )
    comparison["delta_spearman_hpo_minus_baseline"] = (
        comparison["hpo_spearman"] - comparison["baseline_spearman"]
    )
    comparison["delta_ae_mae_hpo_minus_baseline"] = (
        comparison["hpo_ae_mae"] - comparison["baseline_ae_mae"]
    )
    atomic_csv(comparison, FINAL_DIR / "vs_baseline_test15.csv")
    weighted = metrics[metrics["property"] == "weighted_all"].iloc[0]
    baseline_weighted = baseline[baseline["property"] == "weighted_all"].iloc[0]
    manifest = {
        "status": "passed",
        "n_random_trials": int(selection["n_random_trials"]),
        "cv_method": "5-fold canonical-solute-solvent GroupKFold within Calibration85",
        "selection_metric": "sample-count-weighted pooled out-of-fold Spearman across Abs, Emi, PLQY, and em",
        "test15_used_for_hpo_selection": False,
        "test_labels_loaded_after_prediction_freeze": True,
        "n_features": len(feature_columns),
        "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test_features)),
        "best_trial_id": int(selection["best_trial_id"]),
        "best_params": config,
        "baseline_weighted_cv_spearman": float(selection["baseline_weighted_val_spearman"]),
        "best_weighted_cv_spearman": float(selection["best_weighted_val_spearman"]),
        "weighted_test15_spearman": float(weighted["spearman"]),
        "baseline_weighted_test15_spearman": float(baseline_weighted["baseline_spearman"]),
        "delta_test15_spearman": float(weighted["spearman"] - baseline_weighted["baseline_spearman"]),
        "weighted_test15_ae_mae": float(weighted["ae_mae"]),
        "baseline_weighted_test15_ae_mae": float(baseline_weighted["baseline_ae_mae"]),
        "delta_test15_ae_mae": float(weighted["ae_mae"] - baseline_weighted["baseline_ae_mae"]),
        "checkpoint": str(checkpoint_path.relative_to(PACKAGE_ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "solvent_reference_summary": solvent_summary,
    }
    atomic_json(manifest, FINAL_DIR / "validation.json")
    return manifest


def main() -> None:
    global _SMOKE_MODE
    args = parse_args()
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive")
    if args.smoke_test:
        _SMOKE_MODE = True
        BASELINE_CONFIG["epochs"] = 3
        args.n_trials = min(args.n_trials, 2)
        args.workers = min(args.workers, 2)
    HPO_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    calibration, _, solvent_summary = prepare_base_frames()
    if args.check:
        required = [
            PACKAGE_ROOT / "results/intermediate/calibration_features_ae_pre.csv",
            PACKAGE_ROOT / "results/intermediate/test_features_ae_pre.csv",
            PACKAGE_ROOT / "results/intermediate/test_labels_ae_pre.csv",
            PACKAGE_ROOT / "data/splits/deployment/deployment.csv",
            BASELINE_METRICS,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing HPO inputs:\n" + "\n".join(missing))
        print(
            json.dumps(
                {
                    "status": "passed",
                    "package_root": str(PACKAGE_ROOT),
                    "calibration_rows": int(len(calibration)),
                    "n_features": len(FEATURE_COLUMNS),
                    "n_random_trials": int(args.n_trials),
                    "test15_used_for_hpo_selection": False,
                },
                indent=2,
            )
        )
        return
    prepare_cv_cache(calibration, reset=args.reset)
    trials, selection = run_search(args.n_trials, args.workers, args.reset)
    print(
        f"Selected trial {selection['best_trial_id']}: "
        f"weighted CV Spearman={selection['best_weighted_val_spearman']:.6f}",
        flush=True,
    )
    if not args.hpo_only and not args.smoke_test:
        manifest = train_final_and_evaluate(selection["best_params"], selection, solvent_summary)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    else:
        print(trials.head(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
