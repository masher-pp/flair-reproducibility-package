from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


EXPECTED_PROPERTIES = ("abs", "emi", "plqy", "em")


def _torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _activation(name: str) -> torch.nn.Module:
    if name == "relu":
        return torch.nn.ReLU()
    if name == "leaky_relu":
        return torch.nn.LeakyReLU(negative_slope=0.05)
    if name == "silu":
        return torch.nn.SiLU()
    if name == "gelu":
        return torch.nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _build_network(n_features: int, config: dict[str, Any]) -> torch.nn.Module:
    layers: list[torch.nn.Module] = [
        torch.nn.Linear(n_features, int(config["hidden1"])),
        _activation(str(config["activation"])),
    ]
    dropout = float(config["dropout"])
    if dropout > 0:
        layers.append(torch.nn.Dropout(dropout))
    layers.extend(
        [
            torch.nn.Linear(int(config["hidden1"]), int(config["hidden2"])),
            _activation(str(config["activation"])),
        ]
    )
    if dropout > 0:
        layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(int(config["hidden2"]), 1))
    return torch.nn.Sequential(*layers)


class LogMLPPropertyModel:
    def __init__(self, payload: dict[str, Any], n_features: int) -> None:
        self.config = dict(payload["config"])
        self.x_mean = np.asarray(payload["x_mean"], dtype=np.float32)
        self.x_scale = np.asarray(payload["x_scale"], dtype=np.float32)
        self.y_center = float(payload["y_center"])
        self.y_scale = float(payload["y_scale"])
        if self.x_mean.shape != (n_features,) or self.x_scale.shape != (n_features,):
            raise ValueError("Checkpoint scaler dimensions do not match feature_columns")
        self.network = _build_network(n_features, self.config)
        self.network.load_state_dict(payload["state_dict"], strict=True)
        self.network.eval()

    def predict_log_target(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.x_mean.size:
            raise ValueError(
                f"Expected a 2D array with {self.x_mean.size} features, got {values.shape}"
            )
        scaled = (values - self.x_mean) / self.x_scale
        with torch.no_grad():
            prediction = self.network(torch.from_numpy(scaled)).numpy().reshape(-1)
        return prediction * self.y_scale + self.y_center


def is_log_mlp_bundle(bundle: object) -> bool:
    return isinstance(bundle, dict) and str(bundle.get("model_family", "")) == "mlp_log_hpo100"


def load_log_mlp_bundle(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    bundle = _torch_load(checkpoint_path)
    if not is_log_mlp_bundle(bundle):
        raise ValueError(f"Not a mlp_log_hpo100 checkpoint: {checkpoint_path}")
    feature_columns = list(bundle.get("feature_columns", []))
    if len(feature_columns) != 32 or len(set(feature_columns)) != 32:
        raise ValueError("Log-MLP checkpoint must contain 32 unique feature columns")
    properties = bundle.get("models")
    if not isinstance(properties, dict) or set(properties) != set(EXPECTED_PROPERTIES):
        raise ValueError("Checkpoint must contain exactly abs/emi/plqy/em property models")
    epsilon = bundle.get("epsilon_by_property")
    if not isinstance(epsilon, dict) or set(epsilon) != set(EXPECTED_PROPERTIES):
        raise ValueError("Checkpoint epsilon_by_property is incomplete")
    bundle["_runtime_models"] = {
        prop: LogMLPPropertyModel(properties[prop], len(feature_columns))
        for prop in EXPECTED_PROPERTIES
    }
    bundle["checkpoint_path"] = str(checkpoint_path)
    bundle.setdefault("model_name", "mlp_log_hpo100")
    bundle.setdefault("base_prediction_type", "mean_of_5_base_models_before_ae_prediction")
    bundle.setdefault(
        "calibration_features_csv",
        "../../results/intermediate/calibration_features_ae_pre.csv",
    )
    return bundle


def predict_pre_ae(bundle: dict[str, Any], prop: str, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    if prop not in EXPECTED_PROPERTIES:
        raise KeyError(prop)
    runtime_models = bundle.get("_runtime_models")
    if not isinstance(runtime_models, dict):
        raise ValueError("Checkpoint has not been loaded with load_log_mlp_bundle")
    feature_columns = list(bundle["feature_columns"])
    if isinstance(features, pd.DataFrame):
        missing = sorted(set(feature_columns) - set(features.columns))
        if missing:
            raise ValueError(f"Missing Reliability features: {missing}")
        x = features.reindex(columns=feature_columns).to_numpy(dtype=float)
    else:
        x = np.asarray(features, dtype=float)
    log_target = runtime_models[prop].predict_log_target(x)
    epsilon = float(bundle["epsilon_by_property"][prop])
    prediction = np.maximum(np.exp(np.asarray(log_target, dtype=float)) - epsilon, 0.0)
    if not np.isfinite(prediction).all():
        raise ValueError(f"Non-finite Pre_AE prediction for property={prop}")
    return prediction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Log-MLP Reliability .pt checkpoint.")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    path = Path(args.checkpoint).expanduser().resolve()
    bundle = load_log_mlp_bundle(path)
    print(
        json.dumps(
            {
                "status": "passed",
                "checkpoint": str(path),
                "sha256": sha256(path),
                "model_family": bundle["model_family"],
                "properties": list(EXPECTED_PROPERTIES),
                "n_features": len(bundle["feature_columns"]),
                "selected_params": bundle["selected_params"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
