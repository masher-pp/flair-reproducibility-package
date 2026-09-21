from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RELIABILITY_DIR = PACKAGE_ROOT / "src/reliability_model"
OFFICIAL_CHECKPOINT = (
    PACKAGE_ROOT / "models/reliability_model/mlp_log_hpo100_reliability_model_bundle.pt"
)
BEST_PARAMS_PATH = PACKAGE_ROOT / "results/hpo/mlp_log_hpo100_20260824/best_params.json"
OFFICIAL_METRICS_PATH = (
    PACKAGE_ROOT / "results/final/mlp_log_hpo100_20260824/test15_by_property.csv"
)


def load_training_module():
    path = RELIABILITY_DIR / "06_hpo_log_mlp_random.py"
    spec = importlib.util.spec_from_file_location("flair_log_mlp_hpo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the selected v3 Log-MLP Reliability configuration with a new "
            "initialization seed, without rerunning HPO or overwriting the official model."
        )
    )
    parser.add_argument("--seed", type=int, required=True, help="Base model seed.")
    parser.add_argument(
        "--tag",
        default=None,
        help="Output tag; defaults to mlp_log_hpo100_seed<seed>_20260827.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    tag = args.tag or f"mlp_log_hpo100_seed{args.seed}_20260827"
    if Path(tag).name != tag:
        raise ValueError("--tag must be a single path component")

    training = load_training_module()
    selection = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    config = dict(selection["best_params"])
    calibration, test_features, feature_columns, solvent_summary = training.load_prepared_data()

    output_dir = PACKAGE_ROOT / "results/retrained" / tag
    checkpoint_path = (
        PACKAGE_ROOT / "models/reliability_model/retrained" / f"{tag}_bundle.pt"
    )
    property_seeds = {
        prop: int(args.seed + training.PROPERTY_INDEX[prop]) for prop in training.PROPERTIES
    }

    models: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for prop in training.PROPERTIES:
        train = calibration[calibration["property"] == prop]
        test = test_features[test_features["property"] == prop]
        y = np.log(
            train["abs_error"].to_numpy(dtype=float) + float(training.EPSILON[prop])
        )
        model = training.FittedLogMLP(config).fit(
            train[feature_columns].to_numpy(dtype=float),
            y,
            train["sample_weight"].to_numpy(dtype=float),
            property_seeds[prop],
        )
        models[prop] = model
        predicted_ae = training.inverse_log(
            model.predict(test[feature_columns].to_numpy(dtype=float)), prop
        )
        frame = test[["property", "row_index", "fold"]].copy()
        frame["predicted_ae"] = predicted_ae
        prediction_frames.append(frame)
        print(f"Trained {prop} with seed {property_seeds[prop]}", flush=True)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    prediction_path = output_dir / "test_predictions_label_free.csv"
    atomic_csv(predictions, prediction_path)

    checkpoint = {
        "format_version": 1,
        "model_family": "mlp_log_hpo100",
        "model_name": tag,
        "loss": "sample-weighted L1 in property-specific log(AE + epsilon) space",
        "target_transform": "log(AE + epsilon_by_property); inverse=max(exp(output)-epsilon,0)",
        "epsilon_by_property": training.EPSILON,
        "feature_columns": feature_columns,
        "selected_params": config,
        "selection": selection,
        "retraining": {
            "controlled_change": "model initialization/training seed only",
            "hpo_rerun": False,
            "base_seed": int(args.seed),
            "property_seeds": property_seeds,
            "official_checkpoint_sha256": sha256(OFFICIAL_CHECKPOINT),
        },
        "models": {prop: model.state_payload() for prop, model in models.items()},
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_checkpoint)
    os.replace(temporary_checkpoint, checkpoint_path)

    # Open the sealed labels only after fitting, checkpointing, and freezing predictions.
    labels = training.FEATURES.load_test_labels(
        str(PACKAGE_ROOT / "results/intermediate/test_labels_ae_pre.csv")
    )
    evaluated = test_features.merge(
        predictions, on=["property", "row_index", "fold"], how="left", validate="one_to_one"
    ).merge(labels, on=["property", "row_index"], how="left", validate="many_to_one")
    if evaluated[["predicted_ae", "y_true"]].isna().any().any():
        raise ValueError("Missing value after Test15 evaluation join")

    metric_rows = []
    for prop in training.PROPERTIES:
        subset = evaluated[evaluated["property"] == prop]
        metric_rows.append(
            training.FEATURES.deployment_metric_row(
                variant=tag,
                prop=prop,
                test_df=subset,
                predicted_ae=subset["predicted_ae"].to_numpy(dtype=float),
            )
        )
    metrics = training.FEATURES.add_weighted_row(pd.DataFrame(metric_rows), tag)
    atomic_csv(metrics, output_dir / "test15_by_property.csv")
    atomic_csv(evaluated, output_dir / "test15_evaluated_rows.csv")

    official_metrics = pd.read_csv(OFFICIAL_METRICS_PATH)[
        ["property", "n", "spearman", "ae_mae"]
    ].rename(
        columns={"spearman": "seed42_spearman", "ae_mae": "seed42_ae_mae"}
    )
    comparison = official_metrics.merge(
        metrics[["property", "n", "spearman", "ae_mae"]].rename(
            columns={"spearman": "new_seed_spearman", "ae_mae": "new_seed_ae_mae"}
        ),
        on=["property", "n"],
        validate="one_to_one",
    )
    comparison["delta_spearman_new_minus_seed42"] = (
        comparison["new_seed_spearman"] - comparison["seed42_spearman"]
    )
    comparison["delta_ae_mae_new_minus_seed42"] = (
        comparison["new_seed_ae_mae"] - comparison["seed42_ae_mae"]
    )
    atomic_csv(comparison, output_dir / "vs_seed42_test15.csv")

    weighted = metrics[metrics["property"] == "weighted_all"].iloc[0]
    official_weighted = official_metrics[official_metrics["property"] == "weighted_all"].iloc[0]
    manifest = {
        "status": "passed",
        "controlled_change": "model initialization/training seed only",
        "hpo_rerun": False,
        "base_seed": int(args.seed),
        "property_seeds": property_seeds,
        "best_trial_id": int(selection["best_trial_id"]),
        "best_params": config,
        "test15_used_for_selection": False,
        "test_labels_loaded_after_prediction_freeze": True,
        "n_features": len(feature_columns),
        "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test_features)),
        "weighted_test15_spearman": float(weighted["spearman"]),
        "seed42_weighted_test15_spearman": float(official_weighted["seed42_spearman"]),
        "delta_weighted_test15_spearman": float(
            weighted["spearman"] - official_weighted["seed42_spearman"]
        ),
        "weighted_test15_ae_mae": float(weighted["ae_mae"]),
        "seed42_weighted_test15_ae_mae": float(official_weighted["seed42_ae_mae"]),
        "delta_weighted_test15_ae_mae": float(
            weighted["ae_mae"] - official_weighted["seed42_ae_mae"]
        ),
        "checkpoint": str(checkpoint_path.relative_to(PACKAGE_ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "official_checkpoint": str(OFFICIAL_CHECKPOINT.relative_to(PACKAGE_ROOT)),
        "official_checkpoint_sha256": sha256(OFFICIAL_CHECKPOINT),
        "calibration_features_sha256": sha256(
            PACKAGE_ROOT / "results/intermediate/calibration_features_ae_pre.csv"
        ),
        "test_features_sha256": sha256(
            PACKAGE_ROOT / "results/intermediate/test_features_ae_pre.csv"
        ),
        "test_labels_sha256": sha256(
            PACKAGE_ROOT / "results/intermediate/test_labels_ae_pre.csv"
        ),
        "solvent_reference_summary": solvent_summary,
    }
    atomic_json(manifest, output_dir / "validation.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
