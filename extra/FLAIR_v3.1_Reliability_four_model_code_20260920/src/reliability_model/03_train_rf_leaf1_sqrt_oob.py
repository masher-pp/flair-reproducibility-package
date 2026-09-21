from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config
from data_utils import (
    SOLVENT_REFERENCE_FEATURE_NAMES,
    SOLVENT_SCAFFOLD_FEATURE_NAMES,
    build_solvent_tanimoto_reference,
    get_scaffold_smiles,
    mol_from_smiles,
    solvent_reference_features,
    standardize_columns,
)
from rank_ensemble_features import (
    FINAL_FEATURE_COLUMNS,
    PROPERTIES,
    add_calibration_scaffold_error_features,
)
from solvent_residual_features import (
    DEFAULT_SOLVENT_RESIDUAL_PRIOR_STRENGTH,
    SOLVENT_RESIDUAL_FEATURE_NAMES,
    add_solvent_residual_feature_columns,
)


EPSILON_BY_PROPERTY = {"abs": 1.0, "emi": 1.0, "em": 0.01, "plqy": 0.005}
BASELINE_MODEL_NAME = "rf_property_specific_hpo100_24f_ensemble_base"
SOLVENT_MODEL_NAME = "rf_property_specific_hpo100_32f_solvent_reference_ensemble_base"
MODEL_NAME = "rf_property_specific_hpo100_38f_solvent_scaffold_ensemble_base"
SOLVENT_RESIDUAL_MODEL_NAME = "rf_property_specific_hpo100_38f_solvent_residual_ensemble_base"
SOLVENT_BIAS_CORRECTED_MODEL_NAME = (
    "rf_property_specific_hpo100_38f_solvent_residual_bias_corrected_ensemble_base"
)
MODEL_PARAMS_BY_PROPERTY = {
    "abs": {
        "n_estimators": 100,
        "min_samples_leaf": 6,
        "min_samples_split": 16,
        "max_features": 0.10,
        "max_depth": 6,
        "max_leaf_nodes": 256,
        "max_samples": 0.70,
        "criterion": "squared_error",
    },
    "emi": {
        "n_estimators": 100,
        "min_samples_leaf": 1,
        "min_samples_split": 8,
        "max_features": 0.10,
        "max_depth": 14,
        "max_leaf_nodes": 128,
        "max_samples": 0.70,
        "criterion": "squared_error",
    },
    "plqy": {
        "n_estimators": 100,
        "min_samples_leaf": 6,
        "min_samples_split": 16,
        "max_features": 0.10,
        "max_depth": 6,
        "max_leaf_nodes": 256,
        "max_samples": 0.70,
        "criterion": "squared_error",
    },
    "em": {
        "n_estimators": 300,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "max_features": 0.10,
        "max_depth": 10,
        "max_leaf_nodes": 512,
        "max_samples": None,
        "criterion": "friedman_mse",
    },
}
DROPPED_FEATURE_COLUMNS = [
    "base_prediction_fold_range",
    "calibration_same_scaffold_solvent_count",
    "calibration_same_scaffold_solvent_mean_abs_error",
]
MODEL_FEATURE_COLUMNS = [
    column for column in FINAL_FEATURE_COLUMNS if column not in DROPPED_FEATURE_COLUMNS
]
SOLVENT_AUGMENTED_MODEL_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS + list(SOLVENT_REFERENCE_FEATURE_NAMES)
SOLVENT_SCAFFOLD_AUGMENTED_MODEL_FEATURE_COLUMNS = (
    SOLVENT_AUGMENTED_MODEL_FEATURE_COLUMNS + list(SOLVENT_SCAFFOLD_FEATURE_NAMES)
)
SOLVENT_RESIDUAL_AUGMENTED_MODEL_FEATURE_COLUMNS = (
    SOLVENT_AUGMENTED_MODEL_FEATURE_COLUMNS + list(SOLVENT_RESIDUAL_FEATURE_NAMES)
)
HISTORICAL_FEATURE_COLUMNS = [
    "calibration_same_scaffold_count",
    "calibration_same_scaffold_mean_abs_error",
    "calibration_same_scaffold_median_abs_error",
    "calibration_same_scaffold_max_abs_error",
    "calibration_same_scaffold_std_abs_error",
    "calibration_same_scaffold_solvent_count",
    "calibration_same_scaffold_solvent_mean_abs_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train property-specific RF log-AE models with solvent-reference features.")
    parser.add_argument("--calibration_csv", default="../../results/intermediate/calibration_features_ae_pre.csv")
    parser.add_argument("--test_features_csv", default="../../results/intermediate/test_features_ae_pre.csv")
    parser.add_argument("--test_labels_csv", default="../../results/intermediate/test_labels_ae_pre.csv")
    parser.add_argument(
        "--train_and_val_csv",
        default="../../data/splits/deployment/deployment.csv",
        help="Label-free reference used for solvent occurrence counts and Tanimoto neighborhoods.",
    )
    parser.add_argument(
        "--out_model",
        default=None,
        help="Output model path. Defaults to a mode-specific non-overlapping filename.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output result directory. Defaults to a mode-specific non-overlapping directory.",
    )
    parser.add_argument("--calibration_name", default="Calibration85")
    parser.add_argument("--holdout_name", default="Test15")
    parser.add_argument(
        "--baseline_24_features",
        action="store_true",
        help="Reproduce the published 24-feature model without fitting the new solvent-reference columns.",
    )
    parser.add_argument(
        "--with_solvent_scaffold_features",
        action="store_true",
        help="Fit the experimental 38-feature variant with six additional solvent scaffold columns.",
    )
    parser.add_argument(
        "--with_solvent_residual_features",
        action="store_true",
        help="Fit a 38-feature variant with leakage-safe historical solvent residual statistics.",
    )
    parser.add_argument(
        "--with_solvent_bias_correction",
        action="store_true",
        help=(
            "Fit the solvent-residual variant after subtracting the leakage-safe shrunken "
            "same-solvent signed bias from the main prediction."
        ),
    )
    parser.add_argument(
        "--solvent_residual_prior_strength",
        type=float,
        default=DEFAULT_SOLVENT_RESIDUAL_PRIOR_STRENGTH,
        help="Global pseudo-count used to shrink sparse same-solvent residual statistics.",
    )
    parser.add_argument(
        "--params_by_property_json",
        default=None,
        help=(
            "Optional HPO selection JSON containing selected_by_property.<property>.params. "
            "When omitted, use the bundled default parameters."
        ),
    )
    parser.add_argument("--validate_only", action="store_true", help="Validate the Step 02 table and exit without training.")
    return parser.parse_args()


def inverse_log_error(log_error, prop: str) -> np.ndarray:
    return np.maximum(np.exp(np.asarray(log_error, dtype=float)) - EPSILON_BY_PROPERTY[prop], 0.0)


def smooth_loss_weights(abs_error: np.ndarray, prop: str) -> np.ndarray:
    """Same smoothstep weighting used by Step 02, kept local to avoid PyTorch imports."""
    floor = float(config.ERROR_LOSS_FLOOR_BY_PROPERTY[prop])
    width = max(float(config.ERROR_LOSS_SMOOTH_WIDTH_BY_PROPERTY[prop]), 1e-12)
    scaled = np.clip((np.asarray(abs_error, dtype=float) - floor) / width, 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def spearman_manual(a, b) -> float:
    values = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return float("nan")
    a_rank = values["a"].rank(method="average").to_numpy(dtype=float)
    b_rank = values["b"].rank(method="average").to_numpy(dtype=float)
    a_rank = a_rank - a_rank.mean()
    b_rank = b_rank - b_rank.mean()
    denominator = math.sqrt(float((a_rank * a_rank).sum() * (b_rank * b_rank).sum()))
    return float((a_rank * b_rank).sum() / denominator) if denominator else float("nan")


def weighted(metric: pd.DataFrame, column: str, weight_column: str = "n") -> float:
    values = metric[[column, weight_column]].replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values[weight_column] > 0]
    if values.empty:
        return float("nan")
    return float((values[column] * values[weight_column]).sum() / values[weight_column].sum())


def scaffold_split_spearman(pred_score, true_ae, same_scaffold_count) -> dict[str, float | int]:
    count = np.asarray(same_scaffold_count, dtype=float)
    zero_mask = count == 0
    nonzero_mask = ~zero_mask
    pred_score = np.asarray(pred_score, dtype=float)
    true_ae = np.asarray(true_ae, dtype=float)
    return {
        "n_same_scaffold_count_eq_0": int(zero_mask.sum()),
        "spearman_same_scaffold_count_eq_0": spearman_manual(pred_score[zero_mask], true_ae[zero_mask]),
        "n_same_scaffold_count_ne_0": int(nonzero_mask.sum()),
        "spearman_same_scaffold_count_ne_0": spearman_manual(pred_score[nonzero_mask], true_ae[nonzero_mask]),
    }


def solvent_split_spearman(pred_score, true_ae, solvent_count) -> dict[str, float | int]:
    count = np.asarray(solvent_count, dtype=float)
    unseen_mask = count == 0
    seen_mask = ~unseen_mask
    pred_score = np.asarray(pred_score, dtype=float)
    true_ae = np.asarray(true_ae, dtype=float)
    return {
        "n_solvent_count_eq_0": int(unseen_mask.sum()),
        "spearman_solvent_count_eq_0": spearman_manual(pred_score[unseen_mask], true_ae[unseen_mask]),
        "n_solvent_count_ne_0": int(seen_mask.sum()),
        "spearman_solvent_count_ne_0": spearman_manual(pred_score[seen_mask], true_ae[seen_mask]),
    }


def deployment_evaluation_frame(test_df: pd.DataFrame, predicted_ae) -> pd.DataFrame:
    work = test_df[
        [
            "row_index",
            "fold",
            "y_true",
            "base_prediction",
            "calibration_same_scaffold_count",
            "solvent_train_and_val_count",
        ]
    ].copy()
    work["predicted_ae"] = np.asarray(predicted_ae, dtype=float)
    grouped = work.groupby("row_index", sort=False)
    if (grouped["y_true"].nunique(dropna=False) != 1).any():
        raise ValueError("Test y_true is inconsistent across fold rows for the same row_index/property.")
    if (grouped["calibration_same_scaffold_count"].nunique(dropna=False) != 1).any():
        raise ValueError("same_scaffold_count is inconsistent across fold rows for the same row_index/property.")
    if (grouped["solvent_train_and_val_count"].nunique(dropna=False) != 1).any():
        raise ValueError("solvent_train_and_val_count is inconsistent across fold rows for the same sample/property.")
    sample = grouped.agg(
        y_true=("y_true", "first"),
        ensemble_prediction=("base_prediction", "mean"),
        predicted_ae=("predicted_ae", "mean"),
        same_scaffold_count=("calibration_same_scaffold_count", "first"),
        solvent_count=("solvent_train_and_val_count", "first"),
        n_fold_rows=("fold", "size"),
    )
    sample["true_ae"] = np.abs(sample["ensemble_prediction"] - sample["y_true"])
    return sample.reset_index()


def deployment_metric_row(
    *,
    variant: str,
    prop: str,
    test_df: pd.DataFrame,
    predicted_ae,
) -> dict[str, float | int | str]:
    predicted_ae = np.asarray(predicted_ae, dtype=float)
    fold_true_ae = np.abs(
        test_df["base_prediction"].to_numpy(dtype=float) - test_df["y_true"].to_numpy(dtype=float)
    )
    sample = deployment_evaluation_frame(test_df, predicted_ae)
    sample_predicted_ae = sample["predicted_ae"].to_numpy(dtype=float)
    sample_true_ae = sample["true_ae"].to_numpy(dtype=float)
    return {
        "variant": variant,
        "property": prop,
        "n": int(len(sample)),
        "n_fold_rows": int(len(test_df)),
        "spearman": spearman_manual(sample_predicted_ae, sample_true_ae),
        "spearman_fold_rows": spearman_manual(predicted_ae, fold_true_ae),
        "ae_mae": float(np.mean(np.abs(sample_predicted_ae - sample_true_ae))),
        "ae_mae_fold_rows": float(np.mean(np.abs(predicted_ae - fold_true_ae))),
        "pred_ae_mean": float(np.mean(sample_predicted_ae)),
        "true_ae_mean": float(np.mean(sample_true_ae)),
        **scaffold_split_spearman(
            sample_predicted_ae,
            sample_true_ae,
            sample["same_scaffold_count"].to_numpy(dtype=float),
        ),
        **solvent_split_spearman(
            sample_predicted_ae,
            sample_true_ae,
            sample["solvent_count"].to_numpy(dtype=float),
        ),
    }


def make_model(prop: str, params_by_property: dict[str, dict] | None = None):
    selected = MODEL_PARAMS_BY_PROPERTY if params_by_property is None else params_by_property
    params = dict(selected[prop])
    return RandomForestRegressor(
        **params,
        random_state=42,
        n_jobs=1,
        bootstrap=True,
        oob_score=True,
    )


def add_weighted_row(metric: pd.DataFrame, variant: str) -> pd.DataFrame:
    weighted_row = {
        "variant": variant,
        "property": "weighted_all",
        "n": int(metric["n"].sum()),
        "n_fold_rows": int(metric["n_fold_rows"].sum()),
        "spearman": weighted(metric, "spearman"),
        "spearman_fold_rows": weighted(metric, "spearman_fold_rows", "n_fold_rows"),
        "ae_mae": weighted(metric, "ae_mae"),
        "ae_mae_fold_rows": weighted(metric, "ae_mae_fold_rows", "n_fold_rows"),
        "pred_ae_mean": weighted(metric, "pred_ae_mean"),
        "true_ae_mean": weighted(metric, "true_ae_mean"),
        "n_same_scaffold_count_eq_0": int(metric["n_same_scaffold_count_eq_0"].sum()),
        "spearman_same_scaffold_count_eq_0": weighted(
            metric,
            "spearman_same_scaffold_count_eq_0",
            "n_same_scaffold_count_eq_0",
        ),
        "n_same_scaffold_count_ne_0": int(metric["n_same_scaffold_count_ne_0"].sum()),
        "spearman_same_scaffold_count_ne_0": weighted(
            metric,
            "spearman_same_scaffold_count_ne_0",
            "n_same_scaffold_count_ne_0",
        ),
        "n_solvent_count_eq_0": int(metric["n_solvent_count_eq_0"].sum()),
        "spearman_solvent_count_eq_0": weighted(
            metric,
            "spearman_solvent_count_eq_0",
            "n_solvent_count_eq_0",
        ),
        "n_solvent_count_ne_0": int(metric["n_solvent_count_ne_0"].sum()),
        "spearman_solvent_count_ne_0": weighted(
            metric,
            "spearman_solvent_count_ne_0",
            "n_solvent_count_ne_0",
        ),
    }
    return pd.concat([metric, pd.DataFrame([weighted_row])], ignore_index=True)


def validate_feature_rows(
    df: pd.DataFrame,
    *,
    expected_split: str,
    feature_columns: list[str] | None = None,
) -> list[str]:
    feature_columns = list(MODEL_FEATURE_COLUMNS if feature_columns is None else feature_columns)
    required_columns = set(FINAL_FEATURE_COLUMNS) | {"split", "property", "row_index", "fold"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Step 02 {expected_split} features are missing columns: {missing_columns}")
    if df.empty:
        raise ValueError(f"Step 02 {expected_split} features are empty.")
    split_values = set(df["split"].astype(str))
    if split_values != {expected_split}:
        raise ValueError(f"Expected only split={expected_split}, found: {sorted(split_values)}")
    property_values = set(df["property"].astype(str))
    if property_values != set(PROPERTIES):
        raise ValueError(f"Expected properties {PROPERTIES}, found: {sorted(property_values)}")
    numeric_columns = list(FINAL_FEATURE_COLUMNS) + ["row_index", "fold"]
    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        bad_columns = [
            column
            for column in numeric_columns
            if not np.isfinite(numeric[column].to_numpy(dtype=float)).all()
        ]
        raise ValueError(f"Step 02 {expected_split} features contain non-finite values: {bad_columns}")
    df[numeric_columns] = numeric
    if df.duplicated(["split", "property", "row_index", "fold"]).any():
        raise ValueError(f"Duplicate keys found in Step 02 {expected_split} features.")
    expected_folds = set(range(1, 6))
    if set(df["fold"].astype(int)) != expected_folds:
        raise ValueError(f"Expected folds 1-5, found: {sorted(set(df['fold'].astype(int)))}")
    group_sizes = df.groupby(["split", "property", "row_index"]).size()
    if not (group_sizes == 5).all():
        raise ValueError(f"Every split/property/row_index must have 5 fold rows; found {group_sizes.value_counts().to_dict()}")
    group_keys = ["split", "property", "row_index"]
    expected_fold_std = df.groupby(group_keys)["base_prediction"].transform(lambda values: values.std(ddof=0))
    expected_fold_range = df.groupby(group_keys)["base_prediction"].transform("max") - df.groupby(group_keys)[
        "base_prediction"
    ].transform("min")
    if not np.allclose(df["base_prediction_fold_std"], expected_fold_std, rtol=1e-10, atol=1e-12):
        raise ValueError("base_prediction_fold_std must use ddof=0 and match the 5 fold predictions.")
    if not np.allclose(df["base_prediction_fold_range"], expected_fold_range, rtol=1e-10, atol=1e-12):
        raise ValueError("base_prediction_fold_range does not match the 5 fold predictions.")
    if (df.groupby(group_keys)["calibration_same_scaffold_count"].nunique(dropna=False) != 1).any():
        raise ValueError("calibration_same_scaffold_count must be identical across fold rows for each sample/property.")
    return feature_columns


def load_calibration_features(
    path: str,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    feature_columns = validate_feature_rows(
        df,
        expected_split="calibration",
        feature_columns=feature_columns,
    )
    target_columns = ["y_true", "abs_error", "log_abs_error", "sample_weight"]
    missing = sorted(set(target_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Calibration features are missing training targets: {missing}")
    numeric = df[target_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Calibration targets contain non-finite values.")
    df[target_columns] = numeric
    if (df["abs_error"] < 0).any() or (df["sample_weight"] < 0).any():
        raise ValueError("Calibration abs_error/sample_weight must be nonnegative.")
    if (df.groupby(["property", "row_index"])["y_true"].nunique(dropna=False) != 1).any():
        raise ValueError("Calibration y_true is inconsistent across fold rows.")

    expected_abs_error = np.abs(df["base_prediction"].to_numpy(dtype=float) - df["y_true"].to_numpy(dtype=float))
    if not np.allclose(df["abs_error"].to_numpy(dtype=float), expected_abs_error, rtol=1e-10, atol=1e-12):
        raise ValueError("abs_error does not equal abs(y_pred - y_true).")
    for prop in PROPERTIES:
        prop_mask = df["property"] == prop
        expected_log_error = np.log(df.loc[prop_mask, "abs_error"].to_numpy(dtype=float) + EPSILON_BY_PROPERTY[prop])
        if not np.allclose(
            df.loc[prop_mask, "log_abs_error"].to_numpy(dtype=float),
            expected_log_error,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(f"log_abs_error is inconsistent with abs_error for property={prop}.")

    print(
        f"[RF] using selected {len(feature_columns)}-feature scheme; dropped={DROPPED_FEATURE_COLUMNS}",
        flush=True,
    )
    return df, feature_columns


def load_test_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_feature_rows(df, expected_split="test")
    forbidden = {"y_true", "y_pred", "ae", "true_ae", "abs_error", "log_abs_error", "sample_weight", "epsilon"}
    present = sorted(forbidden & set(df.columns))
    if present:
        raise ValueError(f"Sealed Test feature file contains label columns: {present}")
    return df


def load_test_labels(path: str) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"row_index", "property", "y_true"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Test label file is missing columns: {missing}")
    if set(labels["property"].astype(str)) != set(PROPERTIES):
        raise ValueError("Test label file does not contain exactly the expected properties.")
    labels[["row_index", "y_true"]] = labels[["row_index", "y_true"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(labels[["row_index", "y_true"]].to_numpy(dtype=float)).all():
        raise ValueError("Test labels contain non-finite values.")
    if labels.duplicated(["property", "row_index"]).any():
        raise ValueError("Test label file contains duplicate property/row_index keys.")
    return labels


def aggregate_base_predictions(df: pd.DataFrame, *, include_targets: bool) -> pd.DataFrame:
    """Collapse five base-model rows to one ensemble row per sample/property."""
    keys = ["property", "row_index"]
    group_sizes = df.groupby(keys, sort=False).size()
    if not (group_sizes == 5).all():
        raise ValueError(
            "Expected exactly five base-model rows per sample/property; "
            f"found {group_sizes.value_counts().to_dict()}"
        )

    rebuilt_columns = {
        "base_prediction",
        "base_prediction_fold_std",
        "base_prediction_fold_range",
        *HISTORICAL_FEATURE_COLUMNS,
    }
    constant_features = [column for column in FINAL_FEATURE_COLUMNS if column not in rebuilt_columns]
    constant_columns = ["split", "smiles", "solvent", *constant_features]
    if include_targets:
        constant_columns.append("y_true")
    grouped = df.groupby(keys, as_index=False, sort=False)
    for column in constant_columns:
        if (df.groupby(keys, sort=False)[column].nunique(dropna=False) != 1).any():
            raise ValueError(f"Column {column} is not constant across the five base-model rows.")

    sample = grouped[constant_columns].first()
    prediction_stats = grouped["base_prediction"].agg(
        base_prediction="mean",
        base_prediction_fold_std=lambda values: float(values.std(ddof=0)),
        base_prediction_fold_range=lambda values: float(values.max() - values.min()),
    )
    sample = sample.merge(prediction_stats, on=keys, how="left", validate="one_to_one")
    sample["fold"] = 0
    if include_targets:
        sample["abs_error"] = np.abs(sample["base_prediction"] - sample["y_true"])
        sample["log_abs_error"] = np.nan
        sample["sample_weight"] = np.nan
        for prop in PROPERTIES:
            mask = sample["property"] == prop
            errors = sample.loc[mask, "abs_error"].to_numpy(dtype=float)
            sample.loc[mask, "log_abs_error"] = np.log(errors + EPSILON_BY_PROPERTY[prop])
            sample.loc[mask, "sample_weight"] = smooth_loss_weights(errors, prop)
    else:
        sample["abs_error"] = np.nan
    return sample


def rebuild_ensemble_history_features(
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the unchanged scaffold-history feature engineering to ensemble AE."""
    combined = pd.concat([calibration_df, test_df], ignore_index=True, sort=False)
    scaffold_cache: dict[str, str] = {}

    def scaffold_for_smiles(smiles: object) -> str:
        key = "" if pd.isna(smiles) else str(smiles).strip()
        if key not in scaffold_cache:
            scaffold_cache[key] = get_scaffold_smiles(mol_from_smiles(key)) if key else ""
        return scaffold_cache[key]

    combined["_direct_scaffold"] = combined["smiles"].map(scaffold_for_smiles)
    combined = add_calibration_scaffold_error_features(combined).drop(columns=["_direct_scaffold"])
    calibration = combined[combined["split"] == "calibration"].copy()
    test = combined[combined["split"] == "test"].copy()
    test = test.drop(
        columns=["y_true", "abs_error", "log_abs_error", "sample_weight"],
        errors="ignore",
    )
    return calibration, test


def add_solvent_reference_feature_columns(
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_and_val_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit solvent occurrence/similarity reference on TrainAndVal structures only."""
    reference_df = standardize_columns(pd.read_csv(train_and_val_csv))
    if "solvent" not in reference_df.columns:
        raise ValueError(f"TrainAndVal reference is missing solvent: {train_and_val_csv}")
    reference = build_solvent_tanimoto_reference(reference_df["solvent"].tolist())

    combined = pd.concat([calibration_df, test_df], ignore_index=True, sort=False)
    feature_cache: dict[str, dict[str, float]] = {}
    rows = []
    for solvent in combined["solvent"].tolist():
        key = "" if pd.isna(solvent) else str(solvent).strip()
        if key not in feature_cache:
            feature_cache[key] = solvent_reference_features(key, reference)
        rows.append(feature_cache[key])
    solvent_feature_columns = [*SOLVENT_REFERENCE_FEATURE_NAMES, *SOLVENT_SCAFFOLD_FEATURE_NAMES]
    solvent_features = pd.DataFrame(rows, columns=solvent_feature_columns, index=combined.index)
    combined[solvent_feature_columns] = solvent_features

    split_at = len(calibration_df)
    calibration = combined.iloc[:split_at].copy()
    test = combined.iloc[split_at:].copy()
    test = test.drop(
        columns=["y_true", "abs_error", "log_abs_error", "sample_weight"],
        errors="ignore",
    )
    summary = {
        "reference_rows_with_valid_solvent": int(reference["n_rows"]),
        "reference_unique_solvents": int(reference["n_unique"]),
        "fingerprint_radius": int(reference["radius"]),
        "fingerprint_n_bits": int(reference["n_bits"]),
        "calibration_unseen_solvent_rows": int(
            (calibration["solvent_train_and_val_count"] == 0).sum()
        ),
        "test_unseen_solvent_rows": int((test["solvent_train_and_val_count"] == 0).sum()),
    }
    return calibration, test, summary


def main() -> None:
    args = parse_args()
    os.chdir(Path(__file__).resolve().parent)
    params_by_property = {prop: dict(params) for prop, params in MODEL_PARAMS_BY_PROPERTY.items()}
    if args.params_by_property_json:
        selection_path = Path(args.params_by_property_json).expanduser().resolve()
        selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_by_property = selection_payload.get("selected_by_property")
        if not isinstance(selected_by_property, dict):
            raise ValueError("HPO selection JSON is missing selected_by_property.")
        params_by_property = {
            prop: dict(selected_by_property[prop]["params"])
            for prop in PROPERTIES
        }
        print(f"[RF] using property-specific HPO parameters: {selection_path}", flush=True)
    selected_experimental_modes = sum(
        bool(value)
        for value in [
            args.with_solvent_scaffold_features,
            args.with_solvent_residual_features,
            args.with_solvent_bias_correction,
        ]
    )
    if args.baseline_24_features and selected_experimental_modes:
        raise ValueError("--baseline_24_features cannot be combined with an augmented mode.")
    if selected_experimental_modes > 1:
        raise ValueError("Select only one solvent scaffold/residual/bias-correction mode.")
    if args.baseline_24_features:
        feature_columns = list(MODEL_FEATURE_COLUMNS)
        model_name = BASELINE_MODEL_NAME
    elif args.with_solvent_scaffold_features:
        feature_columns = list(SOLVENT_SCAFFOLD_AUGMENTED_MODEL_FEATURE_COLUMNS)
        model_name = MODEL_NAME
    elif args.with_solvent_bias_correction:
        feature_columns = list(SOLVENT_RESIDUAL_AUGMENTED_MODEL_FEATURE_COLUMNS)
        model_name = SOLVENT_BIAS_CORRECTED_MODEL_NAME
    elif args.with_solvent_residual_features:
        feature_columns = list(SOLVENT_RESIDUAL_AUGMENTED_MODEL_FEATURE_COLUMNS)
        model_name = SOLVENT_RESIDUAL_MODEL_NAME
    else:
        feature_columns = list(SOLVENT_AUGMENTED_MODEL_FEATURE_COLUMNS)
        model_name = SOLVENT_MODEL_NAME
    if args.out_model is None:
        args.out_model = {
            BASELINE_MODEL_NAME: "../../models/reliability_model/reliability_rf_property_specific_hpo100_24features.pkl",
            SOLVENT_MODEL_NAME: "../../models/reliability_model/reliability_rf_property_specific_hpo100_32features_solvent.pkl",
            MODEL_NAME: "../../models/reliability_model/reliability_rf_property_specific_hpo100_38features_solvent_scaffold.pkl",
            SOLVENT_RESIDUAL_MODEL_NAME: "../../models/reliability_model/reliability_rf_property_specific_hpo100_38features_solvent_residual.pkl",
            SOLVENT_BIAS_CORRECTED_MODEL_NAME: "../../models/reliability_model/reliability_rf_property_specific_hpo100_38features_solvent_residual_bias_corrected.pkl",
        }[model_name]
    if args.out_dir is None:
        args.out_dir = {
            BASELINE_MODEL_NAME: "../../results/final/reliability_model",
            SOLVENT_MODEL_NAME: "../../results/final/reliability_model_solvent_features",
            MODEL_NAME: "../../results/final/reliability_model_solvent_scaffold_features",
            SOLVENT_RESIDUAL_MODEL_NAME: "../../results/final/reliability_model_solvent_residual_features",
            SOLVENT_BIAS_CORRECTED_MODEL_NAME: "../../results/final/reliability_model_solvent_residual_bias_corrected",
        }[model_name]
    calibration_df, feature_columns = load_calibration_features(
        args.calibration_csv,
        feature_columns=feature_columns,
    )
    test_features_df = load_test_features(args.test_features_csv)
    print(
        f"[RF] {args.holdout_name} labels remain sealed until all models have been fitted and predictions generated.",
        flush=True,
    )
    calibration_df = aggregate_base_predictions(calibration_df, include_targets=True)
    test_features_df = aggregate_base_predictions(test_features_df, include_targets=False)
    uses_solvent_residual = bool(
        args.with_solvent_residual_features or args.with_solvent_bias_correction
    )
    solvent_residual_summary = None
    if uses_solvent_residual:
        calibration_df, test_features_df, solvent_residual_summary = add_solvent_residual_feature_columns(
            calibration_df,
            test_features_df,
            prior_strength=args.solvent_residual_prior_strength,
            apply_bias_correction=args.with_solvent_bias_correction,
        )
        calibration_df["abs_error"] = np.abs(
            calibration_df["base_prediction"] - calibration_df["y_true"]
        )
        for prop in PROPERTIES:
            mask = calibration_df["property"] == prop
            errors = calibration_df.loc[mask, "abs_error"].to_numpy(dtype=float)
            calibration_df.loc[mask, "log_abs_error"] = np.log(
                errors + EPSILON_BY_PROPERTY[prop]
            )
            calibration_df.loc[mask, "sample_weight"] = smooth_loss_weights(errors, prop)
        print(
            "[RF] solvent residual history: "
            f"features={len(SOLVENT_RESIDUAL_FEATURE_NAMES)} "
            f"prior_strength={args.solvent_residual_prior_strength:g} "
            f"bias_correction={bool(args.with_solvent_bias_correction)}",
            flush=True,
        )
    calibration_df, test_features_df = rebuild_ensemble_history_features(
        calibration_df,
        test_features_df,
    )
    calibration_df, test_features_df, solvent_reference_summary = add_solvent_reference_feature_columns(
        calibration_df,
        test_features_df,
        args.train_and_val_csv,
    )
    print(f"[RF] solvent reference: {solvent_reference_summary}", flush=True)
    if not np.isfinite(
        calibration_df[feature_columns + ["log_abs_error", "sample_weight"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Ensemble Calibration data contain non-finite model values.")
    if not np.isfinite(test_features_df[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Ensemble Test data contain non-finite model features.")
    print(
        "[RF] averaged 5 base predictions before AE training: "
        f"calibration_samples={len(calibration_df)} sealed_test_samples={len(test_features_df)}",
        flush=True,
    )
    if args.validate_only:
        load_test_labels(args.test_labels_csv)
        print("[RF] validation passed; training not started.", flush=True)
        return

    models = {}
    prediction_frames = []
    oob_diagnostics = []
    for prop in PROPERTIES:
        train_df = calibration_df[calibration_df["property"] == prop].copy()
        test_df = test_features_df[test_features_df["property"] == prop].copy()
        print(f"[RF] {prop}: calibration_rows={len(train_df)} sealed_test_rows={len(test_df)}", flush=True)
        model = make_model(prop, params_by_property)
        model.fit(
            train_df[feature_columns].to_numpy(dtype=float),
            train_df["log_abs_error"].to_numpy(dtype=float),
            sample_weight=train_df["sample_weight"].to_numpy(dtype=float),
        )
        models[prop] = model

        oob_pred_log = np.asarray(model.oob_prediction_, dtype=float)
        if not np.isfinite(oob_pred_log).all():
            raise ValueError(f"Non-finite OOB predictions for property={prop}.")
        oob_pred_ae = inverse_log_error(oob_pred_log, prop)
        oob_metric = deployment_metric_row(
            variant=model_name,
            prop=prop,
            test_df=train_df,
            predicted_ae=oob_pred_ae,
        )
        oob_diagnostics.append(
            {
                "property": prop,
                "n_sample_property": int(oob_metric["n"]),
                "oob_r2_log_target": float(model.oob_score_),
                "oob_spearman_sample_diagnostic": float(oob_metric["spearman"]),
                "note": "Diagnostic only; each OOB row is one ensemble sample/property.",
            }
        )

        pred_log = model.predict(test_df[feature_columns].to_numpy(dtype=float))
        prediction = test_df[["property", "row_index", "fold"]].copy()
        prediction["predicted_log_ae"] = pred_log
        prediction["predicted_ae"] = inverse_log_error(pred_log, prop)
        prediction_frames.append(prediction)

    test_predictions = pd.concat(prediction_frames, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "test_predictions.csv"
    oob_path = out_dir / "oob_diagnostics.csv"
    predictions_tmp = predictions_path.with_suffix(".csv.tmp")
    oob_tmp = oob_path.with_suffix(".csv.tmp")
    test_predictions.to_csv(predictions_tmp, index=False)
    pd.DataFrame(oob_diagnostics).to_csv(oob_tmp, index=False)
    os.replace(predictions_tmp, predictions_path)
    os.replace(oob_tmp, oob_path)

    
    model_path = Path(args.out_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": (
            "ae_pre_rf_property_specific_hpo100_24f_ensemble_base_sealed_test_v4"
            if args.baseline_24_features
            else (
                "ae_pre_rf_property_specific_hpo100_38f_solvent_residual_bias_corrected_ensemble_base_sealed_test_v8"
                if args.with_solvent_bias_correction
                else (
                    "ae_pre_rf_property_specific_hpo100_38f_solvent_residual_ensemble_base_sealed_test_v7"
                    if args.with_solvent_residual_features
                    else (
                        "ae_pre_rf_property_specific_hpo100_32f_solvent_reference_ensemble_base_sealed_test_v5"
                        if not args.with_solvent_scaffold_features
                        else "ae_pre_rf_property_specific_hpo100_38f_solvent_scaffold_ensemble_base_sealed_test_v6"
                    )
                )
            )
        ),
        "model_name": model_name,
        "ensemble_type": "property_specific_random_forests",
        "base_prediction_type": "mean_of_5_base_models_before_ae_prediction",
        "optimization_target": "highest 5-fold grouped-CV Spearman independently per property over 100 HPO configs",
        "holdout_policy": (
            f"Model fitted on {args.calibration_name}; {args.holdout_name} labels are loaded after model freezing"
        ),
        "calibration_features_csv": str(Path(args.calibration_csv).resolve()),
        "test_features_csv": str(Path(args.test_features_csv).resolve()),
        "train_and_val_csv": str(Path(args.train_and_val_csv).resolve()),
        "feature_columns": feature_columns,
        "dropped_feature_columns": DROPPED_FEATURE_COLUMNS,
        "solvent_reference_feature_columns": list(SOLVENT_REFERENCE_FEATURE_NAMES),
        "solvent_scaffold_feature_columns": list(SOLVENT_SCAFFOLD_FEATURE_NAMES),
        "solvent_residual_feature_columns": list(SOLVENT_RESIDUAL_FEATURE_NAMES),
        "uses_solvent_residual_features": uses_solvent_residual,
        "solvent_bias_correction": bool(args.with_solvent_bias_correction),
        "solvent_residual_prior_strength": float(args.solvent_residual_prior_strength),
        "solvent_residual_summary": solvent_residual_summary,
        "solvent_reference_summary": solvent_reference_summary,
        "members": [model_name],
        "models": {
            model_name: {
                "models": models,
                "kind": "property_specific_rf",
                "params_by_property": params_by_property,
            }
        },
        "properties": PROPERTIES,
        "epsilon_by_property": EPSILON_BY_PROPERTY,
        "test_predictions_csv": str(predictions_path.resolve()),
        "oob_diagnostics_csv": str(oob_path.resolve()),
    }
    model_tmp = model_path.with_suffix(model_path.suffix + ".tmp")
    with open(model_tmp, "wb") as handle:
        pickle.dump(bundle, handle)
    os.replace(model_tmp, model_path)
    print(
        "[RF] fitted model and label-free Test predictions frozen; loading Test labels for final metrics",
        flush=True,
    )

    test_labels = load_test_labels(args.test_labels_csv)
    evaluated_test = test_features_df.merge(
        test_predictions,
        on=["property", "row_index", "fold"],
        how="left",
        validate="one_to_one",
    ).merge(
        test_labels,
        on=["property", "row_index"],
        how="left",
        validate="many_to_one",
    )
    if evaluated_test[["predicted_ae", "y_true"]].isna().any().any():
        raise ValueError("Test prediction/label join produced missing values.")

    metric_rows = []
    for prop in PROPERTIES:
        prop_test = evaluated_test[evaluated_test["property"] == prop].copy()
        metric_rows.append(
            deployment_metric_row(
                variant=model_name,
                prop=prop,
                test_df=prop_test,
                predicted_ae=prop_test["predicted_ae"].to_numpy(dtype=float),
            )
        )
    metrics = add_weighted_row(pd.DataFrame(metric_rows), model_name)
    correction_metrics = []
    if uses_solvent_residual:
        for prop in PROPERTIES:
            prop_test = evaluated_test[evaluated_test["property"] == prop].copy()
            raw_error = (
                prop_test["raw_base_prediction"].to_numpy(dtype=float)
                - prop_test["y_true"].to_numpy(dtype=float)
            )
            final_error = (
                prop_test["base_prediction"].to_numpy(dtype=float)
                - prop_test["y_true"].to_numpy(dtype=float)
            )
            correction_metrics.append(
                {
                    "property": prop,
                    "n": int(len(prop_test)),
                    "raw_bias": float(np.mean(raw_error)),
                    "final_bias": float(np.mean(final_error)),
                    "raw_mae": float(np.mean(np.abs(raw_error))),
                    "final_mae": float(np.mean(np.abs(final_error))),
                }
            )
    test_summary = metrics[metrics["property"] == "weighted_all"].iloc[0]
    metrics_path = out_dir / "test_metrics.csv"
    short_metrics_path = out_dir / "test_task_summary.csv"
    metrics_tmp = metrics_path.with_suffix(".csv.tmp")
    short_metrics_tmp = short_metrics_path.with_suffix(".csv.tmp")
    metrics.to_csv(metrics_tmp, index=False)
    short_metrics = (
        metrics[metrics["property"].isin(PROPERTIES)][["property", "spearman", "ae_mae"]]
        .rename(columns={"property": "task", "ae_mae": "mae"})
        .reset_index(drop=True)
    )
    short_metrics.to_csv(short_metrics_tmp, index=False)
    os.replace(metrics_tmp, metrics_path)
    os.replace(short_metrics_tmp, short_metrics_path)
    correction_metrics_path = None
    if correction_metrics:
        correction_metrics_path = out_dir / "main_prediction_correction_metrics.csv"
        correction_tmp = correction_metrics_path.with_suffix(".csv.tmp")
        pd.DataFrame(correction_metrics).to_csv(correction_tmp, index=False)
        os.replace(correction_tmp, correction_metrics_path)

    report = {
        "model": model_name,
        "model_params_by_property": params_by_property,
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "dropped_feature_columns": DROPPED_FEATURE_COLUMNS,
        "metric_unit": "one original sample/property; one AE prediction from the mean of 5 base models",
        "true_ae_definition": "abs(mean_5fold_base_prediction - y_true)",
        "holdout_policy": (
            f"Fixed {len(feature_columns)}-feature "
            f"{'baseline' if args.baseline_24_features else 'solvent-augmented'} scheme fitted on "
            f"{args.calibration_name}; "
            f"final report contains {args.holdout_name} metrics only"
        ),
        "weighted_spearman": float(test_summary["spearman"]),
        "weighted_spearman_fold_rows_diagnostic": float(test_summary["spearman_fold_rows"]),
        "same_scaffold_count_eq_0": {
            "n": int(test_summary["n_same_scaffold_count_eq_0"]),
            "spearman": float(test_summary["spearman_same_scaffold_count_eq_0"]),
        },
        "same_scaffold_count_ne_0": {
            "n": int(test_summary["n_same_scaffold_count_ne_0"]),
            "spearman": float(test_summary["spearman_same_scaffold_count_ne_0"]),
        },
        "solvent_count_eq_0": {
            "n": int(test_summary["n_solvent_count_eq_0"]),
            "spearman": float(test_summary["spearman_solvent_count_eq_0"]),
        },
        "solvent_count_ne_0": {
            "n": int(test_summary["n_solvent_count_ne_0"]),
            "spearman": float(test_summary["spearman_solvent_count_ne_0"]),
        },
        "solvent_reference_summary": solvent_reference_summary,
        "uses_solvent_residual_features": uses_solvent_residual,
        "solvent_bias_correction": bool(args.with_solvent_bias_correction),
        "solvent_residual_prior_strength": float(args.solvent_residual_prior_strength),
        "main_prediction_correction_metrics": correction_metrics,
        "main_prediction_correction_metrics_csv": (
            str(correction_metrics_path.resolve()) if correction_metrics_path else None
        ),
        "weighted_ae_mae": float(test_summary["ae_mae"]),
        "final_model": str(model_path.resolve()),
        "metrics_csv": str(metrics_path.resolve()),
        "short_task_metrics_csv": str(short_metrics_path.resolve()),
        "test_predictions_csv": str(predictions_path.resolve()),
    }
    report_path = out_dir / "test_report.json"
    report_tmp = report_path.with_suffix(".json.tmp")
    report_tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(report_tmp, report_path)
    print(metrics.to_string(index=False), flush=True)
    print("[RF] short task metrics", flush=True)
    print(short_metrics.to_string(index=False), flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
