from __future__ import annotations

import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
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
EPSILON_BY_PROPERTY = {"abs": 1.0, "emi": 1.0, "em": 0.01, "plqy": 0.005}
DROPPED_FEATURE_COLUMNS = [
    "base_prediction_fold_range",
    "calibration_same_scaffold_solvent_count",
    "calibration_same_scaffold_solvent_mean_abs_error",
]
MODEL_FEATURE_COLUMNS = [
    column for column in FINAL_FEATURE_COLUMNS if column not in DROPPED_FEATURE_COLUMNS
]
SOLVENT_AUGMENTED_MODEL_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS + list(SOLVENT_REFERENCE_FEATURE_NAMES)
HISTORICAL_FEATURE_COLUMNS = [
    "calibration_same_scaffold_count",
    "calibration_same_scaffold_mean_abs_error",
    "calibration_same_scaffold_median_abs_error",
    "calibration_same_scaffold_max_abs_error",
    "calibration_same_scaffold_std_abs_error",
    "calibration_same_scaffold_solvent_count",
    "calibration_same_scaffold_solvent_mean_abs_error",
]


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
        f"[Reliability] using selected {len(feature_columns)}-feature scheme; dropped={DROPPED_FEATURE_COLUMNS}",
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


