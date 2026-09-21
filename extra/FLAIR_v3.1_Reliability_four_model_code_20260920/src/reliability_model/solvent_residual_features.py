from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from data_utils import canonical_smiles


DEFAULT_SOLVENT_RESIDUAL_PRIOR_STRENGTH = 10.0
SOLVENT_RESIDUAL_FEATURE_NAMES = [
    "calibration_same_solvent_error_count",
    "calibration_same_solvent_mean_signed_error",
    "calibration_same_solvent_mean_abs_error",
    "calibration_same_solvent_rmse",
    "calibration_same_solvent_std_signed_error",
    "calibration_same_solvent_positive_error_fraction",
]


def _canonical_solvent(value: object) -> str:
    canonical = canonical_smiles(value)
    if canonical:
        return canonical
    raw = "" if pd.isna(value) else str(value).strip()
    return f"RAW::{raw}"


def _moments(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "count": float(len(values)),
        "sum": float(values.sum()),
        "sum_abs": float(np.abs(values).sum()),
        "sum_sq": float(np.square(values).sum()),
        "positive": float((values > 0).sum()),
    }


def _subtract(moment: Dict[str, float], value: float) -> Dict[str, float]:
    return {
        "count": moment["count"] - 1.0,
        "sum": moment["sum"] - value,
        "sum_abs": moment["sum_abs"] - abs(value),
        "sum_sq": moment["sum_sq"] - value * value,
        "positive": moment["positive"] - float(value > 0),
    }


def _feature_values(
    group: Dict[str, float],
    global_moment: Dict[str, float],
    prior_strength: float,
) -> Dict[str, float]:
    n = max(float(group["count"]), 0.0)
    global_n = max(float(global_moment["count"]), 1.0)
    global_mean = float(global_moment["sum"] / global_n)
    global_mae = float(global_moment["sum_abs"] / global_n)
    global_mse = float(global_moment["sum_sq"] / global_n)
    global_positive = float(global_moment["positive"] / global_n)
    denominator = n + float(prior_strength)
    mean_signed = float((group["sum"] + prior_strength * global_mean) / denominator)
    mean_abs = float((group["sum_abs"] + prior_strength * global_mae) / denominator)
    mean_sq = float((group["sum_sq"] + prior_strength * global_mse) / denominator)
    positive_fraction = float(
        (group["positive"] + prior_strength * global_positive) / denominator
    )
    return {
        "calibration_same_solvent_error_count": n,
        "calibration_same_solvent_mean_signed_error": mean_signed,
        "calibration_same_solvent_mean_abs_error": mean_abs,
        "calibration_same_solvent_rmse": float(np.sqrt(max(mean_sq, 0.0))),
        "calibration_same_solvent_std_signed_error": float(
            np.sqrt(max(mean_sq - mean_signed * mean_signed, 0.0))
        ),
        "calibration_same_solvent_positive_error_fraction": positive_fraction,
    }


def build_solvent_residual_reference(calibration_df: pd.DataFrame) -> Dict:
    required = {"property", "solvent", "base_prediction", "y_true"}
    missing = sorted(required - set(calibration_df.columns))
    if missing:
        raise ValueError(f"Calibration residual reference is missing columns: {missing}")
    work = calibration_df[list(required)].copy()
    work["_solvent_key"] = work["solvent"].map(_canonical_solvent)
    work["_signed_error"] = (
        work["base_prediction"].to_numpy(dtype=float)
        - work["y_true"].to_numpy(dtype=float)
    )
    by_property: Dict[str, Dict] = {}
    for prop, prop_df in work.groupby("property", sort=False):
        values = prop_df["_signed_error"].to_numpy(dtype=float)
        groups = {
            solvent: _moments(group["_signed_error"].to_numpy(dtype=float))
            for solvent, group in prop_df.groupby("_solvent_key", sort=False)
        }
        by_property[str(prop)] = {"global": _moments(values), "solvent": groups}
    return {"by_property": by_property}


def solvent_residual_feature_row(
    solvent: object,
    prop: str,
    reference: Dict,
    *,
    prior_strength: float = DEFAULT_SOLVENT_RESIDUAL_PRIOR_STRENGTH,
) -> Dict[str, float]:
    prop_reference = reference["by_property"][str(prop)]
    group = prop_reference["solvent"].get(
        _canonical_solvent(solvent),
        {"count": 0.0, "sum": 0.0, "sum_abs": 0.0, "sum_sq": 0.0, "positive": 0.0},
    )
    return _feature_values(group, prop_reference["global"], prior_strength)


def add_solvent_residual_feature_columns(
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    prior_strength: float = DEFAULT_SOLVENT_RESIDUAL_PRIOR_STRENGTH,
    apply_bias_correction: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Add property-specific solvent residual history without target leakage.

    Calibration rows use leave-one-row-out moments. Test rows use the complete
    calibration reference. Bias correction subtracts the shrunken signed error.
    """
    calibration = calibration_df.copy()
    test = test_df.copy()
    calibration["raw_base_prediction"] = calibration["base_prediction"].to_numpy(dtype=float)
    test["raw_base_prediction"] = test["base_prediction"].to_numpy(dtype=float)
    calibration["_solvent_key"] = calibration["solvent"].map(_canonical_solvent)
    test["_solvent_key"] = test["solvent"].map(_canonical_solvent)
    calibration["_signed_error"] = (
        calibration["raw_base_prediction"].to_numpy(dtype=float)
        - calibration["y_true"].to_numpy(dtype=float)
    )

    calibration_rows: dict[int, Dict[str, float]] = {}
    test_rows: dict[int, Dict[str, float]] = {}
    reference_by_property: Dict[str, Dict] = {}
    for prop, prop_df in calibration.groupby("property", sort=False):
        global_moment = _moments(prop_df["_signed_error"].to_numpy(dtype=float))
        group_moments = {
            solvent: _moments(group["_signed_error"].to_numpy(dtype=float))
            for solvent, group in prop_df.groupby("_solvent_key", sort=False)
        }
        reference_by_property[str(prop)] = {
            "global": global_moment,
            "solvent": group_moments,
        }
        for index, row in prop_df.iterrows():
            value = float(row["_signed_error"])
            loo_global = _subtract(global_moment, value)
            loo_group = _subtract(group_moments[row["_solvent_key"]], value)
            calibration_rows[index] = _feature_values(loo_group, loo_global, prior_strength)

        prop_test = test[test["property"] == prop]
        empty = {"count": 0.0, "sum": 0.0, "sum_abs": 0.0, "sum_sq": 0.0, "positive": 0.0}
        for index, row in prop_test.iterrows():
            group = group_moments.get(row["_solvent_key"], empty)
            test_rows[index] = _feature_values(group, global_moment, prior_strength)

    calibration_features = pd.DataFrame.from_dict(calibration_rows, orient="index")
    test_features = pd.DataFrame.from_dict(test_rows, orient="index")
    calibration[SOLVENT_RESIDUAL_FEATURE_NAMES] = calibration_features.loc[
        calibration.index, SOLVENT_RESIDUAL_FEATURE_NAMES
    ]
    test[SOLVENT_RESIDUAL_FEATURE_NAMES] = test_features.loc[
        test.index, SOLVENT_RESIDUAL_FEATURE_NAMES
    ]
    if apply_bias_correction:
        calibration["base_prediction"] = (
            calibration["raw_base_prediction"]
            - calibration["calibration_same_solvent_mean_signed_error"]
        )
        test["base_prediction"] = (
            test["raw_base_prediction"]
            - test["calibration_same_solvent_mean_signed_error"]
        )

    calibration = calibration.drop(columns=["_solvent_key", "_signed_error"])
    test = test.drop(columns=["_solvent_key"])
    return calibration, test, {"by_property": reference_by_property}
