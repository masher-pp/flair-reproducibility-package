from __future__ import annotations

import numpy as np
import pandas as pd

from data_utils import TANIMOTO_NEIGHBOR_FEATURE_NAMES, get_scaffold_smiles, mol_from_smiles


PROPERTIES = ["plqy", "emi", "em", "abs"]
BASE_FEATURE_COLUMNS = [
    "solute_max_tanimoto",
    "solute_scaffold_novel",
    "solute_scaffold_similarity",
    "base_prediction",
]
ENHANCED_SCAFFOLD_FEATURE_COLUMNS = [
    "solute_scaffold_in_train_and_val",
    "solute_scaffold_train_and_val_count",
    "solute_scaffold_train_and_val_count_log1p",
    "solute_scaffold_train_and_val_count_sqrt",
    "solute_scaffold_rare_in_train_and_val",
    "solute_scaffold_common_in_train_and_val",
    "calibration_same_scaffold_count",
    "calibration_same_scaffold_mean_abs_error",
]
UNCERTAINTY_SCAFFOLD_FEATURE_COLUMNS = ENHANCED_SCAFFOLD_FEATURE_COLUMNS + [
    "base_prediction_fold_std",
    "base_prediction_fold_range",
    "calibration_same_scaffold_median_abs_error",
    "calibration_same_scaffold_max_abs_error",
    "calibration_same_scaffold_std_abs_error",
    "calibration_same_scaffold_solvent_count",
    "calibration_same_scaffold_solvent_mean_abs_error",
]
NEIGHBOR_FEATURE_COLUMNS = list(TANIMOTO_NEIGHBOR_FEATURE_NAMES)
FINAL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + UNCERTAINTY_SCAFFOLD_FEATURE_COLUMNS + NEIGHBOR_FEATURE_COLUMNS


def build_scaffold_counts(train_and_val_csv: str) -> dict[str, int]:
    reference = pd.read_csv(train_and_val_csv)
    if "_scaffold" in reference.columns:
        scaffolds = reference["_scaffold"].fillna("").astype(str).str.strip()
    else:
        scaffolds = reference["smiles"].map(lambda smiles: get_scaffold_smiles(mol_from_smiles(smiles)))
    scaffolds = scaffolds[scaffolds != ""]
    return scaffolds.value_counts().astype(int).to_dict()


def add_direct_scaffold_count_features(df: pd.DataFrame, scaffold_counts: dict[str, int]) -> pd.DataFrame:
    scaffold_by_smiles: dict[str, str] = {}

    def scaffold_for_smiles(smiles: object) -> str:
        key = "" if pd.isna(smiles) else str(smiles).strip()
        if not key:
            return ""
        if key not in scaffold_by_smiles:
            scaffold_by_smiles[key] = get_scaffold_smiles(mol_from_smiles(key))
        return scaffold_by_smiles[key]

    result = df.copy()
    result["_direct_scaffold"] = result["smiles"].map(scaffold_for_smiles)
    counts = result["_direct_scaffold"].map(lambda scaffold: int(scaffold_counts.get(scaffold, 0))).astype(float)
    result["solute_scaffold_train_and_val_count"] = counts
    result["solute_scaffold_in_train_and_val"] = (counts > 0).astype(float)
    result["solute_scaffold_train_and_val_count_log1p"] = np.log1p(counts)
    result["solute_scaffold_train_and_val_count_sqrt"] = np.sqrt(counts)
    result["solute_scaffold_rare_in_train_and_val"] = ((counts > 0) & (counts <= 2)).astype(float)
    result["solute_scaffold_common_in_train_and_val"] = (counts >= 10).astype(float)
    return result


def add_prediction_uncertainty_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    grouped = result.groupby(["split", "row_index", "property"])["y_pred"].agg(
        pred_std=lambda values: float(values.std(ddof=0)),
        pred_min="min",
        pred_max="max",
    )
    keys = list(zip(result["split"], result["row_index"], result["property"]))
    result["base_prediction_fold_std"] = [
        float(grouped.loc[key, "pred_std"])
        if key in grouped.index and np.isfinite(grouped.loc[key, "pred_std"])
        else 0.0
        for key in keys
    ]
    result["base_prediction_fold_range"] = [
        float(grouped.loc[key, "pred_max"] - grouped.loc[key, "pred_min"]) if key in grouped.index else 0.0
        for key in keys
    ]
    return result


def add_calibration_scaffold_error_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["calibration_same_scaffold_count"] = 0.0
    result["calibration_same_scaffold_mean_abs_error"] = np.nan
    result["calibration_same_scaffold_median_abs_error"] = np.nan
    result["calibration_same_scaffold_max_abs_error"] = np.nan
    result["calibration_same_scaffold_std_abs_error"] = np.nan
    result["calibration_same_scaffold_solvent_count"] = 0.0
    result["calibration_same_scaffold_solvent_mean_abs_error"] = np.nan

    for prop in PROPERTIES:
        prop_mask = result["property"] == prop
        train_mask = prop_mask & (result["split"] == "calibration")
        test_mask = prop_mask & (result["split"] == "test")
        train_df = result.loc[train_mask, ["_direct_scaffold", "solvent", "row_index", "abs_error"]].copy()
        global_mean = float(train_df["abs_error"].mean())
        global_median = float(train_df["abs_error"].median())
        global_max = float(train_df["abs_error"].max())
        global_std = float(train_df["abs_error"].std(ddof=0))

        all_row_indices = train_df["row_index"].to_numpy()
        all_errors = train_df["abs_error"].to_numpy(dtype=float)
        global_leave_one_row_stats: dict[object, tuple[float, float, float, float]] = {}
        for row_index in pd.unique(all_row_indices):
            other_global_errors = all_errors[all_row_indices != row_index]
            if len(other_global_errors) == 0:
                raise ValueError(f"Property {prop} needs more than one calibration row_index.")
            global_leave_one_row_stats[row_index] = (
                float(np.mean(other_global_errors)),
                float(np.median(other_global_errors)),
                float(np.max(other_global_errors)),
                float(np.std(other_global_errors, ddof=0)),
            )

        scaffold_errors = train_df.groupby("_direct_scaffold")["abs_error"]
        by_scaffold = scaffold_errors.agg(["count", "sum", "median", "max"])
        by_scaffold["std"] = scaffold_errors.apply(lambda values: float(values.std(ddof=0)))
        by_scaffold_solvent = train_df.groupby(["_direct_scaffold", "solvent"])["abs_error"].agg(["count", "sum"])
        by_row_scaffold_solvent = train_df.groupby(["_direct_scaffold", "solvent", "row_index"])["abs_error"].agg(["count", "sum"])

        leave_one_row_stats: dict[tuple[str, object], tuple[float, float, float, float, float]] = {}
        for scaffold, scaffold_df in train_df.groupby("_direct_scaffold", sort=False):
            row_indices = scaffold_df["row_index"].to_numpy()
            errors = scaffold_df["abs_error"].to_numpy(dtype=float)
            for row_index in pd.unique(row_indices):
                other_errors = errors[row_indices != row_index]
                if len(other_errors) == 0:
                    fallback_mean, fallback_median, fallback_max, fallback_std = global_leave_one_row_stats[row_index]
                    leave_one_row_stats[(scaffold, row_index)] = (
                        0.0,
                        fallback_mean,
                        fallback_median,
                        fallback_max,
                        fallback_std,
                    )
                else:
                    leave_one_row_stats[(scaffold, row_index)] = (
                        float(len(other_errors)),
                        float(np.mean(other_errors)),
                        float(np.median(other_errors)),
                        float(np.max(other_errors)),
                        float(np.std(other_errors, ddof=0)),
                    )

        train_values = []
        for _, row in train_df.iterrows():
            scaffold = row["_direct_scaffold"]
            solvent = row["solvent"]
            row_key = (scaffold, row["row_index"])
            solvent_key = (scaffold, solvent)
            row_solvent_key = (scaffold, solvent, row["row_index"])
            other_count, mean_error, median_error, max_error, std_error = leave_one_row_stats[row_key]

            solvent_count = float(by_scaffold_solvent.loc[solvent_key, "count"]) if solvent_key in by_scaffold_solvent.index else 0.0
            solvent_sum = float(by_scaffold_solvent.loc[solvent_key, "sum"]) if solvent_key in by_scaffold_solvent.index else 0.0
            own_solvent_count = (
                float(by_row_scaffold_solvent.loc[row_solvent_key, "count"])
                if row_solvent_key in by_row_scaffold_solvent.index
                else 0.0
            )
            own_solvent_sum = (
                float(by_row_scaffold_solvent.loc[row_solvent_key, "sum"])
                if row_solvent_key in by_row_scaffold_solvent.index
                else 0.0
            )
            other_solvent_count = max(solvent_count - own_solvent_count, 0.0)
            other_solvent_sum = solvent_sum - own_solvent_sum
            solvent_mean_error = (
                other_solvent_sum / other_solvent_count
                if other_solvent_count > 0
                else global_leave_one_row_stats[row["row_index"]][0]
            )
            train_values.append(
                (
                    other_count,
                    mean_error,
                    median_error,
                    max_error,
                    std_error,
                    other_solvent_count,
                    solvent_mean_error,
                )
            )

        if train_values:
            train_index = result.index[train_mask]
            result.loc[train_index, "calibration_same_scaffold_count"] = [value[0] for value in train_values]
            result.loc[train_index, "calibration_same_scaffold_mean_abs_error"] = [value[1] for value in train_values]
            result.loc[train_index, "calibration_same_scaffold_median_abs_error"] = [value[2] for value in train_values]
            result.loc[train_index, "calibration_same_scaffold_max_abs_error"] = [value[3] for value in train_values]
            result.loc[train_index, "calibration_same_scaffold_std_abs_error"] = [value[4] for value in train_values]
            result.loc[train_index, "calibration_same_scaffold_solvent_count"] = [value[5] for value in train_values]
            result.loc[train_index, "calibration_same_scaffold_solvent_mean_abs_error"] = [value[6] for value in train_values]

        test_stats = by_scaffold.copy()
        test_stats["mean"] = test_stats["sum"] / test_stats["count"]
        test_solvent_stats = by_scaffold_solvent.copy()
        test_solvent_stats["mean"] = test_solvent_stats["sum"] / test_solvent_stats["count"]
        test_scaffolds = result.loc[test_mask, "_direct_scaffold"]
        test_scaffold_solvent_keys = list(zip(result.loc[test_mask, "_direct_scaffold"], result.loc[test_mask, "solvent"]))
        result.loc[test_mask, "calibration_same_scaffold_count"] = (
            test_scaffolds.map(test_stats["count"]).fillna(0.0).astype(float).to_numpy()
        )
        result.loc[test_mask, "calibration_same_scaffold_mean_abs_error"] = (
            test_scaffolds.map(test_stats["mean"]).fillna(global_mean).astype(float).to_numpy()
        )
        result.loc[test_mask, "calibration_same_scaffold_median_abs_error"] = (
            test_scaffolds.map(test_stats["median"]).fillna(global_median).astype(float).to_numpy()
        )
        result.loc[test_mask, "calibration_same_scaffold_max_abs_error"] = (
            test_scaffolds.map(test_stats["max"]).fillna(global_max).astype(float).to_numpy()
        )
        result.loc[test_mask, "calibration_same_scaffold_std_abs_error"] = (
            test_scaffolds.map(test_stats["std"]).fillna(global_std).astype(float).to_numpy()
        )
        result.loc[test_mask, "calibration_same_scaffold_solvent_count"] = [
            float(test_solvent_stats.loc[key, "count"]) if key in test_solvent_stats.index else 0.0
            for key in test_scaffold_solvent_keys
        ]
        result.loc[test_mask, "calibration_same_scaffold_solvent_mean_abs_error"] = [
            float(test_solvent_stats.loc[key, "mean"]) if key in test_solvent_stats.index else global_mean
            for key in test_scaffold_solvent_keys
        ]

    fill_columns = [
        "calibration_same_scaffold_mean_abs_error",
        "calibration_same_scaffold_median_abs_error",
        "calibration_same_scaffold_max_abs_error",
        "calibration_same_scaffold_std_abs_error",
        "calibration_same_scaffold_solvent_mean_abs_error",
    ]
    calibration_mask = result["split"] == "calibration"
    calibration_missing = result.loc[calibration_mask, fill_columns].isna().sum()
    if int(calibration_missing.sum()) > 0:
        missing = calibration_missing[calibration_missing > 0].to_dict()
        raise ValueError(f"Calibration residual features unexpectedly contain missing values: {missing}")
    calibration_mean_by_property = result.loc[calibration_mask].groupby("property")["abs_error"].mean()
    calibration_global_mean = float(result.loc[calibration_mask, "abs_error"].mean())
    for column in fill_columns:
        result[column] = result[column].replace([np.inf, -np.inf], np.nan)
        missing_mask = result[column].isna()
        result.loc[missing_mask, column] = (
            result.loc[missing_mask, "property"]
            .map(calibration_mean_by_property)
            .fillna(calibration_global_mean)
            .to_numpy()
        )
    return result


def engineer_rank_ensemble_features(df: pd.DataFrame, train_and_val_csv: str) -> pd.DataFrame:
    missing_neighbor = [column for column in NEIGHBOR_FEATURE_COLUMNS if column not in df.columns]
    if missing_neighbor:
        raise ValueError(f"Offline rows are missing Step 02 Tanimoto neighbor features: {missing_neighbor}")
    result = add_direct_scaffold_count_features(df, build_scaffold_counts(train_and_val_csv))
    result = add_prediction_uncertainty_features(result)
    result = add_calibration_scaffold_error_features(result)
    return result.drop(columns=["_direct_scaffold"], errors="ignore")
