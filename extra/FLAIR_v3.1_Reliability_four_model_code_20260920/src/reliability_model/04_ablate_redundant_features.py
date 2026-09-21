from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from data_utils import get_scaffold_smiles, mol_from_smiles
from rank_ensemble_features import (
    FINAL_FEATURE_COLUMNS,
    PROPERTIES,
    add_calibration_scaffold_error_features,
)


step02 = importlib.import_module("02_generate_ae_pre_merged_features")

EPSILON_BY_PROPERTY = {"abs": 1.0, "emi": 1.0, "em": 0.01, "plqy": 0.005}
HISTORICAL_COLUMNS = [
    "calibration_same_scaffold_count",
    "calibration_same_scaffold_mean_abs_error",
    "calibration_same_scaffold_median_abs_error",
    "calibration_same_scaffold_max_abs_error",
    "calibration_same_scaffold_std_abs_error",
    "calibration_same_scaffold_solvent_count",
    "calibration_same_scaffold_solvent_mean_abs_error",
]
EXACT_SCAFFOLD_DROPS = [
    "solute_scaffold_novel",
    "solute_scaffold_train_and_val_count_log1p",
    "solute_scaffold_train_and_val_count_sqrt",
    "solute_scaffold_rare_in_train_and_val",
    "solute_scaffold_common_in_train_and_val",
]
EXACT_NEIGHBOR_DROPS = ["solute_neighbor_count_ge_0_5"]
VARIANT_DROPS = {
    "all_27": [],
    "drop_scaffold_novel": ["solute_scaffold_novel"],
    "drop_scaffold_count_derivatives": EXACT_SCAFFOLD_DROPS[1:],
    "drop_neighbor_ge_0_5_total": EXACT_NEIGHBOR_DROPS,
    "drop_fold_range": ["base_prediction_fold_range"],
    "drop_top3_mean_tanimoto": ["solute_top3_mean_tanimoto"],
    "drop_scaffold_median_error": ["calibration_same_scaffold_median_abs_error"],
    "drop_scaffold_max_error": ["calibration_same_scaffold_max_abs_error"],
    "drop_scaffold_std_error": ["calibration_same_scaffold_std_abs_error"],
    "drop_scaffold_solvent_history": [
        "calibration_same_scaffold_solvent_count",
        "calibration_same_scaffold_solvent_mean_abs_error",
    ],
    "drop_solvent_history_and_count_derivatives": [
        "calibration_same_scaffold_solvent_count",
        "calibration_same_scaffold_solvent_mean_abs_error",
        *EXACT_SCAFFOLD_DROPS[1:],
    ],
    "drop_solvent_history_and_fold_range": [
        "calibration_same_scaffold_solvent_count",
        "calibration_same_scaffold_solvent_mean_abs_error",
        "base_prediction_fold_range",
    ],
    "recommended_compact_20": [
        "calibration_same_scaffold_solvent_count",
        "calibration_same_scaffold_solvent_mean_abs_error",
        *EXACT_SCAFFOLD_DROPS[1:],
        "base_prediction_fold_range",
    ],
    "exact_redundancy_compact": EXACT_SCAFFOLD_DROPS + EXACT_NEIGHBOR_DROPS,
    "correlation_compact": EXACT_SCAFFOLD_DROPS
    + EXACT_NEIGHBOR_DROPS
    + [
        "base_prediction_fold_range",
        "solute_top3_mean_tanimoto",
        "calibration_same_scaffold_median_abs_error",
        "calibration_same_scaffold_max_abs_error",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe grouped-CV ablation of redundant Step 03 features.")
    parser.add_argument("--calibration_csv", default="../../results/intermediate/calibration_features_ae_pre.csv")
    parser.add_argument("--output_csv", default="../../results/diagnostics/ablation/redundancy_ablation_grouped_cv.csv")
    parser.add_argument("--output_json", default="../../results/diagnostics/ablation/redundancy_ablation_summary.json")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--variants", nargs="*", choices=list(VARIANT_DROPS), default=None)
    return parser.parse_args()


def spearman_manual(a, b) -> float:
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


def build_cv_frames(df: pd.DataFrame, n_splits: int) -> list[tuple[pd.DataFrame, set[int]]]:
    samples = df[["row_index", "smiles", "solvent"]].drop_duplicates().copy()
    if samples["row_index"].duplicated().any():
        raise ValueError("row_index maps to more than one solute-solvent pair.")
    samples["pair_group"] = step02.canonical_pair_keys(samples)
    splitter = GroupKFold(n_splits=n_splits)
    frames = []
    for cv_fold, (_, validation_positions) in enumerate(
        splitter.split(samples, groups=samples["pair_group"]),
        start=1,
    ):
        validation_ids = set(samples.iloc[validation_positions]["row_index"].astype(int))
        work = df.copy()
        is_validation = work["row_index"].astype(int).isin(validation_ids)
        work["split"] = np.where(is_validation, "test", "calibration")
        work.loc[is_validation, "abs_error"] = np.nan
        work = add_calibration_scaffold_error_features(work)
        if not np.isfinite(work[FINAL_FEATURE_COLUMNS].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite CV features in fold {cv_fold}.")
        frames.append((work, validation_ids))
        print(
            f"[Ablation] prepared CV fold {cv_fold}/{n_splits}: validation_samples={len(validation_ids)}",
            flush=True,
        )
    return frames


def evaluate_variant(
    name: str,
    feature_columns: list[str],
    cv_frames: list[tuple[pd.DataFrame, set[int]]],
    n_jobs: int,
) -> tuple[dict, pd.DataFrame]:
    predictions = []
    for cv_fold, (work, _) in enumerate(cv_frames, start=1):
        for prop in PROPERTIES:
            train = work[(work["split"] == "calibration") & (work["property"] == prop)]
            validation = work[(work["split"] == "test") & (work["property"] == prop)]
            model = RandomForestRegressor(
                n_estimators=150,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
                oob_score=False,
                random_state=42,
                n_jobs=n_jobs,
            )
            model.fit(
                train[feature_columns].to_numpy(dtype=float),
                train["log_abs_error"].to_numpy(dtype=float),
                sample_weight=train["sample_weight"].to_numpy(dtype=float),
            )
            pred_ae = inverse_log_error(model.predict(validation[feature_columns].to_numpy(dtype=float)), prop)
            fold_predictions = validation[
                ["property", "row_index", "fold", "y_true", "base_prediction", "calibration_same_scaffold_count"]
            ].copy()
            fold_predictions["predicted_ae"] = pred_ae
            fold_predictions["cv_fold"] = cv_fold
            predictions.append(fold_predictions)
    fold_rows = pd.concat(predictions, ignore_index=True)
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
    if not (sample_rows["n_fold_rows"] == 5).all():
        raise ValueError(f"Variant {name} does not have five prediction rows per validation sample/property.")
    sample_rows["true_ae"] = np.abs(sample_rows["base_prediction"] - sample_rows["y_true"])

    metric_rows = []
    for prop in PROPERTIES:
        sub = sample_rows[sample_rows["property"] == prop]
        zero = sub["same_scaffold_count"] == 0
        metric_rows.append(
            {
                "variant": name,
                "property": prop,
                "n_features": len(feature_columns),
                "n": len(sub),
                "spearman": spearman_manual(sub["predicted_ae"], sub["true_ae"]),
                "n_same_scaffold_count_eq_0": int(zero.sum()),
                "spearman_same_scaffold_count_eq_0": spearman_manual(
                    sub.loc[zero, "predicted_ae"], sub.loc[zero, "true_ae"]
                ),
                "n_same_scaffold_count_ne_0": int((~zero).sum()),
                "spearman_same_scaffold_count_ne_0": spearman_manual(
                    sub.loc[~zero, "predicted_ae"], sub.loc[~zero, "true_ae"]
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows)

    def weighted(column: str, weight: str = "n") -> float:
        valid = metrics[[column, weight]].replace([np.inf, -np.inf], np.nan).dropna()
        valid = valid[valid[weight] > 0]
        return float(np.average(valid[column], weights=valid[weight]))

    summary = {
        "variant": name,
        "n_features": len(feature_columns),
        "dropped_features": json.dumps(VARIANT_DROPS[name]),
        "weighted_spearman": weighted("spearman"),
        "n": int(metrics["n"].sum()),
        "n_same_scaffold_count_eq_0": int(metrics["n_same_scaffold_count_eq_0"].sum()),
        "spearman_same_scaffold_count_eq_0": weighted(
            "spearman_same_scaffold_count_eq_0", "n_same_scaffold_count_eq_0"
        ),
        "n_same_scaffold_count_ne_0": int(metrics["n_same_scaffold_count_ne_0"].sum()),
        "spearman_same_scaffold_count_ne_0": weighted(
            "spearman_same_scaffold_count_ne_0", "n_same_scaffold_count_ne_0"
        ),
    }
    print(
        f"[Ablation] {name}: features={len(feature_columns)} weighted_spearman={summary['weighted_spearman']:.6f}",
        flush=True,
    )
    return summary, metrics


def main() -> None:
    args = parse_args()
    os.chdir(Path(__file__).resolve().parent)
    calibration_path = Path(args.calibration_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json = Path(args.output_json).resolve()
    df = pd.read_csv(calibration_path)
    required = set(FINAL_FEATURE_COLUMNS) | {
        "property",
        "row_index",
        "fold",
        "smiles",
        "solvent",
        "y_true",
        "abs_error",
        "log_abs_error",
        "sample_weight",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Calibration feature table is missing columns: {missing}")
    if set(df["split"].astype(str)) != {"calibration"}:
        raise ValueError("Ablation accepts calibration rows only.")
    df = add_direct_scaffold(df)
    cv_frames = build_cv_frames(df, args.n_splits)

    summaries = []
    property_metrics = []
    selected_variants = args.variants or list(VARIANT_DROPS)
    for name in selected_variants:
        dropped = VARIANT_DROPS[name]
        features = [column for column in FINAL_FEATURE_COLUMNS if column not in dropped]
        summary, metrics = evaluate_variant(name, features, cv_frames, args.n_jobs)
        summaries.append(summary)
        property_metrics.append(metrics)

    summary_df = pd.DataFrame(summaries)
    if "all_27" in set(summary_df["variant"]):
        baseline = float(summary_df.loc[summary_df["variant"] == "all_27", "weighted_spearman"].iloc[0])
        summary_df["delta_vs_all_27"] = summary_df["weighted_spearman"] - baseline
    else:
        summary_df["delta_vs_all_27"] = np.nan
    summary_df = summary_df.sort_values("weighted_spearman", ascending=False).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = output_csv.with_suffix(".csv.tmp")
    summary_df.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, output_csv)

    result = {
        "method": "5-fold canonical-solute-solvent GroupKFold within Calibration85 only",
        "test15_read": False,
        "historical_residual_features": "recomputed from each CV training fold only",
        "model": {
            "n_estimators": 150,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": 42,
        },
        "feature_columns": list(FINAL_FEATURE_COLUMNS),
        "results": summary_df.to_dict(orient="records"),
        "property_metrics": pd.concat(property_metrics, ignore_index=True).to_dict(orient="records"),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = output_json.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_json, output_json)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
