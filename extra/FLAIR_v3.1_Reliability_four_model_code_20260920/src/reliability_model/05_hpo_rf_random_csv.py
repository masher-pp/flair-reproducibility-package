from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


hpo = importlib.import_module("05_hpo_rf_staircase")

SCRIPT_DIR = Path(__file__).resolve().parent
PLAN_VERSION_24F = "rf_24f_property_specific_ensemble_base_random_csv_v3"
PLAN_VERSION_32F = "rf_32f_solvent_property_specific_ensemble_base_random_csv_v4"
REQUIRED_COLUMNS = [
    "trial_id",
    "n_estimators",
    "min_samples_leaf",
    "min_samples_split",
    "max_features",
    "max_depth",
    "max_leaf_nodes",
    "max_samples",
    "criterion",
    "bootstrap",
    "rf_seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe RF HPO from a pre-generated candidate CSV.")
    parser.add_argument("--calibration_csv", default="../../results/intermediate/calibration_features_ae_pre.csv")
    parser.add_argument("--train_and_val_csv", default="../../data/splits/deployment/deployment.csv")
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--out_dir", default="../../results/hpo/property_specific_ensemble_100")
    parser.add_argument("--model_dir", default="../../models/reliability_model/property_specific_ensemble_hpo_100")
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--with_solvent_reference_features",
        action="store_true",
        help="Run HPO on the current 32-feature solvent occurrence/Tanimoto scheme.",
    )
    return parser.parse_args()


def optional_int(value: str) -> int | None:
    return None if value.strip().lower() == "none" else int(value)


def optional_float(value: str) -> float | None:
    return None if value.strip().lower() == "none" else float(value)


def max_features_value(value: str) -> str | float:
    text = value.strip().lower()
    return "sqrt" if text == "sqrt" else float(text)


def load_candidate_trials(path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Candidate CSV is missing columns: {missing}")
    if len(raw) != 100:
        raise ValueError(f"Expected exactly 100 random candidates, found {len(raw)}.")
    if raw["trial_id"].duplicated().any():
        raise ValueError("Candidate CSV contains duplicate trial_id values.")
    rf_seeds = {int(value) for value in raw["rf_seed"]}
    if len(rf_seeds) != 1:
        raise ValueError("All candidates must share one rf_seed for fair comparison.")

    trials = []
    seen = set()
    for row in raw.to_dict(orient="records"):
        if row["bootstrap"].strip().lower() != "true":
            raise ValueError("bootstrap must remain true for every candidate.")
        params = {
            "n_estimators": int(row["n_estimators"]),
            "min_samples_leaf": int(row["min_samples_leaf"]),
            "min_samples_split": int(row["min_samples_split"]),
            "max_features": max_features_value(row["max_features"]),
            "max_depth": optional_int(row["max_depth"]),
            "max_leaf_nodes": optional_int(row["max_leaf_nodes"]),
            "max_samples": optional_float(row["max_samples"]),
            "criterion": row["criterion"],
        }
        key = hpo.stable_json(params)
        if key in seen:
            raise ValueError(f"Duplicate parameter configuration in candidate CSV: {row['trial_id']}")
        seen.add(key)
        trials.append({"label": f"random:{row['trial_id']}", "params": params})
    return raw, trials, rf_seeds.pop()


def initialize_manifest(
    out_dir: Path,
    calibration_path: Path,
    candidate_path: Path,
    n_splits: int,
    split_seed: int,
    reset: bool,
    plan_version: str,
) -> dict:
    if reset and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = hpo.manifest_payload(calibration_path)
    manifest.update(
        {
            "plan_version": plan_version,
            "candidate_csv": str(candidate_path),
            "candidate_csv_sha256": hpo.file_sha256(candidate_path),
            "candidate_count": 100,
            "cv_spec": {"n_splits": n_splits, "shuffle": True, "split_seed": split_seed},
            "test_data_used": False,
        }
    )
    manifest["code_sha256"][Path(__file__).name] = hpo.file_sha256(Path(__file__).resolve())
    path = out_dir / "hpo_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        keys = [
            "plan_version",
            "calibration_sha256",
            "candidate_csv_sha256",
            "candidate_count",
            "feature_columns",
            "code_sha256",
            "cv_spec",
        ]
        mismatches = [key for key in keys if existing.get(key) != manifest.get(key)]
        if mismatches:
            raise ValueError(f"Existing random HPO output is incompatible ({mismatches}); use --reset.")
    else:
        hpo.atomic_json(manifest, path)
    return manifest


def select_property_candidates(result: pd.DataFrame, out_dir: Path, plan_version: str) -> dict:
    leaderboard = result.copy()
    for prop in hpo.PROPERTIES:
        leaderboard[f"{prop}_rank"] = leaderboard[f"{prop}_spearman"].rank(
            method="min", ascending=False
        ).astype(int)
    hpo.atomic_csv(leaderboard, out_dir / "hpo_property_specific_leaderboard.csv")

    selected_by_property: dict[str, dict[str, Any]] = {}
    selection_rows = []
    for prop in hpo.PROPERTIES:
        selected = leaderboard.sort_values(
            [f"{prop}_spearman", "estimated_model_size_mb"], ascending=[False, True]
        ).iloc[0]
        params = json.loads(str(selected["params_json"]))
        record = {
            "property": prop,
            "selected_label": str(selected["label"]),
            "selected_config_id": str(selected["config_id"]),
            "cv_spearman": float(selected[f"{prop}_spearman"]),
            "estimated_four_model_size_mb": float(selected["estimated_model_size_mb"]),
            "params": params,
        }
        selected_by_property[prop] = record
        selection_rows.append(
            {
                **{key: value for key, value in record.items() if key != "params"},
                "params_json": hpo.stable_json(params),
            }
        )
    hpo.atomic_csv(pd.DataFrame(selection_rows), out_dir / "hpo_property_specific_selection.csv")
    selection = {
        "plan_version": plan_version,
        "selection_objective": "highest grouped-CV Spearman independently for each property",
        "selected_by_property": selected_by_property,
        "test_data_used": False,
    }
    hpo.atomic_json(selection, out_dir / "hpo_property_specific_selection.json")
    return selection


def train_property_models(
    *,
    calibration: pd.DataFrame,
    selection: dict,
    model_dir: Path,
    out_dir: Path,
    n_jobs: int,
) -> dict:
    feature_columns = list(hpo.MODEL_FEATURE_COLUMNS)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_paths: dict[str, str] = {}
    final_rows = []

    for prop in hpo.PROPERTIES:
        selected = selection["selected_by_property"][prop]
        params = dict(selected["params"])
        train = calibration[calibration["property"] == prop]
        model = RandomForestRegressor(
            **params,
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=n_jobs,
        )
        model.fit(
            train[feature_columns].to_numpy(dtype=float),
            train["log_abs_error"].to_numpy(dtype=float),
            sample_weight=train["sample_weight"].to_numpy(dtype=float),
        )

        model_path = model_dir / f"{prop}_rf.pkl"
        model_tmp = model_path.with_suffix(".pkl.tmp")
        with model_tmp.open("wb") as handle:
            pickle.dump(
                {
                    "property": prop,
                    "model": model,
                    "params": params,
                    "feature_columns": feature_columns,
                    "epsilon": hpo.EPSILON_BY_PROPERTY[prop],
                    "selected_config_id": selected["selected_config_id"],
                    "selection_cv_spearman": selected["cv_spearman"],
                    "base_prediction_type": "mean_of_5_base_models_before_ae_prediction",
                    "training_sample_count": int(len(train)),
                },
                handle,
            )
        os.replace(model_tmp, model_path)
        model_paths[prop] = str(model_path.resolve())
        final_rows.append(
            {
                "property": prop,
                "selected_config_id": selected["selected_config_id"],
                "cv_spearman": selected["cv_spearman"],
                "training_sample_count": int(len(train)),
                "oob_r2_log_target_diagnostic": float(model.oob_score_),
                "model_size_mb": float(model_path.stat().st_size / (1024**2)),
                "model_path": str(model_path.resolve()),
                "params_json": hpo.stable_json(params),
            }
        )
    final_models = pd.DataFrame(final_rows)
    hpo.atomic_csv(final_models, out_dir / "property_specific_final_models.csv")
    report = {
        "selection": selection,
        "model_paths": model_paths,
        "final_models_csv": str((out_dir / "property_specific_final_models.csv").resolve()),
        "test_data_used": False,
    }
    hpo.atomic_json(report, out_dir / "property_specific_final_report.json")
    return report


def main() -> None:
    args = parse_args()
    os.chdir(SCRIPT_DIR)
    calibration_path = hpo.resolve_from_script(args.calibration_csv)
    train_and_val_path = hpo.resolve_from_script(args.train_and_val_csv)
    candidate_path = Path(args.candidate_csv).expanduser().resolve()
    out_dir = hpo.resolve_from_script(args.out_dir)
    model_dir = hpo.resolve_from_script(args.model_dir)
    raw, random_trials, rf_seed = load_candidate_trials(candidate_path)
    calibration = hpo.load_and_validate_calibration(calibration_path)
    calibration = hpo.step03.aggregate_base_predictions(calibration, include_targets=True)
    plan_version = PLAN_VERSION_24F
    if args.with_solvent_reference_features:
        if not train_and_val_path.exists():
            raise FileNotFoundError(train_and_val_path)
        empty_test = calibration.iloc[0:0].copy()
        calibration, _, solvent_summary = hpo.step03.add_solvent_reference_feature_columns(
            calibration,
            empty_test,
            str(train_and_val_path),
        )
        hpo.MODEL_FEATURE_COLUMNS = list(hpo.step03.SOLVENT_AUGMENTED_MODEL_FEATURE_COLUMNS)
        plan_version = PLAN_VERSION_32F
        print(f"[Random HPO] 32-feature solvent reference: {solvent_summary}", flush=True)
    if args.check:
        print(
            f"[Random HPO] CHECK PASSED: candidates={len(raw)}, calibration_rows={len(calibration)}, "
            f"features={len(hpo.MODEL_FEATURE_COLUMNS)}, ensemble_rows_per_sample=1, "
            "Test inputs are not accepted.",
            flush=True,
        )
        return

    hpo.LOG_PATH = out_dir / "hpo_random_run.log"
    if args.reset and model_dir.exists():
        shutil.rmtree(model_dir)
    manifest = initialize_manifest(
        out_dir,
        calibration_path,
        candidate_path,
        args.n_splits,
        args.split_seed,
        args.reset,
        plan_version,
    )
    calibration = hpo.add_direct_scaffold(calibration)
    calibration = hpo.add_calibration_scaffold_error_features(calibration)
    trials = random_trials
    result = hpo.run_stage(
        stage=1,
        name="random_100",
        trials=trials,
        cv_spec={"n_splits": args.n_splits, "shuffle": True, "split_seed": args.split_seed},
        rf_seed=rf_seed,
        df_with_scaffold=calibration,
        out_dir=out_dir,
        manifest=manifest,
        n_jobs=args.n_jobs,
    )
    selection = select_property_candidates(result, out_dir, plan_version)
    report = train_property_models(
        calibration=calibration,
        selection=selection,
        model_dir=model_dir,
        out_dir=out_dir,
        n_jobs=args.n_jobs,
    )
    print("[Random HPO] COMPLETE", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
