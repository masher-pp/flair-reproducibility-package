from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_DIR = "../../data/splits/deployment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated external-test + 5-fold split CSV files.")
    parser.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--n_folds", type=int, default=5)
    return parser.parse_args()


def row_id_set(df: pd.DataFrame) -> set[int]:
    if "_original_row_id" not in df.columns:
        raise ValueError("Missing _original_row_id column.")
    return set(pd.to_numeric(df["_original_row_id"], errors="raise").astype(int).tolist())


def check_no_overlap(name_a: str, ids_a: set[int], name_b: str, ids_b: set[int]) -> None:
    overlap = ids_a & ids_b
    if overlap:
        raise RuntimeError(f"{name_a} and {name_b} overlap: {len(overlap)} rows.")


def main() -> None:
    args = parse_args()
    raw_split_dir = Path(args.split_dir).expanduser()
    split_dir = raw_split_dir.resolve() if raw_split_dir.is_absolute() else (SCRIPT_DIR / raw_split_dir).resolve()

    deployment_test = pd.read_csv(split_dir / "deployment_test.csv")
    deployment = pd.read_csv(split_dir / "deployment.csv")
    if "cv_fold" not in deployment.columns:
        raise ValueError("deployment.csv is missing the cv_fold column.")
    fold_values = pd.to_numeric(deployment["cv_fold"], errors="raise").astype(int)
    expected_folds = set(range(1, args.n_folds + 1))
    actual_folds = set(fold_values.tolist())
    if actual_folds != expected_folds:
        raise ValueError(f"Expected cv_fold values {sorted(expected_folds)}, found {sorted(actual_folds)}.")
    deployment_test_ids = row_id_set(deployment_test)
    deployment_ids = row_id_set(deployment)
    check_no_overlap("Deployment_Test", deployment_test_ids, "Deployment", deployment_ids)

    fold_summary = []
    reference_test = pd.read_csv(split_dir / "deployment_test.csv")
    reference_test_ids = row_id_set(reference_test)
    if reference_test_ids != deployment_test_ids:
        raise RuntimeError("deployment_test.csv is not internally consistent by row ids.")

    for fold_idx in range(1, args.n_folds + 1):
        train = deployment.loc[fold_values.ne(fold_idx)].copy()
        val = deployment.loc[fold_values.eq(fold_idx)].copy()
        test = pd.read_csv(split_dir / "deployment_test.csv")
        train_ids = row_id_set(train)
        val_ids = row_id_set(val)
        test_ids = row_id_set(test)

        check_no_overlap(f"fold{fold_idx} train", train_ids, f"fold{fold_idx} val", val_ids)
        check_no_overlap(f"fold{fold_idx} train", train_ids, f"fold{fold_idx} test", test_ids)
        check_no_overlap(f"fold{fold_idx} val", val_ids, f"fold{fold_idx} test", test_ids)
        if test_ids != deployment_test_ids:
            raise RuntimeError("deployment_test.csv is not internally consistent by row ids.")
        if train_ids | val_ids != deployment_ids:
            raise RuntimeError(f"fold {fold_idx} train + val does not equal Deployment.")

        fold_summary.append(
            {
                "fold": fold_idx,
                "train_rows": len(train),
                "val_rows": len(val),
                "test_rows": len(test),
                "train_val_test_disjoint": True,
                "test_is_fixed_deployment_test": True,
            }
        )

    result = {
        "split_dir": str(split_dir),
        "deployment_test_rows": len(deployment_test),
        "deployment_rows": len(deployment),
        "deployment_test_disjoint": True,
        "folds": fold_summary,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
