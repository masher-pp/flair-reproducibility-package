from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires RDKit. Run it in your Pytorch/RDKit environment.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = "../../data/source/flair_photophysical_property_dataset.csv"
DEFAULT_OUTPUT_DIR = "../../data/splits/deployment"
PROPERTIES = ["abs", "emi", "plqy", "em"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one fixed external test set from 5% small-scaffold-first scaffold rows "
            "+ 5% random rows, then make 5-fold CV splits from the remaining rows."
        )
    )
    parser.add_argument("--input_csv", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--scaffold_fraction", type=float, default=0.05)
    parser.add_argument("--random_fraction", type=float, default=0.05)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep_original_split",
        action="store_true",
        help="Keep the existing split column as original_split. The new split column is always written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split CSV/JSON files in the output directory.",
    )
    return parser.parse_args()


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True)


def bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return f"__INVALID__::{smiles}"
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
    if scaffold_smiles:
        return scaffold_smiles
    return f"__NO_SCAFFOLD__::{Chem.MolToSmiles(mol, canonical=True)}"


def target_count(n_rows: int, fraction: float) -> int:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"Fraction must be in [0, 1], got {fraction}.")
    return int(round(n_rows * fraction))


def scaffold_size_table(df: pd.DataFrame, scaffold_col: str) -> pd.DataFrame:
    return (
        df.groupby(scaffold_col, sort=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["n_rows", scaffold_col], ascending=[True, True], kind="mergesort")
        .reset_index(drop=True)
    )


def select_small_scaffold_rows(df: pd.DataFrame, scaffold_col: str, target_n: int) -> Tuple[np.ndarray, pd.DataFrame]:
    scaffold_sizes = scaffold_size_table(df, scaffold_col)
    selected_scaffolds: List[str] = []
    selected_n = 0
    for _, row in scaffold_sizes.iterrows():
        selected_scaffolds.append(str(row[scaffold_col]))
        selected_n += int(row["n_rows"])
        if selected_n >= target_n:
            break
    selected_table = scaffold_sizes[scaffold_sizes[scaffold_col].isin(selected_scaffolds)].copy()
    selected_table["cumulative_rows_small_first"] = selected_table["n_rows"].cumsum()
    return df[scaffold_col].isin(selected_scaffolds).to_numpy(), selected_table


def random_rows_from_remaining(remaining_indices: np.ndarray, target_n: int, seed: int) -> np.ndarray:
    if target_n > len(remaining_indices):
        raise ValueError(f"Cannot select {target_n} random rows from only {len(remaining_indices)} remaining rows.")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(remaining_indices, size=target_n, replace=False))


def add_cv_split_columns(dev_df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    out = dev_df.copy()
    out["cv_fold"] = -1
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (_, val_pos) in enumerate(kf.split(out), start=1):
        out.iloc[val_pos, out.columns.get_loc("cv_fold")] = fold_idx
    if (out["cv_fold"] < 1).any():
        raise RuntimeError("Failed to assign all development rows to a CV fold.")
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def resolve_relative_to_script(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (SCRIPT_DIR / path).resolve()


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path.resolve())


def expected_output_files(output_path: Path, n_folds: int) -> List[Path]:
    return [output_path / "deployment.csv", output_path / "deployment_test.csv"]


def guard_against_overwrite(output_path: Path, n_folds: int, overwrite: bool) -> None:
    existing = [path for path in expected_output_files(output_path, n_folds) if path.exists()]
    if existing and not overwrite:
        listed = "\n".join(f"  - {path}" for path in existing[:20])
        more = "" if len(existing) <= 20 else f"\n  ... and {len(existing) - 20} more"
        raise FileExistsError(
            "Split outputs already exist. Use --overwrite only if you intentionally want to regenerate them.\n"
            f"{listed}{more}"
        )


def value_counts_dict(series: pd.Series) -> Dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def property_non_null_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {prop: int(pd.to_numeric(df[prop], errors="coerce").notna().sum()) for prop in PROPERTIES if prop in df.columns}


def validate_no_overlap(parts: Dict[str, pd.DataFrame]) -> None:
    index_sets = {name: set(df["_original_row_id"].tolist()) for name, df in parts.items()}
    names = list(index_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = index_sets[left] & index_sets[right]
            if overlap:
                raise RuntimeError(f"Unexpected row overlap between {left} and {right}: {len(overlap)} rows.")


def make_splits(
    *,
    input_csv: str,
    output_dir: str,
    smiles_col: str,
    scaffold_fraction: float,
    random_fraction: float,
    n_folds: int,
    seed: int,
    keep_original_split: bool,
    overwrite: bool,
) -> Dict:
    input_path = resolve_relative_to_script(input_csv)
    output_path = resolve_relative_to_script(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    guard_against_overwrite(output_path, n_folds, overwrite)

    df = pd.read_csv(input_path)
    if smiles_col not in df.columns:
        raise ValueError(f"Missing SMILES column {smiles_col!r}. Columns: {list(df.columns)}")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    df = df.copy()
    df["_original_row_id"] = np.arange(len(df), dtype=int)
    if "split" in df.columns:
        if keep_original_split:
            df = df.rename(columns={"split": "original_split"})
        else:
            df = df.drop(columns=["split"])

    print(f"[Split] input: {input_path}")
    print(f"[Split] rows: {len(df)}")
    print("[Split] computing canonical SMILES and Bemis-Murcko scaffolds")
    unique_smiles = pd.Series(df[smiles_col].astype(str).unique(), name=smiles_col)
    smiles_map = pd.DataFrame({smiles_col: unique_smiles})
    smiles_map["_canonical_smiles"] = smiles_map[smiles_col].map(canonical_smiles)
    smiles_map["_scaffold"] = smiles_map[smiles_col].map(bemis_murcko_scaffold)
    df = df.merge(smiles_map, on=smiles_col, how="left", validate="many_to_one")

    n_total = len(df)
    scaffold_target = target_count(n_total, scaffold_fraction)
    random_target = target_count(n_total, random_fraction)

    scaffold_mask, _ = select_small_scaffold_rows(df, "_scaffold", scaffold_target)
    scaffold_test = df.loc[scaffold_mask].copy()

    remaining_after_scaffold = df.loc[~scaffold_mask].copy()
    random_indices = random_rows_from_remaining(
        remaining_after_scaffold.index.to_numpy(dtype=int),
        random_target,
        seed,
    )
    random_mask = df.index.isin(random_indices)
    random_test = df.loc[random_mask].copy()
    dev = df.loc[~scaffold_mask & ~random_mask].copy()
    external_test = pd.concat([scaffold_test, random_test], ignore_index=True)

    scaffold_test["external_test_source"] = "small_scaffold_first_5pct"
    random_test["external_test_source"] = "random_5pct_from_scaffold_remaining"
    scaffold_test["split"] = "external_test"
    random_test["split"] = "external_test"
    external_test = pd.concat([scaffold_test, random_test], ignore_index=True)
    external_test["split"] = "external_test"
    dev["split"] = "development"

    validate_no_overlap(
        {
            "external_scaffold": scaffold_test,
            "external_random": random_test,
            "development": dev,
        }
    )

    dev = add_cv_split_columns(dev, n_folds=n_folds, seed=seed)
    external_test = external_test.sort_values("_original_row_id").reset_index(drop=True)
    dev = dev.sort_values("_original_row_id").reset_index(drop=True)

    write_csv(external_test, output_path / "deployment_test.csv")
    write_csv(dev, output_path / "deployment.csv")

    fold_summaries = []
    for fold_idx in range(1, n_folds + 1):
        train = dev[dev["cv_fold"] != fold_idx].copy()
        val = dev[dev["cv_fold"] == fold_idx].copy()
        train["split"] = "train"
        val["split"] = "valid"
        external_fold_test = external_test.copy()

        deployment_path = output_path / "deployment.csv"
        test_path = output_path / "deployment_test.csv"
        fold_summaries.append(
            {
                "fold": fold_idx,
                "deployment_csv": portable_path(deployment_path),
                "train_selector": f"cv_fold != {fold_idx}",
                "val_selector": f"cv_fold == {fold_idx}",
                "test_csv": portable_path(test_path),
                "train_rows": int(len(train)),
                "val_rows": int(len(val)),
                "test_rows": int(len(external_fold_test)),
                "train_property_non_null": property_non_null_counts(train),
                "val_property_non_null": property_non_null_counts(val),
            }
        )

    summary = {
        "input_csv": portable_path(input_path),
        "output_dir": portable_path(output_path),
        "seed": int(seed),
        "n_total_rows": int(n_total),
        "scaffold_fraction_requested": float(scaffold_fraction),
        "scaffold_target_rows": int(scaffold_target),
        "scaffold_test_rows": int(len(scaffold_test)),
        "scaffold_test_unique_scaffolds": int(scaffold_test["_scaffold"].nunique()),
        "random_fraction_requested": float(random_fraction),
        "random_target_rows": int(random_target),
        "random_test_rows": int(len(random_test)),
        "deployment_test_rows": int(len(external_test)),
        "deployment_rows": int(len(dev)),
        "deployment_test_fraction_actual": float(len(external_test) / n_total),
        "deployment_fraction_actual": float(len(dev) / n_total),
        "n_folds": int(n_folds),
        "property_non_null_total": property_non_null_counts(df),
        "property_non_null_deployment_test": property_non_null_counts(external_test),
        "property_non_null_deployment": property_non_null_counts(dev),
        "deployment_test_source_counts": value_counts_dict(external_test["external_test_source"]),
        "cv_fold_counts": value_counts_dict(dev["cv_fold"]),
        "folds": fold_summaries,
        "files": {
            "deployment": portable_path(output_path / "deployment.csv"),
            "deployment_test": portable_path(output_path / "deployment_test.csv"),
        },
    }

    print("[Split] done")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    make_splits(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        smiles_col=args.smiles_col,
        scaffold_fraction=args.scaffold_fraction,
        random_fraction=args.random_fraction,
        n_folds=args.n_folds,
        seed=args.seed,
        keep_original_split=args.keep_original_split,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
