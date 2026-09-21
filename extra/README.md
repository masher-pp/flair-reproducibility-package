# Extra: FLAIR v3.1 four-model reliability code (read-only copy)

> **This directory is not part of the v3 reproducibility workflow described in the
> repository README, and it is not required to run Steps 1–4.**
> It is a supplementary, read-only code copy retained for reference.
> Nothing here was modified from the original.

## What it contains

The model-family code for four alternative reliability models:

- Random Forest (`rf`)
- Extra Trees (`extratrees`)
- Histogram Gradient Boosting (`histgb`)
- XGBoost (`xgb`)

## Where the logic lives

- `src/reliability_model/07_hpo_ml_model_families.py` — unified HPO, five-fold grouped
  cross-validation, final fitting, and Test15 evaluation for all four model families.
- `src/reliability_model/03_train_rf_leaf1_sqrt_oob.py` — base RF training, feature
  definitions, and data-reading logic.
- `src/reliability_model/04_ablate_redundant_features.py`,
  `05_hpo_rf_random_csv.py`, `05_hpo_rf_staircase.py` — the three v3.1 scripts that
  instantiate RF directly, retained for ablation and HPO reference.
- The remaining Python files are local dependencies imported directly or indirectly by
  the above. `src/main_model/` is retained because feature generation depends on the
  main-model interface.

## Scope and limitations

This folder collects **code and environment notes only**. It does **not** include the
original data, training results, model weights, or caches. It is therefore a code copy,
**not** a self-contained runnable data package.

To execute it, the original FLAIR v3.1 directory structure and its corresponding data
and model files must already be in place.

## Provenance

Copied read-only from a v3.1 working directory. Several scripts in this copy reference
author-side absolute paths (for example `/Users/yczhao/Desktop/...`); those are recorded
as-is and are not portability guarantees.

## Relationship to the v3 package

The repository root holds the authoritative v3 Log-MLP-HPO100 workflow
(`scripts/`, `src/`, `data/`, `models/`, `results/`). The four alternative model families
in this folder belong to the separate v3.1 exploration and use a different code path.
