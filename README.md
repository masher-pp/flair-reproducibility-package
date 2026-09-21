# FLAIR reproducibility package

Version 3 Log-MLP-HPO100 (2026-08-25).

This package contains the data splits, source code, pretrained weights, trained five-fold main models, property-specific reliability model, and result tables associated with the FLAIR manuscript.

## Scientific scope

The main model predicts absorption wavelength (`abs`), emission wavelength (`emi`), photoluminescence quantum yield (`plqy`), and log10 molar extinction coefficient (`em`). The reliability model predicts the absolute error (AE) expected for each main-model prediction.

The reported Test15 set is a fixed held-out split of the merged evaluation pool. It is not a completely independent external test set. File names therefore use `holdout` rather than `external`.

The only deployable reliability model is `models/reliability_model/mlp_log_hpo100_reliability_model_bundle.pt`. It uses the 32-feature solvent-augmented representation and four property-specific MLPs trained with sample-weighted L1 loss in `log(AE + epsilon)` space. One hundred shared hyperparameter candidates were evaluated by leakage-safe five-fold canonical solute-solvent GroupKFold inside Calibration85; Test15 was excluded from selection. Trial 65 was selected with weighted validation Spearman 0.631056. The frozen Test15 weighted Spearman is 0.658998 and weighted AE-MAE is 10.046812.

## Directory layout

- `scripts/`: ordered workflow entry points.
- `src/main_model/`: main neural-network model implementation.
- `src/reliability_model/`: feature extraction, HPO, reliability-model training, and inference.
- `src/data_splitting/`: deterministic held-out and five-fold split generation and validation.
- `data/source/`: source datasets used by the workflow.
- `data/splits/deployment/`: the runtime deployment table with fixed `cv_fold` assignments plus its fixed test table.
- `data/splits/development/`: alternative development partitions retained for analysis; the ordered runtime workflow does not read them.
- `data/templates/`: prediction input template.
- `environment/`: Conda and pip dependency installation commands.
- `models/pretrained/`: pretrained MORE encoder weights used by FLAIR.
- `models/main_model/`: five trained fold checkpoints and their training records.
- `models/reliability_model/`: deployable reliability models and deployment cache.
- `results/final/mlp_log_hpo100_20260824/`: final Test15 metrics, predictions, audit rows, and validation record.
- `results/hpo/mlp_log_hpo100_20260824/`: the complete 100-candidate grouped-CV search.
- `results/intermediate/`: reproducibility intermediates generated during reliability feature extraction.
- `extra/`: supplementary read-only code copies that are **not** part of the ordered
  v3 workflow and are not needed to run Steps 1–4. See `extra/README.md`.

## Large files

Model weights, cached objects, and the single oversized calibration CSV are stored with
**Git LFS**. Clone with Git LFS installed to obtain the real files:

```bash
git lfs install
git clone https://github.com/masher-pp/flair-reproducibility-package.git
git lfs pull
```

Without Git LFS, those paths check out as small text pointer files instead of the
actual binary content, and the workflow cannot run.

## Environment

All workflow wrappers are locked to this workstation's `Pytorch` Conda environment at `/opt/anaconda3/envs/Pytorch/bin/python3.10`. The validated environment uses Python 3.10, PyTorch 2.10.0 CPU, PyTorch Geometric 2.7.0, torch-scatter 2.1.2, scikit-learn 1.7.2, pandas 2.3.3, and NumPy 1.24.4. See `environment/pytorch_environment.txt` for the validation command.

## Workflow

Run commands from the package root:

```bash
python scripts/01_train_main_model.py --check
python scripts/02_extract_reliability_features.py --check
python scripts/03_train_reliability_model.py --check
python scripts/04_predict_single.py --check
python src/reliability_model/06_hpo_log_mlp_random.py --check
```

Remove `--check` to execute a stage. A full main-model retraining is computationally intensive. Step 3 reproduces the current Log-MLP model and its 100-candidate search; the search is resumable and writes each completed trial to `results/hpo/mlp_log_hpo100_20260824/trials.csv`.

Validate the supplied fixed data splits with:

```bash
python src/data_splitting/check_generated_splits.py
```

Example prediction:

```bash
python scripts/04_predict_single.py --smiles "CCO" --solvent "O"
```

## Reproducibility notes

- Random seeds and model hyperparameters are stored in the scripts, checkpoints, HPO summaries, and final model bundle.
- The Log-MLP HPO results and exact trial table are stored in `results/hpo/mlp_log_hpo100_20260824/`.
- The five main-model folds are read from `data/splits/deployment/deployment.csv`: Fold *k* uses `cv_fold != k` for training and `cv_fold == k` for validation.
- Paths stored in distributed CSV, JSON, and model metadata are relative to this package.
- `scripts/05_build_release_package.py` generates `MANIFEST.sha256` for all distributed files except the manifest itself.

## Before public release

The authors must replace the placeholder citation authorship in `CITATION.cff` and select an explicit code/data license in `LICENSE`. These legal and bibliographic choices cannot be inferred from the engineering files.
