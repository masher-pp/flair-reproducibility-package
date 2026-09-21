# Release notes

## v3 Log-MLP-HPO100 — 2026-08-25

- Adds the deployable `mlp_log_hpo100_reliability_model_bundle.pt` checkpoint and makes it the default Reliability model for single-molecule prediction.
- Adds a strict `.pt` loader plus portable, resumable 100-trial random-search HPO code.
- Preserves the 32 input features and property-specific `log(AE + epsilon)` target with sample-weighted L1 loss.
- Selects hyperparameters only from leakage-safe five-fold grouped validation inside Calibration85; Test15 is evaluated once after selection.
- Records all 100 trials, the selected parameters, label-free Test15 predictions, evaluated rows, final metrics, four requested-case predictions, and SHA-256 provenance.
- Removes superseded Reliability models, results, backups, and one-off batch scripts so the package contains one unambiguous v3 workflow.
- Locks all executable workflow wrappers to this workstation's validated `Pytorch` Conda environment.
- Validates relocated release archives with strict five-fold Base loading and real v3 single-molecule prediction.

The package remains a manuscript-review/reproducibility artifact. See `LICENSE` and `CITATION.cff` before redistribution or public release.
