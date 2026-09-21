# Log-MLP Reliability HPO100

This directory records the 100-candidate random search completed on 2026-08-24.

- Target: property-specific `log(AE + epsilon)` with sample-weighted L1 loss.
- Features: the unchanged 32-feature solvent-augmented Reliability representation.
- Validation: five-fold canonical solute-solvent GroupKFold inside Calibration85.
- Leakage control: scaffold-history features and training weights are rebuilt from each CV training fold only.
- Selection: highest sample-count-weighted pooled out-of-fold Spearman across Abs, Emi, PLQY, and em.
- Test15: excluded from HPO selection and opened once after the winner and label-free predictions were frozen.

The selected trial is 65. Its weighted validation Spearman is 0.6310563285 versus 0.6105847312 for the original Log-MLP configuration. The final Test15 weighted Spearman is 0.6589980233 versus 0.6455226529 for the original Log-MLP; weighted AE-MAE is 10.0468116360 versus 9.8864712514.

Files:

- `trials.csv`: all 100 unique random candidates and validation metrics.
- `best_params.json`: deterministic winner and baseline comparison.
- `baseline_cv.json`: original Log-MLP under the same grouped-CV protocol.
- `baseline_log_mlp_test15_metrics.csv`: frozen pre-HPO Test15 baseline used only for final comparison.
- `../../final/mlp_log_hpo100_20260824/`: label-free predictions, evaluated rows, final metrics, and validation manifest.
- `../../../models/reliability_model/mlp_log_hpo100_reliability_model_bundle.pt`: deployable checkpoint.
- `../../../src/reliability_model/06_hpo_log_mlp_random.py`: portable, resumable HPO implementation.
- `../../../src/reliability_model/log_mlp_reliability.py`: strict checkpoint loader and inference implementation.
