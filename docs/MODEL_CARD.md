# Model card

## Main model

Five fold-specific checkpoints are stored under `models/main_model/fold_01` through `fold_05`. The deployed prediction for each property is the arithmetic mean of the five strictly loaded Base models.

## Reliability model

- Model family: `mlp_log_hpo100`
- Model file: `models/reliability_model/mlp_log_hpo100_reliability_model_bundle.pt`
- Runtime: `/opt/anaconda3/envs/Pytorch/bin/python3.10`
- Input: 32 structural, neighborhood, fold-variation, historical-error, and solvent-reference features
- Output: nonnegative predicted absolute error (`pre_AE`) for each Base ensemble prediction
- Target: property-specific `log(AE + epsilon)` with sample-weighted L1 loss
- Selection: highest sample-count-weighted pooled five-fold validation Spearman within Calibration85
- Selected trial: 65 of 100
- Weighted validation Spearman: 0.6310563284911787
- Weighted Test15 Spearman: 0.6589980233120313
- Weighted Test15 AE MAE: 10.046811636040012

Test15 was excluded from model selection. Its labels were loaded only after the selected model and label-free predictions were frozen. Property-level results are stored in `results/final/mlp_log_hpo100_20260824/test15_by_property.csv`.

## Intended use

The package supports reproduction of the FLAIR manuscript workflow and risk ranking of solute-solvent photophysical-property predictions. `pre_AE` is a model-derived predicted absolute-error score, not a calibrated uncertainty interval or guaranteed error bound.

## Limitations

The held-out set is derived from the merged evaluation pool and is not a completely independent external cohort. Applicability may degrade for novel chemical scaffolds or solvents outside the represented domain. PyTorch checkpoint files should only be loaded from trusted sources.
