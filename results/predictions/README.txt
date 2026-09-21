FLAIR predictions - contents and usage notes
============================================

Generated with the FLAIR reproducibility package (v3 Log-MLP-HPO100) on 2026-09-20.
No file inside the original package was modified; this directory holds new outputs only.


FILES
-----

solute1_solute2_x_6solvents_predictions.csv   (12 rows x 37 cols)
    The main deliverable. Two aza-BODIPY solutes x six solvents.
    pred_abs / pred_emi (nm), pred_plqy (fraction), pred_em (log10 molar
    extinction coefficient) are the mean of the five fold-specific main models.
    pre_AE_* is the reliability model's predicted absolute error.
    fold1_*..fold5_* are the individual fold predictions; fold_std_* is their
    population standard deviation.

nn_domain_distance.csv                        (2 rows x 7 cols)
    How far the queries sit from the training domain, measured the same way the
    reliability model measures it. Written by domain_distance_fixed.py.

nn_top5_neighbours.csv                        (10 rows x 4 cols)
    Five most similar training solutes per query, by Morgan/Tanimoto.

nn_measured_comparison.csv                    (24 rows x 13 cols)
    Predicted vs measured values of the nearest neighbours, per solvent.

scaffold_reference_coverage.csv               (6 rows x 6 cols)
    How many same-scaffold training compounds have measured data in each solvent.

scaffold_reference_comparison.csv             (48 rows x 11 cols)
    Prediction vs the measured distribution of the same-scaffold reference set.


IMPORTANT - READING THESE FILES
-------------------------------

Use a correctly-rounded float parser. pandas' default reader is NOT correctly
rounded and shifts roughly one third of the pre_AE cells by 1 ULP, which looks
like a data error but is not:

    pd.read_csv(path, float_precision="round_trip")     # exact
    pd.read_csv(path)                                   # may be 1 ULP off

The text in the files is exact: float(text) reproduces the computed value
bit-for-bit. Verified across 32 sampled cells (32/32 exact via float() and via
float_precision="round_trip"; only 22/32 via the default parser).


INTERPRETATION CAVEATS
----------------------

1. pre_AE is a model-derived error RISK SCORE, not a confidence interval and not
   an error bound. The reliability model's weighted Test15 Spearman is 0.659.

2. Solvent ranking is NOT resolvable. The total solvent-induced spread in abs is
   10.4 nm while pre_AE is 5.7-8.2 nm; for emi the error (6.1-10.8 nm) exceeds the
   entire 8.0 nm spread. Only the extreme contrast (DCM/CHCl3 vs EtOH) is
   separable; adjacent gaps of 1.0-3.3 nm are not.

3. The two solutes are regioisomers, not different chromophores. Both carry one
   4-hydroxyphenyl plus three phenyls (C32H22BF2N3O, identical Murcko scaffold).
   They differ only in which aryl ring holds the OH. Their predicted difference
   (0.78 nm in abs) is far below pre_AE and should not be interpreted.

4. These are interpolation, not extrapolation. Max Tanimoto to a training solute
   is 0.894 and the scaffold is present in training (12 distinct solutes), so the
   queries sit inside the model's domain.

5. External support is uneven. Chloroform has 7 same-scaffold reference compounds
   and the prediction falls inside the measured range with ~0 nm bias. DCM (n=2)
   and acetonitrile (n=1) have too few references to judge, and one DCM reference
   is an atypical aldehyde analogue. Ethyl acetate, ethanol and methanol have NO
   same-scaffold reference data at all and cannot be checked.


ENVIRONMENT
-----------

The package hardcodes /opt/anaconda3/envs/Pytorch/bin/python3.10, which does not
exist on this machine. The equivalent conda environment "FLAIR" was used instead
(Python 3.10.15, torch 2.10.0, PyG 2.7.0, sklearn 1.7.2, pandas 2.3.3,
numpy 1.24.4 - an exact match to the versions documented in the package README).

The batch table was verified bit-for-bit against the package's official
single-pair script for four solute/solvent pairs (32/32 cells exact).
