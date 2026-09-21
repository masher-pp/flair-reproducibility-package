# Data files

`source/flair_photophysical_property_dataset.csv` is the main photophysical-property dataset. `source/ae_pre_holdout_candidate_pool.csv` is the additional candidate pool used when constructing the reliability calibration/held-out split.

`splits/deployment/deployment.csv` is the deployment dataset and contains the complete fixed five-fold assignment in `cv_fold`. For Fold *k*, rows with `cv_fold != k` are training data and rows with `cv_fold == k` are validation data. `splits/deployment/deployment_test.csv` is its fixed test dataset and is shared across all five folds.

Only these two files in `splits/deployment/` are read by the ordered runtime workflow. `splits/development/` retains alternative Random/Scaffold development partitions for analysis; it is not an input to Steps 1-4. Scaffold maps, selection tables and assignment summaries are reproducible intermediates and are intentionally not included.

The held-out data are an internal fixed holdout, not a completely independent external dataset. Definitions and units are provided in `../docs/data_dictionary.csv`.

Rows contain literature provenance fields such as `reference(doi)`. Before public release, the authors should confirm that redistribution of every source dataset and derived table is compatible with the applicable licenses and journal policy.
