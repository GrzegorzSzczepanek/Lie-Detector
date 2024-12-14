# Documentation

## Folder Overview

This folder contains the results of various experiments and evaluations. The contents primarily include:

- **Splits**: Information about data splits used in the experiments.
- **Cross-Validation (CV)**: Details and results of cross-validation procedures.
- **Times**: Timing information for different stages of the experiments.
- **Accuracies**: Accuracy metrics for the models evaluated.

### Folder Structure

- **feature_set_approach**: Contains results from different classifiers using a feature extraction approach.
  - `knn_60%`
  - `random_forest_over_70%_acc`
  - `svc_60%`
- **standardized_data**: Contains results from different classifiers using standardized data.
  - `knn`
  - `logistic_regression`
  - `random_forest`
  - `svc`
- **treshold_based_classification_results**: Contains results from threshold-based classification experiments on response time statistic.
  - `roc_auc_results_1_20240821_230314.json`
  - `threshold_classification_results_1.json`
  - `threshold_classification_results_1_20240821_230152.json`
  - `threshold_classification_results_1_20240821_230314.json`

### Notable Results

The most significant results in this folder are from experiments involving Random Forests using a feature extraction approach. These results are highlighted due to their importance and relevance to the overall analysis.
