# Machine Learning Project

## Overview

This project contains various scripts and notebooks for preprocessing, training, and evaluating machine learning models on EEG data. Below is a brief description of the contents of each folder and file.

## Folder Structure

### Root Directory

- `README.md`: Project documentation.
- `ml_lib.py`: Functions for creating feature sets from EEG data and training models.
- `files_lib.py`: Functions for saving, opening, and reading files and results into JSON.
- `preprocessing_lib.py`: Functions for preprocessing EEG data.
- `visualisation_lib.py`: Functions for plotting results and data.

### `ica`

- `README.md`: Documentation for the `ica` folder.
- `ica_mne.ipynb`: Jupyter notebook for ICA using MNE.
- `python_fast_ica.ipynb`: Jupyter notebook for Fast ICA.
- `python_fast_ica.py`: Script for Fast ICA written from scratch.

### `results`

- `README.md`: Documentation for the `results` folder.
- `feature_set_approach`: Contains results related to feature set approaches.
- `standardized_data`: Contains standardized data results.
- `treshold_based_classification_results`: Contains response time threshold-based classification results.
- `visualization.ipynb`: Jupyter notebook for visualizing results.

### `training`

- `README.md`: Documentation for the `training` folder.
- `feature_set_search`: Contains Jupyter notebooks for experimenting with different feature sets.
  - `knn.ipynb`
  - `logistic_regression_features_approach.ipynb`
  - `random_forest_eeg_features_approach.ipynb`
  - `svc_eeg_features_approach.ipynb`
- `preprocessing_params_search`: Contains notebooks and scripts for searching optimal preprocessing parameters.
  - `random_forest_preprocessing_params_search.ipynb`
  - `param_configs.py`
  - `results/`
- `raw_data_search`: Contains notebooks for experimenting with raw data using random forest models in slightly different versions of grid search function.
  - `random_forest_v1.ipynb`
  - `random_forest_v2.ipynb`
- `results`: Stores JSON files with configurations and results from various random forest model experiments.

## Usage

To get started, explore the Jupyter notebooks in the `training` folder to understand the different experiments conducted. Use the functions in `ml_lib.py` and `preprocessing_lib.py` for feature extraction and preprocessing. Visualize the results using the scripts in `visualisation_lib.py`.

For more detailed information, refer to the individual README files in each folder.
