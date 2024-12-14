# Final Models

## Overview

The `final_models` folder contains scripts and notebooks for testing machine learning models on various data splits. This includes the best Random Forest model obtained after searching for the best hyperparameters and preprocessing parameters.

## Folder Structure

### Root Directory

- `ExampleLieModel.py`: An example implementation of a lie detection model.
- `ExperimentManager.py`: Manages the experiments, including training and evaluating models on different data splits.
- `LieModel.py`: Abstract base class for lie detection models.
- `config.py`: Configuration settings.
- `final_runs.ipynb`: Jupyter notebook for running final experiments.

### `neural_networks`

Contains scripts and notebooks related to neural network models.

- `constants.py`: Constants used in neural network models.
- `dataset`: Dataset-related scripts and files.
- `dgcnn_final_model.py`: Final model script for DGCNN.
- `fbcnet_final_model.py`: Final model script for FBCNet.
- `lstm_final_model.py`: Final model script for LSTM.
- `neural_networks_run.py`: Script to run neural network models.
- `previous_results_just_in_case`: Backup of previous results.
- `results`: Directory to store results.
- `typings.py`: Type definitions.

### `random_forest`

Contains scripts and notebooks related to the Random Forest model.

- `README.md`: Documentation for the Random Forest folder.
- `RandomForestModel.py`: Implementation of the Random Forest model.
- `best_random_forest_conf_matrix.png`: Confusion matrix of the best Random Forest model.
- `best_results`: Directory to store the best results.
- `final_runs.ipynb`: Jupyter notebook for running final Random Forest experiments.
- `result_analysis.ipynb`: Jupyter notebook for analyzing the results.

## Usage

To get started, explore the Jupyter notebooks in the `final_models` folder to understand the different experiments conducted. Use the `ExperimentManager.py` script to manage and run experiments on various data splits. Implement your own models by extending the `LieModel` abstract base class.

For more detailed information, refer to the individual README files in each folder.
