# Neural Networks

## Overview

This folder contains the implementation and results of various neural network models used in our experiments. It includes scripts for training and evaluating models such as DGCNN, FBCNet, and LSTM on different data splits.

## Folder Structure

- `README.md`: Documentation for the Neural Networks folder.
- `ai`: Contains the main implementation and scripts for neural network models.
  - `README.md`: Documentation for the `ai` folder.
  - `all_tested_values.ipynb`: Jupyter notebook with all tested values.
  - `dataset`: Contains dataset-related scripts and files.
    - `bde_dataset.py`: Script for BDE dataset.
    - `eeg_dataset.py`: Script for EEG dataset.
    - `microvolt_dataset.py`: Script for Microvolt dataset.
  - `logger`: Logging utilities.
  - `logs.old`: Old log files.
  - `train.py`: Script to train the models.
  - `__pycache__`: Compiled Python files.
  - `constants.py`: Constants used in neural network models.
  - `log_analysis`: Log analysis scripts.
  - `logs`: Directory for log files.
  - `models`: Directory for model files.
  - `trainer`: Training utilities and scripts.
- `poetry.lock`: Poetry lock file for dependencies.
- `pyproject.toml`: Poetry configuration file.

### `previous_results_just_in_case`

Contains backup of previous results for different data splits.

- `Random2137`: Previous results for the Random2137 split.
- `Random42`: Previous results for the Random42 split.
- `SmallTest42`: Previous results for the SmallTest42 split.
- `SubjectBased42`: Previous results for the SubjectBased42 split.

### `results`

Contains the results for different data splits.

- `Random2137`: Results for the Random2137 split.
  - `neural_networks`: Contains results for DGCNN, FBCNet, and LSTM models.
- `Random42`: Results for the Random42 split.
  - `neural_networks`: Contains results for DGCNN, FBCNet, and LSTM models.
- `SmallTest42`: Results for the SmallTest42 split.
  - `neural_networks`: Contains results for DGCNN, FBCNet, and LSTM models.
- `SubjectBased42`: Results for the SubjectBased42 split.
  - `neural_networks`: Contains results for DGCNN, FBCNet, and LSTM models.

## Usage

To get started, explore the `train.py` script in the `ai` folder to understand how to run the neural network models. Use the individual model scripts (`dgcnn_final_model.py`, `fbcnet_final_model.py`, `lstm_final_model.py`) to implement and train the respective models. Analyze the results using the JSON files in the `results` folder.

## Installation

### Using Poetry

1. **Create a virtual environment**:

   ```sh
   python -m venv venv
   ```

2. **Activate the virtual environment**:

   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

3. **Install Poetry**:

   ```sh
   pip install poetry
   ```

4. **Install all dependencies**:

   ```sh
   poetry install
   ```

5. **Add any dependencies**:
   ```sh
   poetry add <name>
   ```

### Using `requirements.txt`

1. **Create a virtual environment**:

   ```sh
   python -m venv venv
   ```

2. **Activate the virtual environment**:

   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## `requirements.txt`

```txt
torcheeg==1.1.2
mne==1.8.0
ipykernel==6.29.5
pandas==2.2.2
tensorboard==2.17.1
frozendict==2.4.6
torch==2.5.1+cu124
torchvision==0.20.1+cu124
torchaudio==2.5.1+cu124
```
