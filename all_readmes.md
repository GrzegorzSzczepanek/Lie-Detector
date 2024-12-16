# EDA Folder

## Overview

The EDA (Exploratory Data Analysis) folder contains Jupyter notebooks used for exploring and visualizing the EEG data. The insights gained from this analysis were primarily used for understanding the data and generating interesting plots for the publication. However, the findings from this analysis were not directly used in the training models.

## Notebooks

### `eda.ipynb`

This notebook contains various plots and exploratory analyses of the EEG data. It was used to generate insights and fun facts about the data, but the information was not directly used in the training models.

#### Key Plots and Analyses

- **Response Times by Block**: Visualizes the average response times across different blocks.
- **Response Times by Gender**: Compares the average response times between different genders.
- **Distribution of Response Times**: Shows the distribution of response times across all participants.
- **Participant Consistency**: Analyzes the consistency of response times for each participant.
- **Age-Related Differences**: Examines the relationship between age and response times.
- **Incorrect Answers by Block**: Visualizes the count of incorrect answers across different blocks.

### `plots_for_paper.ipynb`

This notebook contains plots and analyses that were used in the publication. It highlights interesting findings and fun facts about the data and the examined participants.

#### Key Plots and Analyses

- **Average Response Times by Block**: Shows the average response times for each block.
- **ERP Plots**: Visualizes the Event-Related Potentials (ERPs) for different conditions.
- **Response Time Distribution**: Shows the distribution of response times.
- **Participant Consistency by Gender**: Analyzes the consistency of response times for each gender.

## Example Plots

### Average Response Times by Block

![Average Response Times by Block](plots_images/average_response_times_per_block.png)

### Average Response Times by Gender

![Average Response Times by Gender](plots_images/average_response_time_by_sex.png)

### ERP Plot

![ERP Plots](plots_images/p300_plot_mean.png)

### Response Time Distribution

![Response Time Distribution](plots_images/respose_time_distribution.png)

### Participant Consistency by Gender

![Participant Consistency by Gender](plots_images/response_time_consistency_per_sex.png)

## Conclusion

The EDA folder provides valuable insights and visualizations that help in understanding the EEG data. While the findings were not directly used in the training models, they offer interesting perspectives and fun facts about the data and the participants.
# Training folder

The `/training` folder in your project is organized to facilitate different aspects of training machine learning models. Here's a breakdown of its structure:

- **`/training/feature_set_search`**: Contains Jupyter notebooks for experimenting with different feature sets using various classifiers.
  - `knn.ipynb`
  - `logistic_regression_features_approach.ipynb`
  - `random_forest_eeg_features_approach.ipynb`
  - `svc_eeg_features_approach.ipynb`
- **`/training/preprocessing_params_search`**: Contains notebooks and scripts for searching optimal preprocessing parameters.
  - `random_forest_preprocessing_params_search.ipynb`
  - `param_configs.py`
  - `results/`
- **`/training/raw_data_search`**: Contains notebooks for experimenting with raw data using different versions of random forest models. Minor changes in grid search functions.
  - `random_forest_v1.ipynb`
  - `random_forest_v2.ipynb`
- **`/training/results`**: Stores JSON files with configurations and results from various random forest model experiments.
  - `random_forest_config_0.json` to `random_forest_config_48.json`

This structure helps in organizing different experiments and configurations related to training machine learning models.

## Feature Extraction

Since hyperparameter tuning on scaled data produced unsatisfactory results, with accuracy ranging between 50% and 60%, it was evident that KNN, SVM, and Random Forest models struggled to capture relationships between the inputs, in the form of EEG data ($X$), and the outputs, which were labels ($y$) indicating whether the response was truthful or deceptive. To improve model performance, a new feature set was engineered comprising statistical features (mean, standard deviation), time-domain features (kurtosis and skewness), and extracted brainwave powers (delta, theta, alpha, beta, gamma). The data was structured as a 3D array with dimensions corresponding to the number of samples, channels, and time points (frequency). The preprocessing algorithm consisted of the following steps:

1. Create an empty feature set.
2. Extract the number of samples and channels from $X$.
3. Loop over each sample:
   - Loop over each channel within the sample:
     - Extract the data for the channel.
     - Calculate and add to the feature set: mean, standard deviation, and variance (calculated using NumPy), as well as skewness and kurtosis (calculated using SciPy).
     - Compute brainwave band powers by calculating the power spectrum using Welch’s method (SciPy) and integrate over frequency bands (delta, theta, alpha, beta, gamma) using NumPy’s trapezoidal rule.
   - Append the extracted features for all channels to the feature set.

After processing all samples and channels, the resulting feature set was used to train machine learning models.

## Preprocessing Parameter Search

To enhance model performance, we systematically experimented with various preprocessing configurations, focusing on parameters such as low-frequency cutoff (lfreq), high-frequency cutoff (hfreq), notch filter frequencies (notch filter), baseline correction (baseline), and time windows (tmin, tmax). These parameters, sourced from the MNE Python library, were adjusted to isolate neural activity effectively while reducing artifacts and noise. The preprocessing pipeline was optimized to maximize model fitting and predictive accuracy.

To achieve this, we conducted an extensive search for the best preprocessing parameter configuration. After evaluating multiple combinations, the optimal setup for the Random Forest classifier was determined. We used a low-frequency cutoff of 0.3 Hz and a high-frequency cutoff of 70 Hz. A notch filter was applied at 60 Hz, while the absence of baseline correction simplified data processing. The time window for analysis was set from 0 to 0.6 seconds post-stimulus onset.

### Results on Feature-Extracted Dataset

| Model         | Best Parameters                                                                             | Test Score |
| ------------- | ------------------------------------------------------------------------------------------- | ---------- |
| Random Forest | • Bootstrap: False<br>• Class Weight: Balanced<br>• Max Features: sqrt<br>• Estimators: 300 | 0.759      |
| KNN           | • Metric: Manhattan<br>• Neighbors: 25<br>• Weights: Distance                               | 0.558      |
| SVM           | • Kernel: Linear<br>• C: 10<br>• Class Weight: Balanced                                     | 0.555      |

### Results on Preprocessing Parameter Search for Various Splits

To further evaluate the performance of our preprocessing parameter configurations, we tested the Random Forest model across different data splits. Results were obtained in preprocessing parameter search on the ebst model from feature extraction Random `Forest Classifier` that on was trained on data that was preprocessed using `MNE` default preprocessing parameters. The results are summarized below:

| Split Strategy | Accuracy | F1-Score | Preprocessing Parameters                               |
| -------------- | -------- | -------- | ------------------------------------------------------ |
| Random42       | 0.880    | 0.880    | lfreq: 0.3, hfreq: 70, notch: [60], tmin: 0, tmax: 0.6 |
| Random2137     | 0.849    | 0.841    | lfreq: 0.3, hfreq: 70, notch: [60], tmin: 0, tmax: 0.6 |
| SmallTest42    | 0.843    | 0.842    | lfreq: 0.3, hfreq: 70, notch: [60], tmin: 0, tmax: 0.6 |
| SubjectBased42 | 0.492    | 0.515    | lfreq: 0.3, hfreq: 70, notch: [60], tmin: 0, tmax: 0.6 |

The confusion matrices for the best-performing Random Forest model across different splits are shown in Fig. 19. While non-subject-based splits tend to result in a more balanced distribution of predicted labels, subject-based splits clearly demonstrate a tendency to overpredict label 1. This leads to a higher number of false positives relative to true positives. This bias may be attributed to the limited diversity of subjects in the training data, restricting the model’s ability to generalize to unseen individuals.
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
# This document provides an overview of the implementation of the Independent Component Analysis (ICA) algorithm

# and its use cases within the MNE (Magnetoencephalography and Electroencephalography) framework.

# Note that this implementation was not utilized in the final version of the project.

implementation of ICA algorithm and usecases for MNE vesrion of it. It didn't found it's use in final.
# Random Forest

## Overview

This folder contains the implementation and results of the Random Forest model used in our experiments. It includes scripts for training and evaluating the model, as well as the best results obtained from our hyperparameter and preprocessing parameter searches.

## Folder Structure

- `README.md`: Documentation for the Random Forest folder.
- `RandomForestModel.py`: Implementation of the Random Forest model.
- `best_random_forest_conf_matrix.png`: Confusion matrix of the best Random Forest model.
- `final_runs.ipynb`: Jupyter notebook for running final Random Forest experiments.
- `result_analysis.ipynb`: Jupyter notebook for analyzing the results.

### `best_results`

Contains the best results for different data splits.

- `Random2137`: Results for the Random2137 split.
- `Random42`: Results for the Random42 split.
- `SmallTest42`: Results for the SmallTest42 split.
- `SubjectBased42`: Results for the SubjectBased42 split.

## Usage

To get started, explore the `final_runs.ipynb` notebook to understand the final experiments conducted with the Random Forest model. Use the `RandomForestModel.py` script to implement and train the Random Forest model. Analyze the results using the `result_analysis.ipynb` notebook.

## Confusion Matrix

Below is the confusion matrix for the best Random Forest model:

![Best Random Forest Confusion Matrix](best_random_forest_conf_matrix.png)

The confusion matrices for the best-performing Random Forest model across different splits are shown in the figure above. Non-subject-based splits generally result in a more balanced distribution of predicted labels. In contrast, subject-based splits tend to overpredict label 1, leading to a higher number of false positives compared to true positives. This bias is likely due to the limited diversity of subjects in the training data, which restricts the model's ability to generalize to new, unseen individuals.
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
# Neural Networks

## Overview

This folder contains the implementation and results of various neural network models used in our experiments. It includes scripts for training and evaluating models such as DGCNN, FBCNet, and LSTM on different data splits.

## Folder Structure

- `README.md`: Documentation for the Neural Networks folder.
- `constants.py`: Constants used in neural network models.
- `dgcnn_final_model.py`: Final model script for DGCNN.
- `fbcnet_final_model.py`: Final model script for FBCNet.
- `lstm_final_model.py`: Final model script for LSTM.
- `neural_networks_run.py`: Script to run neural network models.
- `typings.py`: Type definitions.

### `dataset`

Contains dataset-related scripts and files.

- `bde_dataset.py`: Script for BDE dataset.
- `eeg_dataset.py`: Script for EEG dataset.
- `microvolt_dataset.py`: Script for Microvolt dataset.

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

To get started, explore the `neural_networks_run.py` script to understand how to run the neural network models. Use the individual model scripts (`dgcnn_final_model.py`, `fbcnet_final_model.py`, `lstm_final_model.py`) to implement and train the respective models. Analyze the results using the JSON files in the `results` folder.

## Confusion Matrices

Below are the confusion matrices for the best-performing neural network models across different splits:

### DGCNN Model

#### Random2137 Split

![DGCNN Random2137 Confusion Matrix](confusion_matrixes/DGCNN_Random2137.png)

#### Random42 Split

![DGCNN Random42 Confusion Matrix](confusion_matrixes/DGCNN_Random42.png)

#### SmallTest42 Split

![DGCNN SmallTest42 Confusion Matrix](confusion_matrixes/DGCNN_SmallTest42.png)

#### SubjectBased42 Split

![DGCNN SubjectBased42 Confusion Matrix](confusion_matrixes/DGCNN_SubjectBased42.png)

### FBCNet Model

#### Random2137 Split

![FBCNet Random2137 Confusion Matrix](confusion_matrixes/FBCNet_Random2137.png)

#### Random42 Split

![FBCNet Random42 Confusion Matrix](confusion_matrixes/FBCNet_Random42.png)

#### SmallTest42 Split

![FBCNet SmallTest42 Confusion Matrix](confusion_matrixes/FBCNet_SmallTest42.png)

#### SubjectBased42 Split

![FBCNet SubjectBased42 Confusion Matrix](confusion_matrixes/FBCNet_SubjectBased42.png)

### LSTM Model

#### Random2137 Split

![LSTM Random2137 Confusion Matrix](confusion_matrixes/LSTM_Random2137.png)

#### Random42 Split

![LSTM Random42 Confusion Matrix](confusion_matrixes/LSTM_Random42.png)

#### SmallTest42 Split

![LSTM SmallTest42 Confusion Matrix](confusion_matrixes/LSTM_SmallTest42.png)

#### SubjectBased42 Split

![LSTM SubjectBased42 Confusion Matrix](confusion_matrixes/LSTM_SubjectBased42.png)
# Data Extractor

## Overview

The `data_extractor.py` script is designed to load and preprocess EEG data from multiple subjects. It extracts relevant features and labels from the raw EEG data, making it ready for further analysis and model training.

## Functions

### `_desired_answer(record)`

Determines the desired answer based on the data type and block number.

### `_extract_eeg_epoch(eeg, event, baseline, tmin, tmax)`

Extracts an EEG epoch from the raw data using MNE's `Epochs` function.

### `_load_one_subject(dir_path, lfreq, hfreq, notch_filter, baseline, tmin, tmax)`

Loads and preprocesses EEG data for a single subject. It applies filtering, notch filtering, and extracts events and annotations to create records.

### `load_df(root_path, lfreq=1, hfreq=50, notch_filter=[50, 100], baseline=(0,0), tmin=0, tmax=1)`

Loads and preprocesses EEG data for all subjects in the specified root directory. It returns a DataFrame containing the extracted records.

### `extract_X_y_from_df(df)`

Extracts features (X) and labels (y) from the DataFrame. The features are the EEG data, and the labels are the target values for model training.

## Data Format

After extracting the data using the `load_df` function, the resulting DataFrame will have the following structure:

| subject  | block_no | duration | field      | data_type | answer | eeg     | desired_answer                          | label |
| -------- | -------- | -------- | ---------- | --------- | ------ | ------- | --------------------------------------- | ----- | --- |
| 1299BF1A | 1        | 0.840    | BIRTH_DATE | REAL      | YES    | <Epochs | 1 events (good & bad), 0 – 1 s (base... | YES   | 1   |
| 1299BF1A | 1        | 0.744    | HOMETOWN   | REAL      | YES    | <Epochs | 1 events (good & bad), 0 – 1 s (base... | YES   | 1   |
| 1299BF1A | 1        | 0.676    | HOMETOWN   | REAL      | YES    | <Epochs | 1 events (good & bad), 0 – 1 s (base... | YES   | 1   |
| 1299BF1A | 1        | 0.620    | HOMETOWN   | REAL      | YES    | <Epochs | 1 events (good & bad), 0 – 1 s (base... | YES   | 1   |
| 1299BF1A | 1        | 0.652    | NAME       | REAL      | YES    | <Epochs | 1 events (good & bad), 0 – 1 s (base... | YES   | 1   |

## Example Usage

```python
from data_extractor import load_df, extract_X_y_from_df

# Load the data
df = load_df("path/to/data")

# Filter the data and add the 'label' column
df = df.query("desired_answer == answer and data_type in ['REAL', 'FAKE']")
df['label'] = df.apply(lambda x: 1 if x.block_no in [1, 3] else 0, axis=1)

# Display the first few rows of the DataFrame
print(df.head())

# Extract features and labels
X, y = extract_X_y_from_df(df)

# Display the shapes of the extracted data
print(X.shape)
print(y.shape)
```
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
Extensions allow extending the debugger without modifying the debugger code. This is implemented with explicit namespace
packages.

To implement your own extension:

1. Ensure that the root folder of your extension is in sys.path (add it to PYTHONPATH) 
2. Ensure that your module follows the directory structure below
3. The ``__init__.py`` files inside the pydevd_plugin and extension folder must contain the preamble below,
and nothing else.
Preamble: 
```python
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    import pkgutil
    __path__ = pkgutil.extend_path(__path__, __name__)
```
4. Your plugin name inside the extensions folder must start with `"pydevd_plugin"`
5. Implement one or more of the abstract base classes defined in `_pydevd_bundle.pydevd_extension_api`. This can be done
by either inheriting from them or registering with the abstract base class.

* Directory structure:
```
|--  root_directory-> must be on python path
|    |-- pydevd_plugins
|    |   |-- __init__.py -> must contain preamble
|    |   |-- extensions
|    |   |   |-- __init__.py -> must contain preamble
|    |   |   |-- pydevd_plugin_plugin_name.py
```PyZMQ's CFFI support is designed only for (Unix) systems conforming to `have_sys_un_h = True`.
Project was inspired by this [paper](https://www.researchgate.net/publication/368455020_Neural_processes_underlying_faking_and_concealing_a_personal_identity_An_electroencephalogram_study)

Done with Lie-Detector team:
# Lie Detector

## Overview

This project aims to detect lies about one’s own identity through brainwave analysis. The experiment is divided into four blocks, each with different instructions for the participant. The participant's brainwave data is recorded using an EEG headset, and the data is analyzed to determine the truthfulness of their responses.

## Experiment Process

1. **Initialization**:

   - The participant's real and fake identity data is loaded.
   - The EEG headset is connected and initialized.

2. **Experiment Blocks**:

   - The experiment consists of four blocks:
     - Honest response to true identity
     - Deceitful response to true identity
     - Honest response to fake identity
     - Deceitful response to fake identity
   - Each block has specific instructions for the participant on how to respond to different types of data.

3. **Data Annotation**:

   - During each block, the EEG data is annotated with information about the data shown and the participant's response.

4. **Cleanup**:
   - After the experiment, the participant's data is erased, and the EEG headset is disconnected.

## GUI Instructions

First 5 screens - navigate to the next/previous screen using the arrow keys.

"End of block" screen - proceed to the next block by pressing Enter.

Last screen, with the number of correct answers - exit by pressing Q.

## Poetry Instructions

1. Create a virtual environment from the terminal: `py -m venv venv` (if you are using an environment manager like conda, do it your way).

2. Activate the virtual environment: `.\venv\Scripts\activate`.

3. Install Poetry: `pip install poetry`.

4. Install all dependencies: `poetry install`.

5. If you want to add any dependencies: `poetry add <name>`, e.g., `poetry add requests`.

## Identities

We employed four types of data that the experiment participant has to memorize and answer with yes or no. We made sure data types have nothing in common (based on fields mentioned in [identity generator](#identitygenerator) section). Data types consisted of:

- **Real** (user real data)
- **Celebrity** - a well-known person that the participant knew. The answer for this is always _yes_ and it serves the purpose of balancing _yes_ and _no_ answers to have an equal amount of labels. The participant was explained to lie that the celebrity's data is theirs. It was a sort of impersonation.
- **Random** - Opposite of the celebrity. The participant was supposed to always respond that **Random** data are not theirs.
- **Fake** - after the second block we do

## Experiment Blocks

The experiment is separated into four blocks:

- Honest response to true identity (yes answer for **Real** and **Celebrity**, no for **Random**)
- Deceitful response to true identity (no answer for **Real** and **Random**, yes for **Celebrity**)
- Honest response to fake identity (no answer for **Fake** and for **Random**, yes to **Celebrity**)
- Deceitful response to fake identity (yes answer for **Fake** and **Celebrity**, no for **Random**)

> After the second block, the user's real data is replaced and we give them time to learn the fake ID.

<!-- ## Screenshots -->

<!-- ![Screenshot 1](path/to/screenshot1.png)
![Screenshot 2](path/to/screenshot2.png) -->

## References to other modules

- [Personal Data Module README](/experiment/src/personal_data/README.md)
- [EEG Headset Module README](/experiment/src/eeg_headset/README.md)
- [GUI Module README](/experiment/src/gui/README.md)
- [AI Module README](/classificators_and_data/README.md)
# GUI Module

## Overview

The `Gui` class is responsible for managing the graphical user interface of the Lie Detector experiment. It handles the display of instructions, experiment data, and feedback to the participant. The GUI is built using the Pygame library and includes various screens and sequences to guide the participant through the experiment.

## Components

### Initialization

The `Gui` class initializes various components required for the experiment, including experiment blocks, screen sequences, response keys, and events.

```python
from .gui import Gui

gui = Gui()
```

### Key Methods

#### `start()`

Starts the GUI and initializes the Pygame display. It also sets up the personal data manager and begins the main loop.

```python
gui.start()
```

#### `_mainloop()`

The main loop of the GUI, which handles events and updates the display.

```python
def _mainloop(self) -> None:
    while self._running:
        self._handle_events()
        self._clock.tick(60)
        pygame.display.update()

    self._personal_data_manager.cleanup()
```

#### `_handle_events()`

Handles Pygame events, such as key presses and custom events.

```python
def _handle_events(self) -> None:
    for event in pygame.event.get():
        match event.type:
            case pygame.QUIT:
                self._running = False
            case pygame.KEYDOWN if event.key == QUIT_KEY and self._can_quit:
                self._running = False
            case pygame.KEYDOWN:
                self._handle_keydown(event.key)
            case self._go_to_next_part_event.type:
                self._go_to_next_part()
            case self._run_block_event.type:
                self._run_block()
            case self._timeout_event.type:
                self._register_response(ParticipantResponse.TIMEOUT)
```

#### `_handle_keydown(key)`

Handles keydown events based on the current state of the GUI.

```python
def _handle_keydown(self, key: int) -> None:
    if not self._main_screens_sequence.is_at_marked() and key in (GO_BACK_KEY, GO_FORWARD_KEY):
        self._handle_screen_change(key)
    elif self._experiment_blocks_sequence.get_current().is_practice and not self._do_handle_participant_response and key == CONFIRMATION_KEY:
        self._go_to_next_part()
    elif self._main_screens_sequence.is_at_marked() and self._experiment_block_parts_sequence.is_at_marked() and self._do_handle_participant_response and key in self._response_keys:
        self._handle_participant_response(key)
    elif self._wait_for_confirmation and key == CONFIRMATION_KEY:
        self._go_to_next_part()
```

#### `_draw_fixation_cross()`

Draws a fixation cross on the screen.

```python
def _draw_fixation_cross(self) -> None:
    screen_width, screen_height = self._get_screen_size()
    fixation_cross_length = int(FIXATION_CROSS_LENGTH_AS_WIDTH_PERCENTAGE * screen_width)
    fixation_cross_width = int(FIXATION_CROSS_WIDTH_AS_WIDTH_PERCENTAGE * screen_width)

    horizontal_x_coord = self._calculate_margin(fixation_cross_length, screen_width)
    horizontal_y_coord = self._calculate_margin(fixation_cross_width, screen_height)
    horizontal_rect = pygame.Rect(horizontal_x_coord, horizontal_y_coord, fixation_cross_length, fixation_cross_width)

    vertical_x_coord = self._calculate_margin(fixation_cross_width, screen_width)
    vertical_y_coord = self._calculate_margin(fixation_cross_length, screen_height)
    vertical_rect = pygame.Rect(vertical_x_coord, vertical_y_coord, fixation_cross_width, fixation_cross_length)

    self._draw_background()
    pygame.draw.rect(self._main_surface, FIXATION_CROSS_COLOR, horizontal_rect)
    pygame.draw.rect(self._main_surface, FIXATION_CROSS_COLOR, vertical_rect)
    self._set_timeout(self._go_to_next_part_event, self._get_random_from_range(FIXATION_CROSS_TIME_RANGE_MILLIS))
```

#### `_draw_experiment_data()`

Draws the experiment data on the screen.

```python
def _draw_experiment_data(self) -> None:
    self._data_to_show = self._personal_data_manager.get_next()
    self._draw_data(self._data_to_show)
    self._set_timeout(self._timeout_event, TIMEOUT_MILLIS)
```

#### `_draw_break_between_practice_and_proper()`

Draws the break screen between practice and proper trials.

```python
def _draw_break_between_practice_and_proper(self) -> None:
    self._draw_data(BREAK_BETWEEN_PRACTICE_AND_PROPER_TEXTS)
    self._set_timeout(self._run_block_event, BREAK_BETWEEN_PRACTICE_AND_PROPER_MILLIS)
```

#### `_draw_break_between_blocks()`

Draws the break screen between experiment blocks.

```python
def _draw_break_between_blocks(self) -> None:
    next_block = self._experiment_blocks_sequence.get_current()
    break_texts = [
        BLOCK_END_TEXT,
        "",
        BREAK_BETWEEN_BLOCKS_TEXT,
        *EXPERIMENT_BLOCK_TRANSLATIONS[next_block.block],
        "",
        GO_FORWARD_TEXT,
    ]
    self._draw_data(break_texts)
    self._wait_for_confirmation = True
```

#### `_draw_results()`

Draws the results screen at the end of the experiment.

```python
def _draw_results(self) -> None:
    self._can_quit = True
    results = self._personal_data_manager.get_response_counts()
    self._draw_data([RESULTS_TEXT, f"{results.correct} / {results.correct + results.incorrect}"])
```

### Helper Methods

#### `_calculate_margin(element_size, screen_size)`

Calculates the margin for centering an element on the screen.

```python
def _calculate_margin(self, element_size: int, screen_size: int) -> int:
    return (screen_size - element_size) // 2
```

#### `_get_screen_size()`

Returns the size of the screen.

```python
def _get_screen_size(self) -> Size:
    display_info = pygame.display.Info()
    return Size(display_info.current_w, display_info.current_h)
```

#### `_get_random_from_range(range)`

Returns a random integer from the specified range.

```python
def _get_random_from_range(self, range: tuple[int, int]) -> int:
    start, end = range
    return random.randrange(start, end + 1)
```

#### `_set_timeout(event, millis)`

Sets a timeout for a Pygame event.

```python
def _set_timeout(self, event: pygame.event.Event, millis: int) -> None:
    pygame.time.set_timer(event, millis, 1)
```

#### `_clear_timeout(event)`

Clears a timeout for a Pygame event.

```python
def _clear_timeout(self, event: pygame.event.Event) -> None:
    pygame.time.set_timer(event, 0)
```

#### `_sort_by_personal_data_field(personal_data)`

Sorts personal data by the field.

```python
def _sort_by_personal_data_field(self, personal_data: dict[PersonalDataField, str]) -> list[str]:
    personal_data_field_sequence = list(PersonalDataField)
    return [
        data
        for _, data in sorted(
            personal_data.items(),
            key=lambda k: personal_data_field_sequence.index(k[0]),
        )
    ]
```

## Configuration

The configuration for the GUI is defined in `config.py`. Key configuration parameters include:

- **Screen Parameters**:
  - `SCREEN_PARAMS`: Screen size and mode (fullscreen or windowed).
- **Timing**:
  - `FIXATION_CROSS_TIME_RANGE_MILLIS`: Time range for displaying the fixation cross.
  - `TIMEOUT_MILLIS`: Timeout duration for participant responses.
  - `BREAK_BETWEEN_TRIALS_TIME_RANGE_MILLIS`: Time range for breaks between trials.
  - `BREAK_BETWEEN_PRACTICE_AND_PROPER_MILLIS`: Duration of the break between practice and proper trials.
- **Colors**:
  - `BACKGROUND_COLOR_PRIMARY`: Primary background color.
  - `BACKGROUND_COLOR_INCORRECT`: Background color for incorrect responses.
  - `FIXATION_CROSS_COLOR`: Color of the fixation cross.
  - `TEXT_COLOR`: Color of the text.
- **Text**:
  - `TEXT_FONT`: Font for the text.
  - `TEXT_FONT_SIZE_AS_WIDTH_PERCENTAGE`: Font size as a percentage of the screen width.
  - Various text strings for instructions, feedback, and results.

## Example Usage

1. **Initialization**:

   - Initialize the `Gui` class.

   ```python
   from .gui import Gui

   gui = Gui()
   ```

2. **Start the GUI**:

   - Start the GUI and begin the experiment.

   ```python
   gui.start()
   ```

3. **Handle Events**:

   - The GUI will handle events and update the display in the main loop.

   ```python
   def _mainloop(self) -> None:
       while self._running:
           self._handle_events()
           self._clock.tick(60)
           pygame.display.update()

       self._personal_data_manager.cleanup()
   ```

4. **Draw Experiment Data**:

   - Draw the experiment data on the screen.

   ```python
   def _draw_experiment_data(self) -> None:
       self._data_to_show = self._personal_data_manager.get_next()
       self._draw_data(self._data_to_show)
       self._set_timeout(self._timeout_event, TIMEOUT_MILLIS)
   ```

5. **Draw Results**:

   - Draw the results screen at the end of the experiment.

   ```python
   def _draw_results(self) -> None:
       self._can_quit = True
       results = self._personal_data_manager.get_response_counts()
       self._draw_data([RESULTS_TEXT, f"{results.correct} / {results.correct + results.incorrect}"])
   ```

## Conclusion

The `Gui` class provides a comprehensive interface for managing the graphical user interface of the Lie Detector experiment. It handles the display of instructions, experiment data, and feedback to the participant, ensuring a smooth and interactive experience.
# EEGHeadset Class

## Overview

The `EEGHeadset` class is designed to interface with the BrainAccess EEG device, manage data acquisition, and annotate and save EEG data during an experiment. This class handles the connection to the EEG headset, starts and stops data acquisition, and annotates the data with relevant information about the experiment.

## Components

### Initialization

The `EEGHeadset` class is initialized with a participant ID, which is used to create a directory for saving the EEG data.

```python
from .eeg_headset import EEGHeadset

eeg_headset = EEGHeadset(participant_id="12345")
```

### Key Methods

#### [`_connect()`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A31%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition")

Establishes a connection to the EEG headset.

```python
eeg_headset._connect()
```

#### [`_disconnect()`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A43%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition")

Disconnects from the EEG headset.

```python
eeg_headset._disconnect()
```

#### [`start_block(experiment_block)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A57%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition")

Starts a new block of the experiment and begins data acquisition.

```python
from ..common import ExperimentBlock

eeg_headset.start_block(ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY)
```

#### [`stop_and_save_block()`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A73%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition")

Stops the current block and saves the acquired EEG data to a file.

```python
eeg_headset.stop_and_save_block()
```

#### [`annotate_data_shown(shown_data_field, shown_data_type)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A95%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition")

Annotates the EEG data with information about the data shown to the participant.

```python
from ..personal_data import PersonalDataField, PersonalDataType

eeg_headset.annotate_data_shown(PersonalDataField.NAME, PersonalDataType.REAL)
```

#### [`annotate_response(participant_response)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A116%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition")

Annotates the EEG data with the participant's response.

```python
from ..common import ParticipantResponse

eeg_headset.annotate_response(ParticipantResponse.YES)
```

## Configuration

The configuration for the EEG headset is defined in [`eeg_config.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_config.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "/Users/gsk/documents/projects/Lie-Detector/experiment/src/eeg_headset/eeg_config.py"). Key configuration parameters include:

- **Channel Mapping**

:

- [`BRAINACCESS_EXTENDED_KIT_16_CHANNEL`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_config.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A2%2C%22character%22%3A0%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition"): Mapping of the 16 EEG channels.

- **Data Folder Path**:
  - [`DATA_FOLDER_PATH`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_config.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A22%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A10%2C%22character%22%3A24%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition"): Path to the folder where EEG data will be saved.
- **Used Device**:
  - [`USED_DEVICE`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_config.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A24%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A10%2C%22character%22%3A42%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition"): The EEG device configuration to be used.

## Example Usage

1. **Initialization**:

   - Initialize the [`EEGHeadset`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2F__init__.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A0%2C%22character%22%3A25%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A13%2C%22character%22%3A6%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") with a participant ID.

   ```python
   eeg_headset = EEGHeadset(participant_id="12345")
   ```

2. **Start a Block**:

   - Start a new block of the experiment and begin data acquisition.

   ```python
   from ..common import ExperimentBlock

   eeg_headset.start_block(ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY)
   ```

3. **Annotate Data**:

   - Annotate the data shown to the participant.

   ```python
   from ..personal_data import PersonalDataField, PersonalDataType

   eeg_headset.annotate_data_shown(PersonalDataField.NAME, PersonalDataType.REAL)
   ```

   - Annotate the participant's response.

   ```python
   from ..common import ParticipantResponse

   eeg_headset.annotate_response(ParticipantResponse.YES)
   ```

4. **Stop and Save Block**:

   - Stop the current block and save the acquired EEG data.

   ```python
   eeg_headset.stop_and_save_block()
   ```

5. **Disconnect**:

   - Disconnect from the EEG headset.

   ```python
   eeg_headset._disconnect()
   ```

## Detailed Description

### Initialization

The [`EEGHeadset`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2F__init__.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A0%2C%22character%22%3A25%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A13%2C%22character%22%3A6%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") class is initialized with a participant ID, which is used to create a directory for saving the EEG data. The BrainAccess library is initialized, and directories for saving data are created if they do not exist.

### Connection Management

The [`_connect`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A31%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") method establishes a connection to the EEG headset using the BrainAccess library. The [`_disconnect`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A43%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") method disconnects from the EEG headset and ensures that no block is currently being recorded.

### Data Acquisition

The [`start_block`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A57%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") method starts a new block of the experiment and begins data acquisition. The [`stop_and_save_block`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A73%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") method stops the current block, retrieves the acquired EEG data, and saves it to a file.

### Data Annotation

The [`annotate_data_shown`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A95%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") method annotates the EEG data with information about the data shown to the participant, such as the field (e.g., name, birth date) and the type of data (e.g., real, fake). The [`annotate_response`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A116%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") method annotates the EEG data with the participant's response (e.g., yes, no).

### Directory Management

The `_create_dir_if_not_exist` "Go to definition" method ensures that the specified directory exists, creating it if necessary.

## Configuration File: [`eeg_config.py`](./eeg_config.py)

```python
# eeg_config.py

BRAINACCESS_EXTENDED_KIT_16_CHANNEL = {
    0: "Fp1",
    1: "Fp2",
    2: "F3",
    3: "F4",
    4: "C3",
    5: "C4",
    6: "P3",
    7: "P4",
    8: "O1",
    9: "O2",
    10: "T3",
    11: "T4",
    12: "T5",
    13: "T6",
    14: "F7",
    15: "F8",
}

# Path to the folder where EEG data will be saved
DATA_FOLDER_PATH = "eeg_data"

# Device configuration to be used
USED_DEVICE = BRAINACCESS_EXTENDED_KIT_16_CHANNEL
```
# Personal Data Module

## Purpose and explanation

We employed for types of data that experiment participant has to memorize and answer with yes or no. We made sure data types have nothing in common (based on fileds mentioned in [identity generator](#identitygenerator) section) Data types consisted:

- `Real` - user real data
- `Celebrity` - a well known person that participant knew. Answer for this is always _yes_ and it serves purpose of balancing `yes` and `no` answers to have the amout of labels. Participant was explained to lie that celebrity's data is theirs. It was sort of impersonation.
- `Random` - Opposite of the celebrity. Participant was supposed to always respond that _Random_ data are not theirs.
- `Fake` - Fake data that has no connection to user Real identiry as every other identity we assign to user.

Experiment is separated into four blocks:

- Honest response to true identity (yes answer for real and celebrity, no for Random)
- Deceitful response to true identiry (no answer for real and random, yes for celebrity)
- Honest response to fake identity (no answer for fake and for Random, yes to Celebrity)
- Deceitful response to Fake identiry (yes answer for fake for celebrity, no for random)
  > after second block user real data is replaced ad we give them time to learn fake id.

## Overview

The Personal Data module is designed to manage and generate personal data for participants in a study aimed at detecting lies about one’s own identity through brainwave analysis. This module includes two main components: the `IdentityGenerator` and the `PersonalDataManager`.

## Components

### IdentityGenerator

The `IdentityGenerator` class is responsible for generating and managing identity data for the participant, a fake identity, a celebrity, and a random person. It loads data from YAML files and ensures that the generated identities do not overlap with the participant's real data. Data for every person (real, fake, celebrity, random) consist these fields:

- Name + Surname (as one field)
- Hometown
- Day of birth + month of date (as one field, f.e `1st of the January`)

#### Key Methods

- `get_real()`: Returns the participant's real identity data.
- `get_fake()`: Returns the generated fake identity data.
- `get_celebrity()`: Returns the selected celebrity identity data.
- `get_rando()`: Returns the generated random identity data.
- `get_id()`: Returns the participant's ID.

### PersonalDataManager

The `PersonalDataManager` class manages the flow of personal data during the experiment. It interacts with the EEG headset to annotate data and responses, and it keeps track of the participant's responses.

#### Key Methods

- `start_block(experiment_block, is_practice)`: Starts a new block of the experiment, generating data for practice or test trials.
- `stop_block()`: Stops the current block and saves EEG data if it is not a practice block.
- `has_next()`: Checks if there is more data to get.
- `get_next()`: Retrieves the next piece of personal data.
- `get_feedback_for_practice_participant_response(participant_response)`: Provides feedback on the participant's response during practice trials.
- `register_participant_response(participant_response)`: Registers the participant's response and annotates it in the EEG recording.
- `get_response_counts()`: Returns the counts of correct and incorrect responses for the current block.
- `cleanup()`: Erases user data.

## How It Works

1. **Initialization**:

   - The `PersonalDataManager` initializes the `IdentityGenerator`, which loads and validates the participant's real data, fake data, celebrity data, and random data from YAML files.
   - The `EEGHeadset` is initialized and connected to the participant's EEG device.

2. **Experiment Blocks**:

   - The experiment is divided into four blocks, each with different instructions for the participant.
   - The `PersonalDataManager` starts a block by generating data for practice or test trials.
   - The GUI displays the data, and the participant responds with 'Yes' or 'No'.
   - The `PersonalDataManager` checks the participant's response and provides feedback during practice trials.
   - The `EEGHeadset` annotates the data shown and the participant's response.

3. **Data Annotation**:

   - During each block, the `EEGHeadset` annotates the type of data shown (e.g., real, fake, celebrity, random) and the participant's response (e.g., 'Yes', 'No', 'Timeout').

4. **Cleanup**:
   - After the experiment, the `PersonalDataManager` erases the participant's data and disconnects the EEG headset.

> NOTE participant data is not stored. After successful exam what's left after them is just assigned UUID and annotated EED data.

## Configuration

The configuration for the Personal Data module is defined in `config.py`. Key configuration parameters include:

- **File Paths**:

  - `CELEBRITIES_FILE_PATH`: Path to the YAML file containing celebrity data.
  - `FAKE_AND_RANDO_DATA_FILE_PATH`: Path to the YAML file containing fake and random identity data.
  - `USER_DATA_FILE_PATH`: Path to the YAML file containing user data.

- **Trial Multipliers**:

  - `TEST_TRIALS_MULTIPLIER`: Multiplier for the number of test trials for fake and real data.
  - `BIGGER_CATCH_TRIALS_MULTIPLIER`: Multiplier for the number of catch trials for celebrity or random data.
  - `SMALLER_CATCH_TRIALS_MULTIPLIER`: Multiplier for the number of catch trials for celebrity or random data.

- **Block Expected Identity**:

  - `BLOCK_EXPECTED_IDENTITY`: Dictionary mapping each `ExperimentBlock` to the expected identity types and trial multipliers.

- **YAML Template**:
  - `YAML_TEMPLATE`: Template for user data in YAML format.

### Explanation of Trial Multipliers

Multiplying trial data by these values will give us the exact amount of probes we need for each block:

- `TEST_TRIALS_MULTIPLIER = 15`: Used for fake and real data. Each probe (e.g., full name, hometown, birthdate) will be shown 15 times, resulting in 45 trials in our example.
- `BIGGER_CATCH_TRIALS_MULTIPLIER = 18`: Used for celebrity or random data to balance the number of 'Yes' and 'No' answers and to reduce the participant's ability to get used to the same data. Every probe is being shown 3 times during the experiment meaning 54 probes during bigger catch trial.
- `SMALLER_CATCH_TRIALS_MULTIPLIER = 3`: Also used for celebrity or random data for the same balancing purpose. Every probe is being shown 3 times during the experiment so that gives nine probes for smaller catch trial.

> If participant is in block where they are supposed to lie about their identity we give celebrity a bigger catch trial number and less to random identity to balance `yes` and `no` answers. For truth blocks we do opposite.

### Block Expected Identity

Depending on the block, the participant was instructed to respond as follows:

- Honest Response to True Identity :

  - `yes` to Real (user's identity)
  - `yes` to Celebrity
  - `no` to Random

- Honest Response to True Identity :

  - `no` to Real (user's identity)
  - `yes` to Celebrity
  - `no` to Random

- Honest Response to Fake Identity :

  - `no` to Fake (user's identity)
  - `yes` to Celebrity
  - `no` to Random

- Deceitful Response to Fake Identity :
  - `yes` to Fake (user's identity)
  - `yes` to Celebrity
  - `no` to Random

The `BLOCK_EXPECTED_IDENTITY` configuration maps each `ExperimentBlock` to the expected identity types and trial multipliers:

- `ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY`: (REAL, SMALL, BIG)
- `ExperimentBlock.DECEITFUL_RESPONSE_TO_TRUE_IDENTITY`: (REAL, BIG, SMALL)
- `ExperimentBlock.HONEST_RESPONSE_TO_FAKE_IDENTITY`: (FAKE, BIG, SMALL)
- `ExperimentBlock.DECEITFUL_RESPONSE_TO_FAKE_IDENTITY`: (FAKE, SMALL, BIG)
