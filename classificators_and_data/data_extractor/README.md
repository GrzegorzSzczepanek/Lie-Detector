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
