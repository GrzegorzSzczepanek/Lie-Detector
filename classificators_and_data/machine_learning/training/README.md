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
