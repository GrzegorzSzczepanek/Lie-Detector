# [PROJECT LOGO PLACEHOLDER]

# Lie Detector EEG Experiment: Comprehensive Overview

This repository contains the complete implementation and analysis pipeline of an EEG-based lie detection experiment. Inspired by [Neural processes underlying faking and concealing a personal identity: An EEG study](https://www.researchgate.net/publication/368455020_Neural_processes_underlying_faking_and_concealing_a_personal_identity_An_electroencephalogram_study), this project aims to classify truthful versus deceptive responses to identity-related prompts using neural network models, traditional machine learning algorithms, and carefully designed preprocessing and feature extraction techniques.

Developed by the _Lie-Detector Team_, the experiment and analyses included here serve as a comprehensive exploration—from raw EEG data collection, preprocessing, and exploratory analysis, through feature engineering, model training, and evaluation.

---

## Table of Contents

- [Project Description](#project-description)
- [Project Structure](#project-structure)
  - [Experiment](#experiment)
  - [Classificators and Data](#classificators-and-data)
- [Data and Experiment Setup](#data-and-experiment-setup)
- [Preprocessing, Feature Extraction & ICA](#preprocessing-feature-extraction--ica)
- [Machine Learning & Neural Networks](#machine-learning--neural-networks)
- [EDA & Results Visualization](#eda--results-visualization)
- [References and Related Work](#references-and-related-work)
- [Placeholders for Media](#placeholders-for-media)

---

## Project Description

This project revolves around detecting deception in identity statements using EEG signals. Participants were instructed to respond “yes” or “no” to identity-related information in multiple experimental blocks, sometimes honestly and sometimes deceptively. By analyzing the resulting EEG signals, the objective was to uncover neural markers of truth-telling and deception.

**Key Highlights:**

- **Inspired Research**: Built upon previous EEG studies exploring neural correlates of deception and identity masking.
- **Robust Experimentation**: Controlled EEG experiment with well-defined blocks of trials for truthful and deceptive responses to real or fake personal identities.
- **Comprehensive Data Pipeline**: From EEG headset recordings to final classification models, including preprocessing, ICA, feature extraction, and hyperparameter optimization.
- **Multi-Model Approach**: Includes Random Forest, SVM, KNN, Logistic Regression, and advanced neural network architectures (DGCNN, FBCNet, LSTM).

---

## Project Structure

```text
.
├── README.md (You are here - Main Project Introduction)
├── experiment
│   ├── README.md
│   ├── eeg_data
│   │   └── ...EEG files per participant
│   └── src
│       ├── assets
│       ├── eeg_headset (EEG acquisition code)
│       │   └── README.md
│       ├── gui (Graphical User Interface for experiment)
│       │   └── README.md
│       └── personal_data (Identity generation and management)
│           └── README.md
└── classificators_and_data
    ├── README.md
    ├── data (Processed data folders)
    ├── data_extractor (Data loading and formatting)
    │   └── README.md
    ├── data_preprocessing (Preprocessing scripts)
    ├── final_models (Scripts and results of best models)
    │   ├── README.md
    │   ├── neural_networks
    │   │   └── README.md
    │   └── random_forest
    │       └── README.md
    ├── machine_learning (Classical ML pipelines and results)
    │   ├── README.md
    │   ├── ica (Independent Component Analysis)
    │   │   └── README.md
    │   ├── results (Evaluation metrics, confusion matrices)
    │   └── training (Hyperparameter search, feature selection)
    │       └── README.md
    └── neural_networks (Deep learning models and logs)
        ├── README.md
        └── ai (Core training scripts, dataset management)
```

**For detailed descriptions, please see the individual `README.md` files in the corresponding directories.**

---

## Data and Experiment Setup

Participants responded to identity-related prompts (their own, fake, celebrity, and random identities) under instructions to either tell the truth or lie. The **`experiment`** directory contains code for:

- **Personal Data Generation**: Real, fake, celebrity, and random identity details managed by a personal data module.
- **GUI**: A Pygame-based interface presenting stimuli and recording participant responses.
- **EEG Headset Integration**: Data acquisition scripts utilizing MNE and BrainAccess libraries, with annotated trials.

**Relevant Links:**

- [Experiment README](./experiment/README.md)
- [Personal Data README](./experiment/src/personal_data/README.md)
- [EEG Headset README](./experiment/src/eeg_headset/README.md)
- [GUI README](./experiment/src/gui/README.md)

---

## Preprocessing, Feature Extraction & ICA

Before model training, EEG signals were preprocessed to remove noise and artifacts. Techniques included band-pass filtering, notch filters, and ICA for artifact removal. Additional feature sets were engineered (mean, std, variance, skewness, kurtosis, frequency band powers) to boost classification performance.

**Relevant Links:**

- [Data Extractor README](./classificators_and_data/data_extractor/README.md)
- [Machine Learning ICA README](./classificators_and_data/machine_learning/ica/README.md)
- [Data Preprocessing & Feature Engineering Notebooks](./classificators_and_data/training)

---

## Machine Learning & Neural Networks

Multiple classifiers were tested:

- **Traditional ML**: Random Forest, SVM, KNN, Logistic Regression.
- **Neural Networks**: DGCNN, FBCNet, and LSTM architectures trained on EEG timeseries and extracted features.

Grid searches and hyperparameter tuning refined model performance. Subject-based and random splits were compared, highlighting the challenges in generalizing across individuals.

**Relevant Links:**

- [Machine Learning README](./classificators_and_data/machine_learning/README.md)
- [Final Models README](./classificators_and_data/final_models/README.md)
- [Neural Networks README](./classificators_and_data/final_models/neural_networks/README.md)
- [Random Forest README](./classificators_and_data/final_models/random_forest/README.md)

---

## EDA & Results Visualization

Extensive Exploratory Data Analysis (EDA) provided insights into response times, event-related potentials (ERPs), and participant consistency. While EDA findings did not directly influence model training, they offered a deeper understanding of the dataset.

**Relevant Links:**

- [EDA Folder & Notebooks](./classificators_and_data/EDA/README.md)

_Placeholder for EDA Plots:_

- **Average Response Times**: ![Placeholder: Average Response Times](./classificators_and_data/EDA/plots_images/placeholder_average_response_times.png)
- **ERP Visualizations**: ![Placeholder: ERP Plot](./classificators_and_data/EDA/plots_images/placeholder_erp_plot.png)

---

## References and Related Work

- Original paper: [Neural processes underlying faking and concealing a personal identity: An EEG study](https://www.researchgate.net/publication/368455020_Neural_processes_underlying_faking_and_concealing_a_personal_identity_An_electroencephalogram_study)
- EEG and MNE documentation: [MNE-Python](https://mne.tools/)
- TorchEEG: [TorchEEG Documentation](https://torcheeg.readthedocs.io/)

---

## Placeholders for Media

- **Logo at top**: Add your project logo above.
- **Screenshots from the GUI or Experiment Setup**:
  - ![Placeholder: GUI Screenshot](./experiment/src/gui/assets/placeholder_gui_screenshot.png)
  - ![Placeholder: EEG Setup Screenshot](./experiment/src/assets/placeholder_eeg_setup.png)

---
