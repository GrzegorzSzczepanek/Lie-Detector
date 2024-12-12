
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
import json
import os
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime as datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import machine_learning.files_lib as FL
import machine_learning.preprocessing_lib as PL
import machine_learning.visualisation_lib as VL

def save_model(best_estimator, result_folder):
    """
    Save the best model to the result folder.
    """
    # Update the step name to 'model'
    model_filename = f"{best_estimator.named_steps['model'].__class__.__name__}_best_model_{int(datetime.now().timestamp())}.joblib"
    model_filepath = os.path.join(result_folder, model_filename)

    joblib.dump(best_estimator, model_filepath)

def grid_search_multiple_models(models_param_grids, X, y, test_size=0.2, cv=5, scoring='accuracy', result_folder='attempt'):
    # X = PL.preprocess_input_data(X)
    print(X.shape)
    X = X.reshape(X.shape[0], -1)    # Flatten 3D data to 2D
    assert X.shape[0] == y.shape[0], "Mismatch in sample sizes between X and y."
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    best_overall_model = None
    best_test_score = -np.inf

    if os.path.exists(result_folder):
        result_folder = f"{result_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    os.makedirs(result_folder, exist_ok=True)

    for model, param_grid in models_param_grids:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring=scoring, verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        test_score = grid_search.score(X_test, y_test)

        result = {
            'best_params': grid_search.best_params_,
            'best_score_cv': grid_search.best_score_,
            'test_score': test_score,
            'cv_results': PL.convert_cv_results(grid_search.cv_results_),
            'model_name': model.__class__.__name__,
            'search_type': 'grid_search_multiple_models',
            'cv_folds': cv,
            'date': f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        }
        
        FL.save_results(result, f"{result_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        if test_score > best_test_score:
            best_test_score = test_score
            best_overall_model = grid_search.best_estimator_

    if best_overall_model:
        save_model(best_overall_model, result_folder)

    return best_overall_model



def grid_search_models_with_saving(models_param_grids, X, y, test_size=0.2, cv=5, scoring='accuracy', result_folder='attempt_3'):
    X = PL.preprocess_input_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    best_overall_model = None
    best_test_score = -np.inf
    os.makedirs(result_folder, exist_ok=True)

    for model, param_grid in models_param_grids:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            verbose=2,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        test_score = grid_search.score(X_test, y_test)
        
        result = {
            'best_params': grid_search.best_params_,
            'best_score_cv': grid_search.best_score_,
            'test_score': test_score,
            'cv_results': PL.convert_cv_results(grid_search.cv_results_),
            'model_name': model.__class__.__name__,
            'search_type': 'grid_search_models_with_saving',
            'cv_folds': cv,
            'date': str(datetime.now())
        }
        
        FL.save_results(result, result_folder)
        
        if test_score > best_test_score:
            best_test_score = test_score
            best_overall_model = grid_search.best_estimator_

    if best_overall_model:
        FL.save_model(best_overall_model, result_folder)
        print("Best model saved successfully.")
    
    return best_overall_model


def grid_search_knn_pca(X, y, pca_components, knn_param_grid, test_size=0.2, cv=5, scoring='accuracy', result_folder='results/knn_with_pca'):
    os.makedirs(result_folder, exist_ok=True)
    
    X_flat = X.reshape(X.shape[0], -1)  
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=test_size, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('knn', KNeighborsClassifier())
    ])
    
    param_grid = {
        'pca__n_components': pca_components,  
        'knn__n_neighbors': knn_param_grid['n_neighbors'],  
        'knn__weights': knn_param_grid['weights'],
        'knn__metric': knn_param_grid['metric']
    }
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    test_score = grid_search.score(X_test, y_test)
    
    result = {
        'best_params': grid_search.best_params_,
        'best_score_cv': grid_search.best_score_,
        'test_score': test_score,
        'cv_results': grid_search.cv_results_,
        'model_name': 'KNeighborsClassifier with PCA',
        'search_type': 'grid_search_knn_pca',
        'cv_folds': cv,
        'date': str(datetime.now())
    }
    
    result_path = os.path.join(result_folder, 'grid_search_results.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4, default=str)
    
    best_model_path = os.path.join(result_folder, 'best_knn_pca_model.joblib')
    joblib.dump(grid_search.best_estimator_, best_model_path)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_}")
    print(f"Test Score: {test_score}")
    print(f"Best model saved to {best_model_path}")
    
    return grid_search.best_estimator_, grid_search.best_params_, test_score



def grid_search_random_forest(X, y, param_grid, test_size=0.2, cv=5, scoring='accuracy', result_folder=None, feature_names=None):
    """
    Perform grid search on RandomForestClassifier, save detailed results, and return the best model.

    Parameters:
    - X: numpy array or pandas DataFrame, feature matrix.
    - y: numpy array or pandas Series, target vector.
    - param_grid: dict, hyperparameters to tune.
    - test_size: float, proportion of the dataset to include in the test split.
    - cv: int, number of cross-validation folds.
    - scoring: str, scoring metric for evaluation.
    - result_folder: str, folder to save results. If None, a new folder is created.
    - feature_names: list, names of the features.

    Returns:
    - best_model: the best estimator found by GridSearchCV.
    - best_params: dict, parameters of the best model.
    - test_score: float, score of the best model on the test set.
    """
    if result_folder is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_folder = f"../results/random_forest_{timestamp}"
    os.makedirs(result_folder, exist_ok=True)
    
    X_flat = X.reshape(X.shape[0], -1)

    
    
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=test_size, random_state=42)
    
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=2,
        n_jobs=-1,
        return_train_score=True  
    )
    
    
    grid_search.fit(X_train, y_train)
    
    
    best_model = grid_search.best_estimator_
    
    
    test_score = best_model.score(X_test, y_test)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    
    result = {
        'best_params': grid_search.best_params_,
        'best_score_cv': grid_search.best_score_,
        'test_score': test_score,
        'cv_results': grid_search.cv_results_,
        'model_name': 'RandomForestClassifier',
        'search_type': 'grid_search_random_forest',
        'cv_folds': cv,
        'date': str(datetime.now()),
        'classification_report': classification_rep
    }
    
    
    FL.save_results(result, result_folder)
    
    
    FL.save_model(best_model, result_folder)
    
    
    
    VL.plot_confusion_matrix(cm, 'RandomForestClassifier', result_folder)
    
    
    VL.plot_cv_results(grid_search.cv_results_, 'RandomForestClassifier', result_folder)
    
    
    VL.plot_fit_and_score_times(grid_search.cv_results_, 'RandomForestClassifier', result_folder)
    
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    VL.plot_feature_importances(best_model, feature_names, 'RandomForestClassifier', result_folder)
    
    
    report_path = os.path.join(result_folder, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(y_test, y_pred))
    print(f"Classification report saved to {report_path}")
    
    return best_model, grid_search.best_params_, test_score




def preprocess_eeg_data(X, fs=256):
    """
    Preprocess EEG data and extract features.

    Parameters:
    - X: numpy array of shape (n_samples, n_channels, n_times)
    - fs: Sampling frequency of the EEG data (default: 256 Hz)

    Returns:
    - features: numpy array of shape (n_samples, n_features)
    """
    n_samples, n_channels, n_times = X.shape
    features_list = []
    for i in range(n_samples):
        sample = X[i]
        features = []
        for ch in range(n_channels):
            channel_data = sample[ch]
            
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            skewness = skew(channel_data)
            kurt = kurtosis(channel_data)
            
            freqs, psd = welch(channel_data, fs=fs, nperseg=256)
            
            delta_band = (0.5, 4)
            theta_band = (4, 8)
            alpha_band = (8, 12)
            beta_band = (12, 30)
            gamma_band = (30, 100)
            
            delta_power = _bandpower(psd, freqs, delta_band)
            theta_power = _bandpower(psd, freqs, theta_band)
            alpha_power = _bandpower(psd, freqs, alpha_band)
            beta_power = _bandpower(psd, freqs, beta_band)
            gamma_power = _bandpower(psd, freqs, gamma_band)
            
            features.extend([
                mean_val, std_val, skewness, kurt,
                delta_power, theta_power, alpha_power, beta_power, gamma_power
            ])
        features_list.append(features)
    return np.array(features_list)


def _bandpower(psd, freqs, band) -> float:
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    band_power = np.trapz(psd[idx_band], freqs[idx_band])
    return band_power


def grid_search_random_forest_eeg(X, y, param_grid, test_size=0.2, cv=5, scoring='accuracy', result_folder=None, random_state=42):
    """
    Grid search for RandomForestClassifier with EEG data preprocessing.
    """
    if result_folder is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        result_folder = f"../results/random_forest_eeg_{timestamp}"
    os.makedirs(result_folder, exist_ok=True)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    
    pipeline = Pipeline([
        ('preprocess', FunctionTransformer(preprocess_eeg_data, validate=False)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])
    
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        verbose=2,
        n_jobs=-1,
        return_train_score=True
    )
    
    grid_search.fit(X_train_raw, y_train)
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test_raw)
    
    return best_model, y_pred

    # I didn't need some of these things below for most of my training so I return only best model and preds
    test_score = best_model.score(X_test_raw, y_test)
    cm = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    n_channels = X.shape[1]
    feature_names = []
    for ch in range(n_channels):
        feature_names.extend([
            f'Ch{ch+1}_mean',
            f'Ch{ch+1}_std',
            f'Ch{ch+1}_skewness',
            f'Ch{ch+1}_kurtosis',
            f'Ch{ch+1}_delta_power',
            f'Ch{ch+1}_theta_power',
            f'Ch{ch+1}_alpha_power',
            f'Ch{ch+1}_beta_power',
            f'Ch{ch+1}_gamma_power'
        ])
    
    result = {
        'best_params': grid_search.best_params_,
        'best_score_cv': grid_search.best_score_,
        'test_score': test_score,
        'cv_results': grid_search.cv_results_,
        'model_name': 'RandomForestClassifier',
        'search_type': 'grid_search_random_forest_eeg',
        'cv_folds': cv,
        'date': str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
        'classification_report': classification_rep
    }
    
    FL.save_results(result, result_folder)
    
    FL.save_model(best_model, result_folder)
    
    VL.plot_confusion_matrix(cm, 'RandomForestClassifier', result_folder)
    VL.plot_cv_results(grid_search.cv_results_, 'RandomForestClassifier', result_folder)
    VL.plot_fit_and_score_times(grid_search.cv_results_, 'RandomForestClassifier', result_folder)
    
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    
    channel_importances = _aggregate_importances_per_channel(feature_importances, n_channels)
    
    VL.plot_feature_importances(channel_importances, n_channels, 'RandomForestClassifier', result_folder)
    
    report_path = os.path.join(result_folder, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(y_test, y_pred))
    print(f"Classification report saved to {report_path}")
    
    return best_model, grid_search.best_params_, test_score

# IMPORTANT - things related to my feature extraction approach
def _aggregate_importances_per_channel(feature_importances, n_channels):
    """
    Aggregate feature importances per channel.
    """
    n_features_per_channel = len(feature_importances) // n_channels
    importances_per_channel = np.zeros(n_channels)
    for ch in range(n_channels):
        start_idx = ch * n_features_per_channel
        end_idx = start_idx + n_features_per_channel
        importances_per_channel[ch] = np.sum(np.abs(feature_importances[start_idx:end_idx]))
    return importances_per_channel


import inspect

def feature_extraction_grid_search(X, y, param_grid, test_size=0.2, cv=5, scoring='accuracy', result_folder=None, random_state=42, classifier=None, param_grid_classifier=None, model_name=''):
    """
    Grid search for with EEG data preprocessing.
    """

    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if result_folder is None:
        assert classifier is not None, "Classifier must be provided."
        result_folder = f"../results/{model_name}_{timestamp}"
    else:
        result_folder = f"../results/{result_folder}_{timestamp}"
    
    os.makedirs(result_folder, exist_ok=True)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    sig = inspect.signature(classifier.__init__)
    if 'random_state' in sig.parameters:
        classifier_instance = classifier(random_state=random_state)
    else:
        classifier_instance = classifier()
        
    if X_train_raw.ndim > 2:
        X_train_raw = X_train_raw.reshape(X_train_raw.shape[0], -1)
        X_test_raw = X_test_raw.reshape(X_test_raw.shape[0], -1)
    
    pipeline = Pipeline([
        ('preprocess', FunctionTransformer(preprocess_eeg_data, validate=False)),
        ('scaler', StandardScaler()),
        ('classifier', classifier_instance)
    ])
    
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        verbose=2,
        n_jobs=-1,
        return_train_score=True
    )
    
    grid_search.fit(X_train_raw, y_train)
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test_raw)
    test_score = best_model.score(X_test_raw, y_test)
    cm = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    n_channels = X.shape[1]
    feature_names = []
    for ch in range(n_channels):
        feature_names.extend([
            f'Ch{ch+1}_mean',
            f'Ch{ch+1}_std',
            f'Ch{ch+1}_skewness',
            f'Ch{ch+1}_kurtosis',
            f'Ch{ch+1}_delta_power',
            f'Ch{ch+1}_theta_power',
            f'Ch{ch+1}_alpha_power',
            f'Ch{ch+1}_beta_power',
            f'Ch{ch+1}_gamma_power'
        ])
    
    result = {
        'best_params': grid_search.best_params_,
        'best_score_cv': grid_search.best_score_,
        'test_score': test_score,
        'cv_results': grid_search.cv_results_,
        'model_name': model_name,
        'search_type': 'grid_search_random_forest_eeg',
        'cv_folds': cv,
        # 'date': str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
        'date': str("test"),
        'classification_report': classification_rep
    }
    
    FL.save_results(result, result_folder)
    
    FL.save_model(best_model, result_folder)
    
    VL.plot_confusion_matrix(cm, model_name, result_folder)
    VL.plot_cv_results(grid_search.cv_results_, model_name, result_folder)
    VL.plot_fit_and_score_times(grid_search.cv_results_, model_name, result_folder)
    
    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        feature_importances = best_model.named_steps['classifier'].feature_importances_
        channel_importances = _aggregate_importances_per_channel(feature_importances, n_channels)
        VL.plot_feature_importances(channel_importances, n_channels, 'RandomForestClassifier', result_folder)
    
    report_path = os.path.join(result_folder, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(y_test, y_pred))
    print(f"Classification report saved to {report_path}")
    
    return best_model, grid_search.best_params_, test_score





def grid_search_logistic_regression_eeg(X, y, param_grid, test_size=0.2, cv=5, scoring='accuracy', result_folder=None, random_state=42):
    """
    Grid search for Logistic Regression with EEG data preprocessing for binary classification.
    """
    if result_folder is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        result_folder = f"../results/logistic_regression_eeg_{timestamp}"
    os.makedirs(result_folder, exist_ok=True)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    pipeline = Pipeline([
        ('preprocess', FunctionTransformer(preprocess_eeg_data, validate=False)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=random_state))
    ])
    
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        verbose=2,
        n_jobs=-1,
        return_train_score=True
    )
    
    grid_search.fit(X_train_raw, y_train)
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test_raw)
    test_score = best_model.score(X_test_raw, y_test)
    cm = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    n_channels = X.shape[1]
    feature_names = []
    for ch in range(n_channels):
        feature_names.extend([
            f'Ch{ch+1}_mean',
            f'Ch{ch+1}_std',
            f'Ch{ch+1}_skewness',
            f'Ch{ch+1}_kurtosis',
            f'Ch{ch+1}_delta_power',
            f'Ch{ch+1}_theta_power',
            f'Ch{ch+1}_alpha_power',
            f'Ch{ch+1}_beta_power',
            f'Ch{ch+1}_gamma_power'
        ])
    
    result = {
        'best_params': grid_search.best_params_,
        'best_score_cv': grid_search.best_score_,
        'test_score_accuracy': test_score,
        'cv_results': grid_search.cv_results_,
        'model_name': 'LogisticRegression',
        'search_type': 'grid_search_logistic_regression_eeg',
        'cv_folds': cv,
        'date': str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
        'classification_report': classification_rep
    }
    
    FL.save_results(result, result_folder)
    
    FL.save_model(best_model, result_folder)
    
    VL.plot_confusion_matrix(cm, 'LogisticRegression', result_folder)
    VL.plot_cv_results(grid_search.cv_results_, 'LogisticRegression', result_folder)
    VL.plot_fit_and_score_times(grid_search.cv_results_, 'LogisticRegression', result_folder)
    
    # For Logistic Regression, coefficients are in coef_
    coefficients = best_model.named_steps['classifier'].coef_.flatten()
    
    channel_coefficients = _aggregate_coefficients_per_channel(coefficients, n_channels)
    
    VL.plot_feature_importances(channel_coefficients, n_channels, 'LogisticRegression', result_folder)
    
    report_path = os.path.join(result_folder, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(y_test, y_pred))
    print(f"Classification report saved to {report_path}")
    
    return best_model, grid_search.best_params_, test_score

def _aggregate_coefficients_per_channel(coefficients, n_channels):
    """
    Aggregate coefficients per channel for classification.
    """
    n_features_per_channel = len(coefficients) // n_channels
    coefficients_per_channel = np.zeros(n_channels)
    for ch in range(n_channels):
        start_idx = ch * n_features_per_channel
        end_idx = start_idx + n_features_per_channel
        coefficients_per_channel[ch] = np.sum(np.abs(coefficients[start_idx:end_idx]))
    return coefficients_per_channel


def grid_search_knn_eeg(X, y, param_grid=None, test_size=0.2, cv=5, scoring='accuracy', result_folder=None, random_state=42, apply_pca=False):
    """
    Grid search for KNeighborsClassifier with EEG data preprocessing.

    Parameters:
    - X: numpy array of shape (n_samples, n_channels, n_times)
    - y: labels
    - param_grid: dictionary with parameters for GridSearchCV
    - test_size: fraction of data to use for test set
    - cv: number of cross-validation folds
    - scoring: scoring metric
    - result_folder: folder to save results
    - random_state: random seed
    - apply_pca: boolean, whether to apply PCA
    """
    if result_folder is None:
        result_folder = f"../results/knn_eeg_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    os.makedirs(result_folder, exist_ok=True)
    
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    
    if apply_pca:
        steps = [
            ('preprocess', FunctionTransformer(preprocess_eeg_data, validate=False)),
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('classifier', KNeighborsClassifier())
        ]
        pipeline = Pipeline(steps)
        
        
        if param_grid is None:
            param_grid = {
                'pca__n_components': [10, 20, 30],
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan']
            }
    else:
        steps = [
            ('preprocess', FunctionTransformer(preprocess_eeg_data, validate=False)),
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ]
        pipeline = Pipeline(steps)
        
        
        if param_grid is None:
            param_grid = {
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan']
            }
        else:
            
            param_grid = {key: param_grid[key] for key in param_grid if not key.startswith('pca__')}
    
    
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        verbose=2,
        n_jobs=-1,
        return_train_score=True
    )
    
    
    grid_search.fit(X_train_raw, y_train)
    
    
    best_model = grid_search.best_estimator_
    
    
    y_pred = best_model.predict(X_test_raw)
    test_score = best_model.score(X_test_raw, y_test)
    cm = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    
    result = {
        'best_params': grid_search.best_params_,
        'best_score_cv': grid_search.best_score_,
        'test_score': test_score,
        'cv_results': grid_search.cv_results_,
        'model_name': 'KNeighborsClassifier',
        'search_type': 'grid_search_knn_eeg',
        'cv_folds': cv,
        'date': str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
        'classification_report': classification_rep
    }
    
    
    FL.save_results(result, result_folder)
    FL.save_model(best_model, result_folder)
    
    
    VL.plot_confusion_matrix(cm, 'KNeighborsClassifier', result_folder)
    VL.plot_cv_results(grid_search.cv_results_, 'KNeighborsClassifier', result_folder)
    VL.plot_fit_and_score_times(grid_search.cv_results_, 'KNeighborsClassifier', result_folder)
    
    
    report_path = os.path.join(result_folder, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(y_test, y_pred))
    print(f"Classification report saved to {report_path}")
    
    return best_model, grid_search.best_params_, test_score


### EXPERIMENTAL APPROACH ###

# def preprocess_eeg_data_2(X, fs=256):
#     """
#     Preprocess EEG data and extract features, including frequency bands relevant to deception detection.

#     Parameters:
#     - X: numpy array of shape (n_samples, n_channels, n_times)
#     - fs: Sampling frequency of the EEG data (default: 256 Hz)

#     Returns:
#     - features: numpy array of shape (n_samples, n_features)
#     """
#     n_samples, n_channels, n_times = X.shape
#     features_list = []
#     for i in range(n_samples):
#         sample = X[i]
#         features = []
#         for ch in range(n_channels):
#             channel_data = sample[ch]
            
#             mean_val = np.mean(channel_data)
#             std_val = np.std(channel_data)
#             skewness_val = skew(channel_data)
#             kurtosis_val = kurtosis(channel_data)
            
#             freqs, psd = welch(channel_data, fs=fs, nperseg=256)
            
#             # Frequency bands relevant to deception detection
#             delta_band = (0.5, 4)
#             theta_band = (4, 8)
#             low_alpha_band = (8, 10)
#             high_alpha_band = (10, 12)
#             low_beta_band = (12, 15)
#             high_beta_band = (15, 30)
#             gamma_band = (30, 45)
#             high_gamma_band = (55, 80)
#             mu_band = (8, 13)
            
#             # Compute band powers
#             delta_power = _bandpower(psd, freqs, delta_band)
#             theta_power = _bandpower(psd, freqs, theta_band)
#             low_alpha_power = _bandpower(psd, freqs, low_alpha_band)
#             high_alpha_power = _bandpower(psd, freqs, high_alpha_band)
#             low_beta_power = _bandpower(psd, freqs, low_beta_band)
#             high_beta_power = _bandpower(psd, freqs, high_beta_band)
#             gamma_power = _bandpower(psd, freqs, gamma_band)
#             high_gamma_power = _bandpower(psd, freqs, high_gamma_band)
#             mu_power = _bandpower(psd, freqs, mu_band)
            
#             features.extend([
#                 mean_val, std_val, skewness_val, kurtosis_val,
#                 delta_power, theta_power,
#                 low_alpha_power, high_alpha_power,
#                 low_beta_power, high_beta_power,
#                 gamma_power, high_gamma_power,
#                 mu_power
#             ])
#         features_list.append(features)
#     return np.array(features_list)

# def _bandpower(psd, freqs, band):
#     idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
#     band_power = np.trapz(psd[idx_band], freqs[idx_band])
#     return band_power

# def grid_search_random_forest_eeg_2(X, y, param_grid, test_size=0.2, cv=5, scoring='accuracy', result_folder=None, random_state=42):
#     """
#     Grid search for RandomForestClassifier with EEG data preprocessing, including frequency bands relevant to deception detection.
#     """
#     if result_folder is None:
#         timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
#         result_folder = f"../results/random_forest_eeg_{timestamp}"
#     os.makedirs(result_folder, exist_ok=True)
    
#     X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
#     pipeline = Pipeline([
#         ('preprocess', FunctionTransformer(preprocess_eeg_data_2, validate=False)),
#         ('scaler', StandardScaler()),
#         ('classifier', RandomForestClassifier(random_state=random_state))
#     ])
    
#     cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
#     grid_search = GridSearchCV(
#         estimator=pipeline,
#         param_grid=param_grid,
#         cv=cv_strategy,
#         scoring=scoring,
#         verbose=2,
#         n_jobs=-1,
#         return_train_score=True
#     )
    
#     grid_search.fit(X_train_raw, y_train)
    
#     best_model = grid_search.best_estimator_
    
#     y_pred = best_model.predict(X_test_raw)
#     test_score = best_model.score(X_test_raw, y_test)
#     cm = confusion_matrix(y_test, y_pred)
#     classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
#     n_channels = X.shape[1]
#     feature_names = []
#     for ch in range(n_channels):
#         feature_names.extend([
#             f'Ch{ch+1}_mean',
#             f'Ch{ch+1}_std',
#             f'Ch{ch+1}_skewness',
#             f'Ch{ch+1}_kurtosis',
#             f'Ch{ch+1}_delta_power',
#             f'Ch{ch+1}_theta_power',
#             f'Ch{ch+1}_low_alpha_power',
#             f'Ch{ch+1}_high_alpha_power',
#             f'Ch{ch+1}_low_beta_power',
#             f'Ch{ch+1}_high_beta_power',
#             f'Ch{ch+1}_gamma_power',
#             f'Ch{ch+1}_high_gamma_power',
#             f'Ch{ch+1}_mu_power'
#         ])
    
#     result = {
#         'best_params': grid_search.best_params_,
#         'best_score_cv': grid_search.best_score_,
#         'test_score': test_score,
#         'cv_results': grid_search.cv_results_,
#         'model_name': 'RandomForestClassifier',
#         'search_type': 'grid_search_random_forest_eeg_2',
#         'cv_folds': cv,
#         'date': str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
#         'classification_report': classification_rep
#     }
    
#     FL.save_results(result, result_folder)
#     FL.save_model(best_model, result_folder)
    
#     VL.plot_confusion_matrix(cm, 'RandomForestClassifier', result_folder)
#     VL.plot_cv_results(grid_search.cv_results_, 'RandomForestClassifier', result_folder)
#     VL.plot_fit_and_score_times(grid_search.cv_results_, 'RandomForestClassifier', result_folder)
    
#     feature_importances = best_model.named_steps['classifier'].feature_importances_
    
#     channel_importances = _aggregate_importances_per_channel_2(feature_importances, n_channels)
    
#     VL.plot_feature_importances(channel_importances, n_channels, 'RandomForestClassifier', result_folder)
    
#     report_path = os.path.join(result_folder, 'classification_report.txt')
#     with open(report_path, 'w') as f:
#         f.write(classification_report(y_test, y_pred))
#     print(f"Classification report saved to {report_path}")
    
#     return best_model, grid_search.best_params_, test_score

# def _aggregate_importances_per_channel_2(feature_importances, n_channels):
#     """
#     Aggregate feature importances per channel for the extended feature set.
#     """
#     n_features_per_channel = len(feature_importances) // n_channels
#     importances_per_channel = np.zeros(n_channels)
#     for ch in range(n_channels):
#         start_idx = ch * n_features_per_channel
#         end_idx = start_idx + n_features_per_channel
#         importances_per_channel[ch] = np.sum(np.abs(feature_importances[start_idx:end_idx]))
#     return importances_per_channel


def preprocess_eeg_data_2(X, fs=256):
    """
    Preprocess EEG data and extract features, including frequency bands relevant to deception detection.
    """
    n_samples, n_channels, n_times = X.shape
    features_list = []
    for i in range(n_samples):
        sample = X[i]
        features = []
        for ch in range(n_channels):
            channel_data = sample[ch]

            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            skewness_val = skew(channel_data)
            kurtosis_val = kurtosis(channel_data)

            entropy_val = _approximate_entropy(channel_data)
            sample_entropy_val = _sample_entropy(channel_data)

            hjorth_mobility, hjorth_complexity = _hjorth_parameters(channel_data)

            freqs, psd = welch(channel_data, fs=fs, nperseg=256)

            delta_band = (0.5, 4)
            theta_band = (4, 8)
            low_alpha_band = (8, 10)
            high_alpha_band = (10, 12)
            low_beta_band = (12, 15)
            high_beta_band = (15, 30)
            gamma_band = (30, 45)
            high_gamma_band = (55, 80)
            mu_band = (8, 13)

            delta_power = _bandpower(psd, freqs, delta_band)
            theta_power = _bandpower(psd, freqs, theta_band)
            low_alpha_power = _bandpower(psd, freqs, low_alpha_band)
            high_alpha_power = _bandpower(psd, freqs, high_alpha_band)
            low_beta_power = _bandpower(psd, freqs, low_beta_band)
            high_beta_power = _bandpower(psd, freqs, high_beta_band)
            gamma_power = _bandpower(psd, freqs, gamma_band)
            high_gamma_power = _bandpower(psd, freqs, high_gamma_band)
            mu_power = _bandpower(psd, freqs, mu_band)

            features.extend([
                mean_val, std_val, skewness_val, kurtosis_val,
                entropy_val, sample_entropy_val,
                hjorth_mobility, hjorth_complexity,
                delta_power, theta_power,
                low_alpha_power, high_alpha_power,
                low_beta_power, high_beta_power,
                gamma_power, high_gamma_power,
                mu_power
            ])
        features_list.append(features)
    return np.array(features_list)

def _bandpower(psd, freqs, band):
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    band_power = np.trapz(psd[idx_band], freqs[idx_band])
    return band_power

def _approximate_entropy(U, m=2, r=None):
    """
    Calculate the approximate entropy of a time series.
    """
    if r is None:
        r = 0.2 * np.std(U)
    N = len(U)
    def _phi(m):
        x = np.array([U[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1)
    return abs(_phi(m + 1) - _phi(m))

def _sample_entropy(U, m=2, r=None):
    """
    Calculate the sample entropy of a time series.
    """
    if r is None:
        r = 0.2 * np.std(U)
    N = len(U)
    def _phi(m):
        x = np.array([U[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0)
        return C
    B = np.sum(_phi(m))
    A = np.sum(_phi(m + 1))
    return -np.log(A / B) if A != 0 and B != 0 else 0

def _hjorth_parameters(U):
    """
    Calculate the Hjorth mobility and complexity parameters.
    """
    first_derivative = np.diff(U)
    second_derivative = np.diff(U, n=2)
    var_zero = np.var(U)
    var_d1 = np.var(first_derivative)
    var_d2 = np.var(second_derivative)
    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / mobility
    return mobility, complexity


from sklearn.feature_selection import RFECV


def grid_search_random_forest_eeg_2(X, y, param_grid, test_size=0.2, cv=5, scoring='accuracy',
                                    result_folder=None, random_state=42):
    """
    Grid search for RandomForestClassifier with EEG data preprocessing, including feature selection and additional features.
    """
    if result_folder is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        result_folder = f"../results/random_forest_eeg_{timestamp}"
    os.makedirs(result_folder, exist_ok=True)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_preprocessed = preprocess_eeg_data_2(X_train_raw)
    X_test_preprocessed = preprocess_eeg_data_2(X_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_preprocessed)
    X_test_scaled = scaler.transform(X_test_preprocessed)

    estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    selector = RFECV(estimator, step=1, cv=StratifiedKFold(cv), scoring=scoring, n_jobs=-1)
    selector = selector.fit(X_train_scaled, y_train)

    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=random_state, n_jobs=-1))
    ])

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        verbose=2,
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train_selected, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_selected)
    test_score = best_model.score(X_test_selected, y_test)
    cm = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    result = {
        'best_params': grid_search.best_params_,
        'best_score_cv': grid_search.best_score_,
        'test_score': test_score,
        'cv_results': grid_search.cv_results_,
        'model_name': 'RandomForestClassifier',
        'search_type': 'grid_search_random_forest_eeg_2',
        'cv_folds': cv,
        'date': str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
        'classification_report': classification_rep
    }

    FL.save_results(result, result_folder)
    FL.save_model(best_model, result_folder)

    VL.plot_confusion_matrix(cm, 'RandomForestClassifier', result_folder)
    VL.plot_cv_results(grid_search.cv_results_, 'RandomForestClassifier', result_folder)
    VL.plot_fit_and_score_times(grid_search.cv_results_, 'RandomForestClassifier', result_folder)

    feature_importances = best_model.named_steps['classifier'].feature_importances_
    selected_features = selector.get_support(indices=True)

    full_feature_importances = np.zeros(X_train_scaled.shape[1])
    full_feature_importances[selected_features] = feature_importances

    n_channels = X.shape[1]
    channel_importances = _aggregate_importances_per_channel_2(full_feature_importances, n_channels)

    VL.plot_feature_importances(channel_importances, n_channels, 'RandomForestClassifier', result_folder)

    report_path = os.path.join(result_folder, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(y_test, y_pred))
    print(f"Classification report saved to {report_path}")

    return best_model, grid_search.best_params_, test_score

def _aggregate_importances_per_channel_2(feature_importances, n_channels):
    """
    Aggregate feature importances per channel for the extended feature set.
    """
    n_features_total = len(feature_importances)
    n_features_per_channel = n_features_total // n_channels
    importances_per_channel = np.zeros(n_channels)
    for ch in range(n_channels):
        start_idx = ch * n_features_per_channel
        end_idx = start_idx + n_features_per_channel
        importances_per_channel[ch] = np.sum(np.abs(feature_importances[start_idx:end_idx]))
    return importances_per_channel


