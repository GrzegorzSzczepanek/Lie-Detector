import ast
import os
import json
import joblib
from datetime import datetime
import pandas as pd
import re

def save_model_2(best_estimator, result_folder):
    """
    Save the entire best model pipeline to the result folder.
    """
    model_path = os.path.join(result_folder, 'best_model.joblib')
    joblib.dump(best_estimator, model_path)
    print(f"Best model saved to {model_path}")





def read_grid_search_results(file_path):
    """
    Reads the grid search results from a JSON file and parses the data.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract cv_results and other relevant data
    cv_results = data['cv_results']
    best_params = data['best_params']
    best_score = data['best_score_cv']
    test_score = data['test_score']
    classification_report = data.get('classification_report', None)
    
    # Function to parse arrays from strings with space-separated values
    def parse_array_string(s, key_name):
        # Remove the square brackets
        s = s.strip()[1:-1]
        # Split the string by any whitespace
        elements = re.split(r'\s+', s)
        result = []
        for elem in elements:
            if elem:
                # Handle parameter keys separately
                if key_name.startswith('param_'):
                    # Try to interpret as bool, None, or keep as string
                    if elem == 'True':
                        value = True
                    elif elem == 'False':
                        value = False
                    elif elem == 'None':
                        value = None
                    elif elem.startswith("'") and elem.endswith("'"):
                        value = elem[1:-1]
                    else:
                        value = elem
                else:
                    # For score-related keys, convert to float
                    try:
                        value = float(elem)
                    except ValueError:
                        # If conversion fails, keep as string
                        value = elem
                result.append(value)
        return result
    
    # Convert string representations of arrays back to actual arrays
    for key in cv_results:
        value = cv_results[key]
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            # Parse the array string
            cv_results[key] = parse_array_string(value, key)
        elif isinstance(value, list) and isinstance(value[0], str) and value[0].startswith('['):
            # For lists of strings representing arrays
            cv_results[key] = [parse_array_string(v, key) for v in value]
    
    # Convert cv_results to a pandas DataFrame
    cv_results_df = pd.DataFrame(cv_results)
    
    return cv_results_df, best_params, best_score, test_score, classification_report

def save_results(result, result_folder):
    os.makedirs(result_folder, exist_ok=True)
    filename = f"result_{result['model_name']}_{result['search_type']}_{result['cv_folds']}fold_{int(datetime.now().timestamp())}.json"
    filepath = os.path.join(result_folder, filename)
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=4, default=str)
    print(f"Results saved to {filepath}")

def save_model(best_estimator, result_folder):
    """
    Save the best model to the result folder.
    """
    model_filename = f"{best_estimator.named_steps['classifier'].__class__.__name__}_best_model_{int(datetime.now().timestamp())}.joblib"
    model_filepath = os.path.join(result_folder, model_filename)
    
    joblib.dump(best_estimator, model_filepath)
    print(f"Model saved to {model_filepath}")
    

def load_results(result_folder='results'):
    result_files = [f for f in os.listdir(result_folder) if f.endswith('.json')]
    all_results = []
    
    for file in result_files:
        filepath = os.path.join(result_folder, file)
        try:
            with open(filepath, 'r') as f:
                result = json.load(f)
                all_results.append(result)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {filepath}: {e}")
    
    return all_results


def display_results_summary(all_results):
    for i, result in enumerate(all_results):
        print(f"Result {i+1}:")
        print(f"  Model: {result['model_name']}")
        print(f"  Search Type: {result['search_type']}")
        print(f"  CV Folds: {result['cv_folds']}")
        print(f"  Best CV Score: {result['best_score_cv']}")
        print(f"  Test Score: {result['test_score']}")
        print(f"  Best Params: {result['best_params']}")
        print("--------------------------------------------------")


def get_all_dir_paths(data_dir):
    """
    Get all directory paths in the specified directory.

    Parameters:
    - data_dir (str): Base directory path containing the data folders.

    Returns:
    - list of str: List of full directory paths.
    """
    dir_paths = []

    for root, dirs, files in os.walk(data_dir):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            dir_paths.append(dir_path)
    
    return dir_paths


def load_json_results_from_folder(folder_path):
    """
    Load all JSON results from the given folder and return them as a list of dictionaries.
    """
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                result = json.load(file)
                results.append(result)
    return results