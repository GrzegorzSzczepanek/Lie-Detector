from datetime import datetime
import json
import sys
import os

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(project_root)

from final_models.LieModel import LieModel
from final_models.ExperimentManager import ExperimentManager
import machine_learning.ml_lib as ML

class RandomForestModel(LieModel):
    def __init__(self, preprocess_params=None) -> None:
        super().__init__()
        self.params_json_path = 'result_RandomForestClassifier_grid_search_random_forest_eeg_5fold_1728934233.json'
        self.model = None
        self.scaler = StandardScaler()
        self.preprocess_params = preprocess_params or {}

    def train(self, X_train, y_train) -> None:
        """
        Train the model using the provided data.
        """
        best_params = self._load_best_params()
        features = self._preprocess_data(X_train)
        self.scaler.fit(features)
        preprocessed_X_train = self.scaler.transform(features)
        self.model = RandomForestClassifier(**best_params)
        self.model.fit(X=preprocessed_X_train, y=y_train)

    def predict(self, X_test) -> np.ndarray:
        """
        Predict the labels for the provided data.
        """
        features = self._preprocess_data(X_test)
        preprocessed_X_test = self.scaler.transform(features)
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call the 'train' method before making predictions.")
        return self.model.predict(preprocessed_X_test)
    
    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess the data.
        """
        return ML.preprocess_eeg_data(X)
    
    def determinate_preprocess_config(self) -> dict:
        """
        Return the configuration of the preprocessing.
        """
        config = {
            "lfreq": 0.3,
            "hfreq": 70,
            "notch_filter": [60],
            "baseline": (None, None),
            "tmin": 0,
            "tmax": 0.6,
        }
        
        config.update(self.preprocess_params)
        return config
    
    def determinate_run_info(self):
        """
        Return the information about the model and the person who created it.
        """
        run_info = {
            "person": "gs",
            "model": f'RandomForestModel{datetime.now().strftime("%Y-%m-%d, %H:%M:%S")}',
            "additional_info": "RandomForestModel",
            "preprocess_params": self.determinate_preprocess_config(),
        }
        return run_info
    
    def _load_best_params(self) -> dict[str, any]:
        """
        Load and process the best parameters from a JSON file.
        """
        if not os.path.exists(self.params_json_path):
            raise FileNotFoundError(f"Parameters file not found: {self.params_json_path}")

        with open(self.params_json_path, 'r') as f:
            data = json.load(f)
        
        best_params = data.get('best_params')
        if best_params is None:
            raise KeyError("'best_params' not found in the JSON file.")

        
        params = {}
        for key, value in best_params.items():
            param_name = key.split('__')[-1]
            params[param_name] = value

        return params
    

if __name__ == "__main__":
    test = RandomForestModel()

    # print(test.debug())
    X_train = np.random.rand(10, 5)  # 10 samples, 5 features each
    y_train = np.random.randint(0, 2, 10)  # 10 labels, binary classification (0 or 1)
    test.train(X_train, y_train)