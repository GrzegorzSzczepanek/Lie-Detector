from LieModel import LieModel
import numpy as np

class ExampleLieModel(LieModel):

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.ones(X_test.shape[0])

    def determinate_preprocess_config(self) -> dict[any, any] | list[dict[any, any]]:
        return [{
            "lfreq": 1,
            "hfreq": 50,
            "notch_filter": [50, 100],
            "baseline": (0, 0),
            "tmin": 0,
            "tmax": 1
        },
        {
            "lfreq": 1,
            "hfreq": 50,
            "notch_filter": [50, 100],
            "baseline": (0, 0),
            "tmin": 0,
            "tmax": 1
        }
    ]
    
    def determinate_run_info(self) -> dict[str, str]:
        return {
            "person": "ms",
            "model": "example",
            "additional_info": "This is an example"
        }