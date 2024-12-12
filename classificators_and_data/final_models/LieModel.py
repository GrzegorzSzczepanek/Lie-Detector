import abc
import numpy as np

class LieModel(metaclass = abc.ABCMeta):
    
    @abc.abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod  
    def determinate_preprocess_config(self) -> dict[any, any] | list[dict[any, any]]:
        return {
            "lfreq": 1,
            "hfreq": 50,
            "notch_filter": [50, 100],
            "baseline": (0, 0),
            "tmin": 0,
            "tmax": 1
        }
    
    @abc.abstractmethod
    def determinate_run_info(self) -> dict[str, str]:
        return {
            "person": "ms",
            "model": "CNN", # CNNv1, CNNv2 itd.
            "additional_info": "To ten taki, co działał dobrze"
        }
