from neural_networks.ai.constants import (
    NUM_OF_CLASSES,
    NUM_OF_ELECTRODES,
    STANDARD_BAND_TO_FREQUENCY_RANGE,
)
from neural_networks.ai.dataset.bde_dataset import BDEDataset
from neural_networks.ai.logger.json_logger import JSONLogger
from neural_networks.ai.trainer.typings import HyperparamType, RunConfig
from torcheeg.trainers import ClassifierTrainer
from torcheeg.models import DGCNN

CONFIG_TO_RUN: RunConfig = {
    "model_name": "DGCNN",
    "run_name": "DGCNN - test after refactor (with tmax 1.997) [ON ALL DATA]",
    "dataset_factory": lambda X, y, **kwargs: BDEDataset(X, y, **kwargs),
    "model_factory": lambda **kwargs: DGCNN(
        in_channels=5,
        num_electrodes=NUM_OF_ELECTRODES,
        num_classes=NUM_OF_CLASSES,
        **kwargs
    ),
    "trainer_factory": lambda **kwargs: ClassifierTrainer(**kwargs),
    "logger_factory": lambda model_name, run_id, run_name: JSONLogger(
        model_name, run_id, run_name
    ),
    "params": [
        {
            "type": HyperparamType.Data,
            "name": "tmax",
            "values_to_test": (),
            "default_value": 1.997,
        },
        {
            "type": HyperparamType.Dataset,
            "name": "band_to_frequency_range",
            "values_to_test": (),
            "default_value": STANDARD_BAND_TO_FREQUENCY_RANGE,
        },
        {
            "type": HyperparamType.DataLoader,
            "name": "batch_size",
            "values_to_test": (),
            "default_value": 16,
        },
        {
            "type": HyperparamType.Model,
            "name": "num_layers",
            "values_to_test": (),
            "default_value": 2,
        },
        {
            "type": HyperparamType.Model,
            "name": "hid_channels",
            "values_to_test": (),
            "default_value": 32,
        },
        {
            "type": HyperparamType.Trainer,
            "name": "lr",
            "values_to_test": (),
            "default_value": 10**-3,
        },
        {
            "type": HyperparamType.Fit,
            "name": "max_epochs",
            "values_to_test": (),
            "default_value": 500,
        },
    ],
}
