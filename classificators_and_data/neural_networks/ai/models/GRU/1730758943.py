from neural_networks.ai.constants import (
    NUM_OF_CLASSES,
    NUM_OF_ELECTRODES,
    STANDARD_BAND_TO_FREQUENCY_RANGE,
)
from neural_networks.ai.dataset.bde_dataset import BDEDataset
from neural_networks.ai.logger.json_logger import JSONLogger
from neural_networks.ai.trainer.typings import HyperparamType, RunConfig
from torcheeg.trainers import ClassifierTrainer
from torcheeg.models import GRU

CONFIG_TO_RUN: RunConfig = {
    "model_name": "GRU",
    "run_name": "GRU [ON ALL DATA]",
    "dataset_factory": lambda X, y, **kwargs: BDEDataset(X, y, **kwargs),
    "model_factory": lambda **kwargs: GRU(
        num_electrodes=NUM_OF_ELECTRODES, num_classes=NUM_OF_CLASSES, **kwargs
    ),
    "trainer_factory": lambda **kwargs: ClassifierTrainer(**kwargs),
    "logger_factory": lambda model_name, run_id, run_name: JSONLogger(
        model_name, run_id, run_name
    ),
    "params": [
        {
            "type": HyperparamType.Data,
            "name": "tmax",
            "values_to_test": (1.997,),
            "default_value": 1,
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
            "values_to_test": (16, 32, 128),
            "default_value": 64,
        },
        {
            "type": HyperparamType.Model,
            "name": "hid_channels",
            "values_to_test": (16, 32, 128, 256),
            "default_value": 64,
        },
        {
            "type": HyperparamType.Trainer,
            "name": "lr",
            "values_to_test": [10**i for i in range(-4, 3)],
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
