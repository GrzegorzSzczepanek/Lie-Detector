from neural_networks.ai.constants import (
    NUM_OF_CLASSES,
    NUM_OF_ELECTRODES,
)
from neural_networks.ai.dataset.microvolt_dataset import MicrovoltDataset
from neural_networks.ai.logger.json_logger import JSONLogger
from neural_networks.ai.trainer.typings import HyperparamType, RunConfig
from torcheeg.trainers import ClassifierTrainer
from torcheeg.models import FBCNet

CONFIG_TO_RUN: RunConfig = {
    "model_name": "FBCNet",
    "run_name": "FBCNet - run 1",
    "dataset_factory": lambda X, y, **kwargs: MicrovoltDataset(X, y, **kwargs),
    "model_factory": lambda **kwargs: FBCNet(
        chunk_size=251,
        in_channels=10,
        stride_factor=1,
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
            "default_value": 1,
        },
        {
            "type": HyperparamType.Data,
            "name": "freqs",
            "values_to_test": (),
            "default_value": tuple(
                (1 if i == 0 else i, i + 5) for i in range(0, 50, 5)
            ),
        },
        {
            "type": HyperparamType.DataLoader,
            "name": "batch_size",
            "values_to_test": (16, 32, 128),
            "default_value": 64,
        },
        {
            "type": HyperparamType.Model,
            "name": "num_S",
            "values_to_test": (16, 64),
            "default_value": 32,
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
