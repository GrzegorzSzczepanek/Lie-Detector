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
    "run_name": "FBCNet - run z num S",
    "dataset_factory": lambda X, y, **kwargs: MicrovoltDataset(X, y, **kwargs),
    "model_factory": lambda **kwargs: FBCNet(
        chunk_size=250,
        in_channels=10,
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
            "default_value": 0.997,
        },
        {
            "type": HyperparamType.Data,
            "name": "freqs",
            "values_to_test": (),
            "default_value": tuple(
                (1 if i == 0 else i, i + 10) for i in range(0, 100, 10)
            ),
        },
        {
            "type": HyperparamType.DataLoader,
            "name": "batch_size",
            "values_to_test": (),
            "default_value": 64,
        },
        {
            "type": HyperparamType.Model,
            "name": "stride_factor",
            "values_to_test": (),
            "default_value": 5,
        },
        {
            "type": HyperparamType.Model,
            "name": "num_S",
            "values_to_test": (128, 256, 512, 1024),
            "default_value": 128,
        },
        {
            "type": HyperparamType.Trainer,
            "name": "lr",
            "values_to_test": (),
            "default_value": 10**-4,
        },
        {
            "type": HyperparamType.Fit,
            "name": "max_epochs",
            "values_to_test": (),
            "default_value": 500,
        },
    ],
}
