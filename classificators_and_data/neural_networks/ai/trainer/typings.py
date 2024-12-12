from enum import Enum, auto
from typing import Any, Callable, Mapping, Sequence, TypedDict

from torch import Tensor
from torch.nn.modules.module import Module
from torcheeg.trainers import ClassifierTrainer

from neural_networks.ai.dataset.eeg_dataset import EEGDataset
from neural_networks.ai.logger.logger import Logger


class HyperparamType(Enum):
    Data = auto()
    Dataset = auto()
    DataLoader = auto()
    Model = auto()
    ModelCheckpoint = auto()
    Trainer = auto()
    Fit = auto()


class HyperparamSetting(TypedDict):
    name: str
    type: HyperparamType
    values_to_test: Sequence[Any]
    default_value: Any


class Param(TypedDict):
    name: str
    type: HyperparamType
    value: Any


class RunConfig(TypedDict):
    run_name: str
    model_name: str
    logger_factory: Callable[[str, str, str], Logger]
    dataset_factory: Callable[[Tensor, Tensor], EEGDataset]
    model_factory: Callable[[], Module]
    trainer_factory: Callable[[], ClassifierTrainer]
    params: Sequence[HyperparamSetting]


Hyperparams = Sequence[HyperparamSetting]
Params = Mapping[str, Any]

ModelFactory = Callable[..., Module]
