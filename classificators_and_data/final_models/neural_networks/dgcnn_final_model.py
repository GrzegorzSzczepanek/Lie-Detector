from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torcheeg.models import DGCNN
from torcheeg.trainers import ClassifierTrainer

from final_models.neural_networks.constants import (
    NUM_OF_CLASSES,
    NUM_OF_ELECTRODES,
    STANDARD_BAND_TO_FREQUENCY_RANGE,
    X_TENSOR_DATA_TYPE,
    Y_TENSOR_DATA_TYPE,
)
from final_models.neural_networks.dataset.bde_dataset import BDEDataset
from final_models.neural_networks.typings import Accelerator
from final_models.LieModel import LieModel


class DcgnnFinalModel(LieModel):
    _checkpoints_path = Path(__file__).parent / "logs" / "checkpoints"
    _checkpoint_filename = "best_checkpoint"
    _train_proportion = 0.8
    _best_model_params = {
        "dataset": {"band_to_frequency_range": STANDARD_BAND_TO_FREQUENCY_RANGE},
        "data_loader": {"batch_size": 16},
        "model": {"in_channels": 5, "num_layers": 2, "hid_channels": 32},
        "trainer": {"lr": 0.001},
        "fit": {"max_epochs": 500},
    }

    def __init__(self, accelerator: Accelerator | None = None) -> None:
        self._model = DGCNN(
            num_electrodes=NUM_OF_ELECTRODES,
            num_classes=NUM_OF_CLASSES,
            **self._best_model_params["model"],
        )
        self._accelerator: Accelerator = "cpu" if accelerator is None else accelerator
        self._create_checkpoints_path()

    def _create_checkpoints_path(self) -> None:
        self._checkpoints_path.mkdir(exist_ok=True)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        train_data_loader, validation_data_loader = self._prepare_data_loaders(
            torch.tensor(X_train, dtype=X_TENSOR_DATA_TYPE),
            torch.tensor(y_train, dtype=Y_TENSOR_DATA_TYPE),
        )

        trainer_kwargs = {
            "model": self._model,
            "num_classes": NUM_OF_CLASSES,
            "accelerator": self._accelerator,
            **self._best_model_params["trainer"],
        }
        trainer = ClassifierTrainer(**trainer_kwargs)

        trainer.fit(
            train_data_loader,
            validation_data_loader,
            enable_progress_bar=True,
            enable_model_summary=True,
            callbacks=ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                dirpath=self._checkpoints_path,
                filename=self._checkpoint_filename,
                save_top_k=1,
            ),
            **self._best_model_params["fit"],
        )

        best_model = ClassifierTrainer.load_from_checkpoint(
            self._checkpoints_path / f"{self._checkpoint_filename}.ckpt",
            **trainer_kwargs,
        )
        self._model = best_model

        self._delete_checkpoint()

    def _prepare_data_loaders(
        self, X_train: Tensor, y_train: Tensor
    ) -> tuple[DataLoader, DataLoader]:
        train_dataset, validation_dataset = random_split(
            BDEDataset(
                X_train,
                y_train,
                device="cuda" if self._accelerator == "gpu" else self._accelerator,
                **self._best_model_params["dataset"],
            ),
            (self._train_proportion, 1 - self._train_proportion),
        )

        return (
            DataLoader(
                train_dataset, shuffle=True, **self._best_model_params["data_loader"]
            ),
            DataLoader(
                validation_dataset,
                shuffle=False,
                **self._best_model_params["data_loader"],
            ),
        )

    def _delete_checkpoint(self) -> None:
        (self._checkpoints_path / f"{self._checkpoint_filename}.ckpt").unlink()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_tensor = BDEDataset.transform_data_to_bde(
            torch.tensor(X_test, dtype=X_TENSOR_DATA_TYPE),
            STANDARD_BAND_TO_FREQUENCY_RANGE,
            "cuda" if self._accelerator == "gpu" else self._accelerator,
        )

        y = torch.argmax(self._model(X_tensor), dim=1).cpu().detach().numpy()
        return y

    def determinate_preprocess_config(self) -> dict[Any, Any]:
        return {
            "lfreq": 1,
            "hfreq": 50,
            "notch_filter": [50, 100],
            "baseline": (0, 0),
            "tmin": 0,
            "tmax": 0.997,
        }

    def determinate_run_info(self) -> dict[str, str]:
        return {
            "person": "AB",
            "model": "DGCNN",
            "additional_info": "88%",
        }
