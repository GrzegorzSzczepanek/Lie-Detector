from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torcheeg.models import FBCNet
from torcheeg.trainers import ClassifierTrainer

from final_models.neural_networks.constants import (
    MICROVOLTS_IN_VOLT,
    NUM_OF_CLASSES,
    NUM_OF_ELECTRODES,
    X_TENSOR_DATA_TYPE,
    Y_TENSOR_DATA_TYPE,
)
from final_models.neural_networks.dataset.microvolt_dataset import MicrovoltDataset
from final_models.neural_networks.typings import Accelerator
from final_models.LieModel import LieModel


class FbcnetFinalModel(LieModel):
    _checkpoints_path = Path(__file__).parent / "logs" / "checkpoints"
    _checkpoint_filename = "best_checkpoint"
    _train_proportion = 0.8
    _best_model_params = {
        "dataset": {},
        "data_loader": {"batch_size": 64},
        "model": {
            "in_channels": 10,
            "chunk_size": 250,
            "num_S": 64,
            "stride_factor": 5,
        },
        "trainer": {"lr": 0.0001},
        "fit": {"max_epochs": 500},
    }
    _fbcnet_transpose_dimensions = (1, 0, 2, 3)

    def __init__(self, accelerator: Accelerator | None = None) -> None:
        self._model = FBCNet(
            num_electrodes=NUM_OF_ELECTRODES,
            num_classes=NUM_OF_CLASSES,
            **self._best_model_params["model"],
        )
        self._accelerator: Accelerator = "cpu" if accelerator is None else accelerator
        self._create_checkpoints_path()

    def _create_checkpoints_path(self) -> None:
        self._checkpoints_path.mkdir(exist_ok=True)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_test_correct_dimensions = X_train.transpose(
            *self._fbcnet_transpose_dimensions
        )
        train_data_loader, validation_data_loader = self._prepare_data_loaders(
            torch.tensor(
                X_test_correct_dimensions,
                dtype=X_TENSOR_DATA_TYPE,
                device="cuda" if self._accelerator == "gpu" else self._accelerator,
            ),
            torch.tensor(
                y_train,
                dtype=Y_TENSOR_DATA_TYPE,
                device="cuda" if self._accelerator == "gpu" else self._accelerator,
            ),
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
            MicrovoltDataset(
                X_train,
                y_train,
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
        X_test_correct_dimensions = X_test.transpose(*self._fbcnet_transpose_dimensions)
        X_tensor = torch.tensor(
            X_test_correct_dimensions * MICROVOLTS_IN_VOLT,
            dtype=X_TENSOR_DATA_TYPE,
            device="cuda" if self._accelerator == "gpu" else self._accelerator,
        )

        y = torch.argmax(self._model(X_tensor), dim=1).cpu().detach().numpy()
        return y

    def determinate_preprocess_config(self) -> dict[Any, Any] | list[dict[Any, Any]]:
        return [
            {
                "lfreq": low,
                "hfreq": high,
                "notch_filter": [50, 100],
                "baseline": (0, 0),
                "tmin": 0,
                "tmax": 0.997,
            }
            for low, high in tuple(
                (1 if i == 0 else i, i + 10) for i in range(0, 100, 10)
            )
        ]

    def determinate_run_info(self) -> dict[str, str]:
        return {
            "person": "AB",
            "model": "FBCNet",
            "additional_info": "79%",
        }
