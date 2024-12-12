from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader, random_split

from neural_networks.ai.constants import (
    CHECKPOINTS_PATH,
    EEG_DATA_PATH,
    NUM_OF_CLASSES,
    TENSOR_DEVICE,
    TENSOR_FEATURES_DATA_TYPE,
    TENSOR_LABELS_DATA_TYPE,
    TENSORBOARD_LOGS_PATH,
)
from neural_networks.ai.dataset.eeg_dataset import EEGDataset
from neural_networks.ai.trainer.config import (
    NUM_OF_WORKERS,
    TRAIN_SET_PERCENTAGE,
    VALIDATION_SET_PERCENTAGE,
)
from neural_networks.ai.trainer.typings import HyperparamSetting, HyperparamType, Param, RunConfig
from data_extractor.data_extractor import extract_X_y_from_df, load_df

FrozenParams = tuple[tuple[str, Any], ...]

FBCNET_MODEL_NAME = "FBCNet"


class SingleParamSearch:
    def __init__(self, run_config: RunConfig):
        self._model_name = run_config["model_name"]
        self._dataset_factory = run_config["dataset_factory"]
        self._model_factory = run_config["model_factory"]
        self._trainer_factory = run_config["trainer_factory"]

        self._params = self._prepare_param_values(run_config["params"])

        self._data_cache: dict[
            FrozenParams,
            tuple[Tensor, Tensor],
        ] = {}
        self._dataset_cache: dict[tuple[FrozenParams, FrozenParams], EEGDataset] = {}
        self._already_tested_params: set[FrozenParams] = set()
        self._run_id = self._generate_run_id()
        self._counter = 1

        self._logger = run_config["logger_factory"](
            self._model_name,
            self._run_id,
            run_config["run_name"],
        )

    def get_run_id(self) -> str:
        return self._run_id

    def _prepare_param_values(
        self, params: Sequence[HyperparamSetting]
    ) -> Sequence[HyperparamSetting]:
        return [
            {**param, "values_to_test": self._add_defaults_to_values_and_sort(param)}
            for param in params
        ]

    def _add_defaults_to_values_and_sort(
        self,
        param: HyperparamSetting,
    ) -> Sequence[Any]:
        values = set(param["values_to_test"])
        values.add(param["default_value"])

        return sorted(values)

    def _generate_run_id(self) -> str:
        return str(int(datetime.now().timestamp()))

    def search(self) -> None:
        for param in self._params:
            self._search_for_one_param(param)

    def _search_for_one_param(self, param: HyperparamSetting) -> None:
        for param_value_to_test in param["values_to_test"]:
            self._search_for_value_of_changing_param(
                {
                    "name": param["name"],
                    "type": param["type"],
                    "value": param_value_to_test,
                }
            )

    def _search_for_value_of_changing_param(self, param_to_test: Param) -> None:
        other_param_values = self._get_default_values_from_other_params(
            param_to_test["name"]
        )
        all_params = [param_to_test, *other_param_values]

        self._search_for_param_set(all_params)

    def _get_default_values_from_other_params(
        self, ignored_param_name: str
    ) -> Sequence[Param]:
        return [
            {
                "name": param["name"],
                "type": param["type"],
                "value": param["default_value"],
            }
            for param in self._params
            if param["name"] != ignored_param_name
        ]

    def _search_for_param_set(self, params: Sequence[Param]) -> None:
        if self._check_and_add_params_already_tested(params):
            return

        X, y = self._prepare_or_get_cached_data(params)
        dataset = self._prepare_or_get_cached_dataset(X, y, params)
        data_loaders = self._prepare_data_loaders_for_dataset(params, dataset)
        model = self._prepare_model(params)
        trainer = self._prepare_trainer_for_model(params, model)
        model_checkpoint = self._prepare_model_checkpoint(params)

        self._fit_trainer(params, model, trainer, model_checkpoint, data_loaders)

    def _check_and_add_params_already_tested(self, params: Sequence[Param]) -> bool:
        all_params = {param["name"]: param["value"] for param in params}
        frozen_all_params = self._freeze_params(all_params)
        if frozen_all_params in self._already_tested_params:
            return True

        self._already_tested_params.add(frozen_all_params)
        return False

    def _prepare_or_get_cached_data(
        self, params: Sequence[Param]
    ) -> tuple[Tensor, Tensor]:
        data_params = self._get_kwargs_from_params_of_type(params, HyperparamType.Data)
        frozen_data_params = self._freeze_params(data_params)

        data = self._data_cache.get(frozen_data_params)
        if data is None:
            data = self._load_data(data_params)
            self._data_cache[frozen_data_params] = data

        return data

    def _load_data(self, data_params: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        X, y = self._load_data_with_freqs(data_params)

        return torch.tensor(
            X, dtype=TENSOR_FEATURES_DATA_TYPE, device=TENSOR_DEVICE
        ), torch.tensor(y, dtype=TENSOR_LABELS_DATA_TYPE, device=TENSOR_DEVICE)

    def _load_data_with_freqs(
        self, data_params: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        freqs = data_params.get("freqs")
        if freqs is None:
            freqs = [(1, 50)]

        Xs = []
        for low, high in freqs:
            df = load_df(
                EEG_DATA_PATH,
                **{k: v for k, v in data_params.items() if k != "freqs"},
                lfreq=low,
                hfreq=high,
            )
            df = df.query("desired_answer == answer and data_type in ['REAL', 'FAKE']")
            df["label"] = df.apply(lambda x: 1 if x.block_no in [1, 3] else 0, axis=1)
            X_np, y_np = extract_X_y_from_df(df)
            Xs.append(X_np)
        if len(Xs) == 1 and self._model_name != FBCNET_MODEL_NAME:
            Xs = Xs[0]

        X = np.array(Xs)
        if self._model_name == FBCNET_MODEL_NAME:
            X = X.transpose(1, 0, 2, 3)

        return X, y_np

    def _prepare_or_get_cached_dataset(
        self, X: Tensor, y: Tensor, params: Sequence[Param]
    ) -> EEGDataset:
        data_params = self._get_kwargs_from_params_of_type(params, HyperparamType.Data)
        dataset_params = self._get_kwargs_from_params_of_type(
            params, HyperparamType.Dataset
        )
        frozen_data_params = self._freeze_params(data_params)
        frozen_dataset_params = self._freeze_params(dataset_params)

        cache_key = (frozen_data_params, frozen_dataset_params)
        dataset = self._dataset_cache.get(cache_key)
        if dataset is None:
            dataset = self._dataset_factory(X, y, **dataset_params)
            self._dataset_cache[cache_key] = dataset

        return dataset

    def _get_kwargs_from_params_of_type(
        self, params: Sequence[Param], param_type: HyperparamType
    ) -> Mapping[str, Any]:
        params_for_type = [param for param in params if param["type"] == param_type]

        return {param["name"]: param["value"] for param in params_for_type}

    def _freeze_params(self, params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted(params.items()))

    def _prepare_data_loaders_for_dataset(
        self, params: Sequence[Param], dataset: EEGDataset
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        data_loader_params = self._get_kwargs_from_params_of_type(
            params, HyperparamType.DataLoader
        )

        train_dataset, validation_dataset, test_dataset = random_split(
            dataset, self._get_train_val_test_proportion()
        )

        return (
            DataLoader(
                train_dataset,
                shuffle=True,
                num_workers=NUM_OF_WORKERS,
                **data_loader_params,
            ),
            DataLoader(
                validation_dataset,
                shuffle=False,
                num_workers=NUM_OF_WORKERS,
                **data_loader_params,
            ),
            DataLoader(
                test_dataset,
                shuffle=False,
                num_workers=NUM_OF_WORKERS,
                **data_loader_params,
            ),
        )

    def _get_train_val_test_proportion(self) -> tuple[float, float, float]:
        return (
            TRAIN_SET_PERCENTAGE,
            VALIDATION_SET_PERCENTAGE,
            1 - (TRAIN_SET_PERCENTAGE + VALIDATION_SET_PERCENTAGE),
        )

    def _prepare_model(self, params: Sequence[Param]) -> Module:
        model_params = self._get_kwargs_from_params_of_type(
            params, HyperparamType.Model
        )

        return self._model_factory(**model_params)

    def _prepare_trainer_for_model(
        self, params: Sequence[Param], model: Module
    ) -> LightningModule:
        trainer_params = self._get_kwargs_from_params_of_type(
            params, HyperparamType.Trainer
        )
        all_trainer_params = self._get_all_trainer_params(trainer_params, model)

        return self._trainer_factory(**all_trainer_params)

    def _get_all_trainer_params(
        self, trainer_params: Mapping[str, Any], model: Module
    ) -> Mapping[str, Any]:
        return {
            "model": model,
            "num_classes": NUM_OF_CLASSES,
            "accelerator": self._translate_device_to_accelerator(TENSOR_DEVICE),
            **trainer_params,
        }

    def _translate_device_to_accelerator(self, device: str) -> str:
        if device == "cuda":
            return "gpu"

        return device

    def _prepare_model_checkpoint(self, params: Sequence[Param]) -> ModelCheckpoint:
        model_checkpoint_params = self._get_kwargs_from_params_of_type(
            params, HyperparamType.ModelCheckpoint
        )

        return ModelCheckpoint(
            dirpath=CHECKPOINTS_PATH,
            filename=self._get_best_checkpoint_filename(),
            save_top_k=1,
            **model_checkpoint_params,
        )

    def _get_best_checkpoint_filename(self) -> str:
        subrun_id = self._get_subrun_id()
        best_checkpoint_filename = f"{subrun_id}-best-checkpoint"
        return best_checkpoint_filename

    def _get_subrun_id(self) -> str:
        return f"{self._run_id}-{self._counter}"

    def _fit_trainer(
        self,
        params: Sequence[Param],
        model: Module,
        trainer: LightningModule,
        model_checkpoint: ModelCheckpoint,
        data_loaders: tuple[DataLoader, DataLoader, DataLoader],
    ) -> None:
        fit_params = self._get_kwargs_from_params_of_type(params, HyperparamType.Fit)

        train_data_loader, validation_data_loader, test_data_loader = data_loaders

        trainer.fit(
            train_data_loader,
            validation_data_loader,
            enable_progress_bar=True,
            enable_model_summary=True,
            logger=TensorBoardLogger(
                TENSORBOARD_LOGS_PATH / self._model_name, name=self._get_subrun_id()
            ),
            callbacks=model_checkpoint,
            **fit_params,
        )

        trainer_params = self._get_kwargs_from_params_of_type(
            params, HyperparamType.Trainer
        )
        all_trainer_params = self._get_all_trainer_params(trainer_params, model)
        best_model = trainer.__class__.load_from_checkpoint(
            CHECKPOINTS_PATH / f"{self._get_best_checkpoint_filename()}.ckpt",
            **all_trainer_params,
        )

        score = best_model.test(test_data_loader)
        self._logger.log(
            {
                **score[0],
                "params": [{**param, "type": param["type"].name} for param in params],
            },
            self._counter,
        )
        self._counter += 1
