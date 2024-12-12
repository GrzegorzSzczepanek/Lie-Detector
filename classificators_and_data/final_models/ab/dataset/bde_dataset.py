from typing import Literal
import numpy as np
import torch
from torch import Tensor
from torcheeg.transforms import BandDifferentialEntropy

from final_models.neural_networks.constants import NUMPY_DATA_TYPE, SAMPLING_RATE
from final_models.neural_networks.dataset.eeg_dataset import EEGDataset


class BDEDataset(EEGDataset):
    def __init__(
        self,
        data: Tensor,
        labels: Tensor,
        band_to_frequency_range: dict[str, tuple[int, int]],
        device: Literal["cpu", "cuda"],
    ) -> None:
        bde_data = self.transform_data_to_bde(data, band_to_frequency_range, device)

        super().__init__(
            bde_data,
            labels,
        )

    @staticmethod
    def transform_data_to_bde(
        data: Tensor,
        band_to_frequency_range: dict[str, tuple[int, int]],
        device: Literal["cpu", "cuda"],
    ) -> Tensor:
        bde_transform = BandDifferentialEntropy(
            sampling_rate=SAMPLING_RATE, band_dict=band_to_frequency_range
        )

        transformed_data = []
        for sample in data.cpu().numpy():
            transformed_data.append(bde_transform(eeg=sample)["eeg"])

        np_transformed_data = np.array(transformed_data, dtype=NUMPY_DATA_TYPE)

        return torch.from_numpy(np_transformed_data).to(device)
