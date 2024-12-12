import numpy as np
import torch
from torch import Tensor
from torcheeg.transforms import BandDifferentialEntropy

from neural_networks.ai.constants import NUMPY_DATA_TYPE, SAMPLING_RATE, TENSOR_DEVICE
from neural_networks.ai.dataset.eeg_dataset import EEGDataset


class BDEDataset(EEGDataset):
    def __init__(
        self,
        data: Tensor,
        labels: Tensor,
        band_to_frequency_range: dict[str, tuple[int, int]],
    ) -> None:
        bde_data = self._transform_data_to_bde(data, band_to_frequency_range)

        super().__init__(
            bde_data,
            labels,
        )

    def _transform_data_to_bde(
        self, data: Tensor, band_to_frequency_range: dict[str, tuple[int, int]]
    ) -> Tensor:
        bde_transform = BandDifferentialEntropy(
            sampling_rate=SAMPLING_RATE, band_dict=band_to_frequency_range
        )

        transformed_data = []
        for sample in data.cpu().numpy():
            transformed_data.append(bde_transform(eeg=sample)["eeg"])

        np_transformed_data = np.array(transformed_data, dtype=NUMPY_DATA_TYPE)

        return torch.from_numpy(np_transformed_data).to(TENSOR_DEVICE)
