import torch
from mne.time_frequency.psd import psd_array_welch
from torch import Tensor

from neural_networks.ai.constants import (
    MAX_FREQUENCY,
    MICROVOLTS_IN_VOLT,
    SAMPLING_RATE,
    TENSOR_DEVICE,
    WELCH_SEGMENT_SIZE,
)
from neural_networks.ai.dataset.eeg_dataset import EEGDataset


class PSDDataset(EEGDataset):
    def __init__(self, data: Tensor, labels: Tensor) -> None:
        psd_data = self._transform_data_to_psd(data)

        super().__init__(
            psd_data,
            labels,
        )

    def _transform_data_to_psd(self, data: Tensor) -> Tensor:
        data_in_microvolts = data * MICROVOLTS_IN_VOLT

        psds, _ = psd_array_welch(
            data_in_microvolts.cpu().numpy(),
            sfreq=SAMPLING_RATE,
            n_per_seg=WELCH_SEGMENT_SIZE,
            fmax=MAX_FREQUENCY,
        )

        return torch.from_numpy(psds).to(TENSOR_DEVICE)
