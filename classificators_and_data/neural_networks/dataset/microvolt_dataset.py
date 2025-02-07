from torch import Tensor

from neural_networks.ai.constants import MICROVOLTS_IN_VOLT
from neural_networks.ai.dataset.eeg_dataset import EEGDataset


class MicrovoltDataset(EEGDataset):
    def __init__(self, data: Tensor, labels: Tensor) -> None:
        super().__init__(
            data * MICROVOLTS_IN_VOLT,
            labels,
        )
