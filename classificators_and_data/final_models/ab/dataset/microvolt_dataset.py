from torch import Tensor

from final_models.neural_networks.constants import MICROVOLTS_IN_VOLT
from final_models.neural_networks.dataset.eeg_dataset import EEGDataset


class MicrovoltDataset(EEGDataset):
    def __init__(self, data: Tensor, labels: Tensor) -> None:
        super().__init__(
            data * MICROVOLTS_IN_VOLT,
            labels,
        )
