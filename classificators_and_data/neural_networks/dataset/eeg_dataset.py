from typing import Callable

from torch import Tensor
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(
        self,
        data: Tensor,
        labels: Tensor,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self._data = data
        self._labels = labels
        self._transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        X = (
            self._data[idx]
            if self._transform is None
            else self._transform(self._data[idx])
        )
        y = self._labels[idx]

        return X, y
