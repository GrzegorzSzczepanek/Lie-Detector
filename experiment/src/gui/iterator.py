from typing import Generic, Sequence, TypeVar

T = TypeVar("T")


class Iterator(Generic[T]):
    def __init__(self, seq: Sequence[T]):
        self._pointer = 0
        self._seq = seq

    def has_next(self) -> bool:
        """
        Returns:
            bool: If there are more elements to get.
        """

        return self._pointer < len(self._seq)

    def get_current(self) -> T:
        return self._seq[self._pointer]

    def advance(self) -> None:
        self._pointer += 1
