from abc import ABC, abstractmethod
from typing import Any, Mapping


class Logger(ABC):
    @abstractmethod
    def __init__(self, model_name: str, run_id: str, run_name: str) -> None:
        pass

    @abstractmethod
    def log(self, message: Mapping[Any, Any], counter: int) -> None:
        pass
