import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from neural_networks.ai.constants import JSON_LOGS_PATH
from neural_networks.ai.logger.logger import Logger


class JSONLogger(Logger):
    def __init__(self, model_name: str, run_id: str, run_name: str) -> None:
        self._model_name = model_name
        self._run_id = run_id
        self._run_name = run_name
        self._dir_path = self._create_dir_if_missing()

    def _create_dir_if_missing(self) -> Path:
        dir_path = JSON_LOGS_PATH / self._model_name

        dir_path.mkdir(parents=True, exist_ok=True)

        return dir_path

    def log(self, message: Mapping[Any, Any], counter: int) -> None:
        file_path = self._dir_path / f"{self._run_id}-{counter}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                self._create_log_json(message, counter),
                f,
                indent=4,
                sort_keys=True,
                ensure_ascii=False,
            )

    def _create_log_json(
        self, message: Mapping[Any, Any], counter: int
    ) -> Mapping[Any, Any]:
        timestamp = datetime.now().isoformat()

        return {
            "run_id": self._run_id,
            "run_name": self._run_name,
            "run_counter": counter,
            "model": self._model_name,
            "timestamp": timestamp,
            **message,
        }
