import json
from typing import Callable

import pandas as pd

from neural_networks.ai.constants import JSON_LOGS_PATH

COLUMN_ORDER = [
    "model",
    "run_id",
    "run_counter",
    "run_name",
    "timestamp",
    "test_loss",
    "test_accuracy",
    "params",
]


def _load_logs(model_name_condition: Callable[[str], bool]) -> pd.DataFrame:
    all_logs = []

    for model_dir_path in JSON_LOGS_PATH.iterdir():
        if model_name_condition(model_dir_path.stem):
            for log_path in model_dir_path.iterdir():
                with open(log_path, encoding="utf-8") as f:
                    log = json.load(f)

                    all_logs.append(log)

    df = pd.DataFrame(all_logs)
    df["run_counter"] = df["run_counter"].fillna(0).astype("int64")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df[COLUMN_ORDER]

    return df


def load_all_logs() -> pd.DataFrame:
    return _load_logs(lambda _: True)


def load_logs_by_model(model_name_to_select: str) -> pd.DataFrame:
    logs = _load_logs(lambda model_name: model_name == model_name_to_select)

    logs["params"] = logs["params"].map(
        lambda params: {param["name"]: param["value"] for param in params}  # type: ignore
    )
    logs = logs.join(pd.json_normalize(logs["params"].tolist(), max_level=0))
    logs = logs.drop(("params"), axis=1)

    return logs
