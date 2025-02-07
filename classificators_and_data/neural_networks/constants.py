from pathlib import Path

import numpy as np
import torch
import torch.cuda as cuda
from frozendict import frozendict

parent_dir = Path(__file__).parent

EEG_DATA_PATH = parent_dir.parent.parent / "eeg_data"
SAVED_DATA_DIR_PATH = parent_dir / "data"
LOGS_PATH = parent_dir / "logs"
MODELS_PATH = parent_dir / "models"
MODEL_CONFIG_PATH = MODELS_PATH / "config_to_run.py"
IDS_TO_RUN_NAMES_FILE_PATH = LOGS_PATH / "ids_to_run_names.txt"
JSON_LOGS_PATH = LOGS_PATH / "json"
CHECKPOINTS_PATH = LOGS_PATH / "checkpoints"
TENSORBOARD_LOGS_PATH = LOGS_PATH / "tensorboard"

IDS_TO_RUN_NAMES_SEPARATOR = "\t"

NUMPY_DATA_TYPE = np.float32
TENSOR_FEATURES_DATA_TYPE = torch.float32
TENSOR_LABELS_DATA_TYPE = torch.long
TENSOR_DEVICE = "cuda" if cuda.is_available() else "cpu"

MICROVOLTS_IN_VOLT = 10**6
STANDARD_BAND_TO_FREQUENCY_RANGE = frozendict(
    {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 14),
        "beta": (14, 31),
        "gamma": (31, 49),
    }
)

SAMPLING_RATE = 250
NUM_OF_CLASSES = 2
NUM_OF_ELECTRODES = 16
NUM_OF_SAMPLES_IN_EPOCH = 250

WELCH_SEGMENT_SIZE = 125
MAX_FREQUENCY = 50
