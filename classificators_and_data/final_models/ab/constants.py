import numpy as np
import torch

SAMPLING_RATE = 250
NUM_OF_CLASSES = 2
NUM_OF_ELECTRODES = 16
STANDARD_BAND_TO_FREQUENCY_RANGE = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 31),
    "gamma": (31, 49),
}

NUMPY_DATA_TYPE = np.float32
X_TENSOR_DATA_TYPE = torch.float32
Y_TENSOR_DATA_TYPE = torch.long

MICROVOLTS_IN_VOLT = 10**6
