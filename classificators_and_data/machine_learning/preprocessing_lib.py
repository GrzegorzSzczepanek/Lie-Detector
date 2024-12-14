import numpy as np
import mne 
import pandas as pd
import data_preprocessing.utils as utils


CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "T3",
    "T4",
    "T5",
    "T6",
    "F7",
    "F8",
]

VOLTS_IN_MICROVOLT = 10**-6
LOWPASS_FREQUENCY = 1
HIGHPASS_FREQUENCY = 50
SAMPLING_FREQUENCY = 250
MAX_FREQUENCY = SAMPLING_FREQUENCY // 2
BANDSTOP_FREQUENCY = np.arange(50, MAX_FREQUENCY, 50)

MAX_TIME_AFTER_EVENT_SECONDS = 2

def preprocess_input_data(X: np.ndarray):
    if X.ndim == 3:
        n_samples, n_channels, n_timepoints = X.shape
        X = X.reshape(n_samples, n_channels * n_timepoints)
    return X

def convert_cv_results(cv_results):
    converted_results = {}
    for key, value in cv_results.items():
        if isinstance(value, np.ndarray):
            converted_results[key] = value.tolist()
        else:
            converted_results[key] = value
    return converted_results



def preprocess_data(raw_data):
    raw_data.pick(CHANNELS)
    raw_data.apply_function(fun=lambda x: x * VOLTS_IN_MICROVOLT)
    raw_data.filter(l_freq=LOWPASS_FREQUENCY, h_freq=HIGHPASS_FREQUENCY)
    raw_data.notch_filter(BANDSTOP_FREQUENCY)

def extract_epochs(raw_data):
    events, event_dict = mne.events_from_annotations(raw_data)
    epochs = mne.Epochs(
        raw_data, events, event_id=event_dict, tmax=MAX_TIME_AFTER_EVENT_SECONDS
    )

    before_dropping = epochs.selection
    epochs.drop_bad()
    after_dropping = epochs.selection

    dropped_epochs = np.setdiff1d(before_dropping, after_dropping)

    return epochs, dropped_epochs

def extract_key_from_dict(dct):
    return list(dct.keys())[0]

def is_answer_correct(personal_data, block, answer):
    yes = "ParticipantResponse.YES"
    no = "ParticipantResponse.NO"

    _, personal_data_type = personal_data.split(": ")
    personal_data_type = personal_data_type.removeprefix("PersonalDataType.")

    match (personal_data_type, block):
        case ("CELEBRITY", _):
            return answer == yes
        case ("RANDO", _):
            return answer == no
        case ("REAL", "HONEST_RESPONSE_TO_TRUE_IDENTITY"):
            return answer == yes
        case ("REAL", "DECEITFUL_RESPONSE_TO_TRUE_IDENTITY"):
            return answer == no
        case ("FAKE", "HONEST_RESPONSE_TO_FAKE_IDENTITY"):
            return answer == no
        case ("FAKE", "DECEITFUL_RESPONSE_TO_FAKE_IDENTITY"):
            return answer == yes
        case _:
            raise ValueError(f"Invalid combination: {personal_data_type}, {block}")
        
        
def count_incorrect_answers(epochs, block):
    incorrect_answers = 0

    i = 0
    while i < len(epochs) - 1:
        shown_data = extract_key_from_dict(epochs[i].event_id)
        answer = extract_key_from_dict(epochs[i + 1].event_id)

        if not is_answer_correct(shown_data, block, answer):
            incorrect_answers += 1

        i += 2

    return incorrect_answers


def count_timeouts(epochs):
    timeouts = 0

    i = 0
    while i < len(epochs) - 1:
        event = extract_key_from_dict(epochs[i].event_id)

        if event == "ParticipantResponse.TIMEOUT":
            timeouts += 1

        i += 1

    return timeouts


def create_dataframe(eeg_data_dir):
    data = []
    for participant in eeg_data_dir.iterdir():
        if not participant.is_dir():
            continue
        for block in participant.iterdir():
            raw = mne.io.read_raw_fif(block)
            raw.load_data()

            preprocess_data(raw)
            epochs, dropped_epochs = extract_epochs(raw)

            block_name = block.stem.split(".")[1].removesuffix("_raw")

            data.append(
                {
                    "participant": participant.stem,
                    "block": block_name,
                    "events_count": len(epochs.events),
                    "incorrect_answers_count": count_incorrect_answers(epochs, block_name),
                    "timeout_count": count_timeouts(epochs),
                    "dropped_epochs": dropped_epochs,
                    "epochs": epochs,  
                }
            )
    return pd.DataFrame(data)


def create_dataset(dir_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    
    for dir_path in dir_paths:
        preprocessed_data, all_events_durations = utils.preprocess_eeg_data(dir_path, l_freq=1, h_freq=50, notch_filter=[50, 100])
        epochs_variable_dict, padded_array_epochs, labels, epoches_durations = utils.process_variable_time_epochs(preprocessed_data, baseline=(None, None), constant_values=0)
        all_epochs, epochs_dict, X, y = utils.process_fixed_time_epochs(preprocessed_data, tmin=0, tmax=0.997, baseline=(None, None))
    
        X_list.append(X)
        y_list.append(y)

    X_full = np.concatenate(X_list, axis=0)
    y_full = np.concatenate(y_list, axis=0)
    
    return X_full, y_full

