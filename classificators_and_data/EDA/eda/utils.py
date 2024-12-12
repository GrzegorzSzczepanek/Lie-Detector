import os
from dataclasses import dataclass

import mne
import numpy as np


@dataclass
class EegData:
    epochs: mne.Epochs
    personal_data_field: str
    personal_data_type: str
    response: str
    duration: float

    def to_dict(self):
        return {
            "experiment_block": self.experiment_block,
            "epochs": self.epochs,
            "personal_data_field": self.personal_data_field,
            "personal_data_type": self.personal_data_type,
            "response": self.response,
            "correct_response": self._get_correct_response(),
            "duration": self.duration,
        }

    def set_block(self, block):
        self.experiment_block = block

    def _get_correct_response(self):
        if self.personal_data_type == "CELEBRITY":
            return "YES"

        if self.personal_data_type == "RANDO":
            return "NO"

        match (self.experiment_block, self.personal_data_type):
            case ("HONEST_RESPONSE_TO_TRUE_IDENTITY", "REAL") | (
                "DECEITFUL_RESPONSE_TO_FAKE_IDENTITY",
                "FAKE",
            ):
                return "YES"
            case ("HONEST_RESPONSE_TO_FAKE_IDENTITY", "FAKE") | (
                "DECEITFUL_RESPONSE_TO_TRUE_IDENTITY",
                "REAL",
            ):
                return "NO"
            case _:
                raise ValueError(
                    f"Invalid experiment block/personal data type combination ({self.experiment_block}, {self.personal_data_type})."
                )


def preprocess_eeg_data(
    dir_path: str,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
    notch_filter: list[float] = [50, 100],
) -> list[tuple[mne.io.Raw, np.ndarray, np.ndarray, dict]]:
    """
    Preprocess EEG data for each block.

    Parameters:
    - dir_path (str): Directory path containing the EEG data files.
    - l_freq (float): Low cutoff frequency for band-pass filtering.
    - h_freq (float): High cutoff frequency for band-pass filtering.
    - notch_filter (list[float]): Frequencies for notch filtering.

    Returns:
    - list of tuples containing:
        - raw (mne.io.Raw): Preprocessed raw EEG data.
        - events (np.ndarray): Array of events.
        - events_timestamps (np.ndarray): Array of event timestamps.
        - event_dict (dict): Dictionary of event types and their codes.
    """
    blocks_paths = _get_block_paths(dir_path)
    preprocessed_data = []

    for block_path in blocks_paths:
        raw = _load_data(block_path)
        events_duration = _compute_event_durations(raw)
        raw = _preprocess_raw_data(raw, l_freq, h_freq, notch_filter)
        events, event_dict, events_timestamps = _extract_events(raw, events_duration)

        block = (
            block_path.split("\\")[-1]
            .removeprefix("EEG_ExperimentBlock.")
            .removesuffix("_raw.fif")
        )

        preprocessed_data.append((raw, events, events_timestamps, event_dict, block))

    return preprocessed_data


def process_fixed_time_epochs(
    preprocessed_data: list[tuple[mne.io.Raw, np.ndarray, np.ndarray, dict]],
    tmin: float = 0.0,
    tmax: float = 0.997,
    baseline: tuple[float, float] = (None, None),
) -> tuple[mne.Epochs, dict[int, mne.Epochs], np.ndarray, np.ndarray]:
    """
    Function to process preprocessed EEG data into fixed-time epochs.

    Parameters:
    - preprocessed_data (list): List of preprocessed data tuples.
    - tmin (float): Start time before event for epoching.
    - tmax (float): End time after event for epoching.
    - baseline (tuple[float, float]): Baseline correction period.

    Returns:
    - all_epochs (mne.Epochs): Concatenated epochs from all blocks.
    - epochs_dict (dict[int, mne.Epochs]): Dictionary of individual epochs from each block.
    - X (np.ndarray): Array with shape (n_events, n_channels, n_timepoints).
    - y (np.ndarray): Array of labels corresponding to each event.
    """
    epochs_dict = {}

    for key, (raw, events, events_timestamps, event_dict) in enumerate(
        preprocessed_data, start=1
    ):
        undesired_answer, personality = _determine_conditions(key)
        events, events_timestamps = _discard_undesired_responses(
            events, event_dict, undesired_answer, events_timestamps
        )
        event_id = _extract_desired_values(event_dict, personality)
        label = _determine_honesty(key)
        events, event_id = _rename_events(events, event_id, label)
        epochs = _create_fixed_time_epochs(raw, events, event_id, tmin, tmax, baseline)
        epochs_dict[key] = epochs

    all_epochs = mne.concatenate_epochs(list(epochs_dict.values()))
    X, y = _convert_to_numpy(all_epochs)

    return all_epochs, epochs_dict, X, y


def process_variable_time_epochs(
    preprocessed_data: list[tuple[mne.io.Raw, np.ndarray, np.ndarray, dict]],
    baseline: tuple[float, float] = (None, None),
    constant_values: int = 0,
) -> tuple[dict[int, list[mne.Epochs]], np.ndarray, np.ndarray]:
    """
    Function to process preprocessed EEG data into variable-time epochs.

    Parameters:
    - preprocessed_data (list): List of preprocessed data tuples.
    - baseline (tuple[float, float]): Baseline correction period.
    - constant_values : The values to set the padded values for each axis.

    Returns:
    - epochs_dict (dict[int, list[mne.Epochs]]): Dictionary of all epochs.
    - padded_array_epochs (np.ndarray): Array with shape (n_events, n_channels, n_timepoints).
    - labels (np.ndarray): Array of labels corresponding to each event.
    """
    all_epochs = []

    for key, (raw, events, events_timestamps, event_dict, block) in enumerate(
        preprocessed_data, start=1
    ):
        undesired_answer, personality = _determine_conditions(key)
        events, events_timestamps = _discard_undesired_responses(
            events, event_dict, undesired_answer, events_timestamps
        )
        event_id = _extract_desired_values(event_dict, personality)
        # label = _determine_honesty(key)
        # events, event_id = _rename_events(events, event_id, label)
        epochs = _create_variable_time_epochs(
            raw, events, event_id, events_timestamps, event_dict, baseline=baseline
        )

        for epoch in epochs:
            epoch.set_block(block)

        all_epochs.extend(epochs)

    padded_array_epochs, labels = _pad_epochs(
        constant_values,
        [epoch.epochs for epoch in all_epochs],
    )

    return all_epochs, padded_array_epochs, labels


def _determine_conditions(key: int) -> tuple[str, str]:
    if key in {1, 3}:
        undesired_answer = "NO"
    elif key in {2, 4}:
        undesired_answer = "YES"

    if key in {1, 2}:
        personality = "REAL"
    elif key in {3, 4}:
        personality = "FAKE"

    return undesired_answer, personality


def _determine_honesty(key: int) -> int:
    if key in {1, 4}:
        return 1
    elif key in {2, 3}:
        return 0


def _get_block_paths(dir_path: str) -> list[str]:
    """
    Read paths of blocks in the directory.

    Parameters:
    - dir_path (str): Directory path containing the EEG data files.

    Returns:
    - list[str]: Paths of the four EEG experiment blocks.
    """
    first_block_path = os.path.join(
        dir_path, "EEG_ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
    )
    second_block_path = os.path.join(
        dir_path, "EEG_ExperimentBlock.DECEITFUL_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
    )
    third_block_path = os.path.join(
        dir_path, "EEG_ExperimentBlock.DECEITFUL_RESPONSE_TO_FAKE_IDENTITY_raw.fif"
    )
    fourth_block_path = os.path.join(
        dir_path, "EEG_ExperimentBlock.HONEST_RESPONSE_TO_FAKE_IDENTITY_raw.fif"
    )

    return [first_block_path, second_block_path, third_block_path, fourth_block_path]


def _load_data(file_path: str) -> mne.io.Raw:
    """
    Load raw EEG data.

    Parameters:
    - file_path (str): Path to the EEG data file.

    Returns:
    - mne.io.Raw: Loaded raw EEG data.
    """
    raw = mne.io.read_raw_fif(file_path)
    raw = raw.load_data()

    return raw


def _compute_event_durations(raw: mne.io.Raw) -> np.ndarray:
    """
    Compute event durations from raw EEG data annotations.

    Parameters:
    - raw (mne.io.Raw): Raw EEG data.

    Returns:
    - np.ndarray: Array of event durations.
    """
    annotations_time = raw.annotations.onset
    end_event = annotations_time[1:]  # Get the start times of the next events
    end_event = np.append(
        end_event, 1
    )  # Add a helper value at the end to enable calculation of the duration for the last event
    events_duration = (
        end_event - annotations_time
    )  # annotations_time represents start_time

    return events_duration


def _preprocess_raw_data(
    raw: mne.io.Raw, l_freq: float, h_freq: float, notch_filter: float
) -> mne.io.Raw:
    """
    Preprocess raw EEG data by applying band-pass and notch filters.

    Parameters:
    - raw (mne.io.Raw): Raw EEG data.
    - l_freq (float): Low cutoff frequency for band-pass filtering.
    - h_freq (float): High cutoff frequency for band-pass filtering.
    - notch_filter (float): Frequency for notch filtering.

    Returns:
    - mne.io.Raw: Preprocessed EEG data.
    """
    raw = raw.pick_types(eeg=True, stim=False, eog=False, exclude="bads")
    raw.apply_function(lambda x: x * 10**-6)
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw.notch_filter(notch_filter)

    return raw


def _extract_events(
    raw_file: mne.io.Raw, events_duration: np.ndarray
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """
    Extract events from raw EEG data annotations.

    Parameters:
    - raw_file (mne.io.Raw): Raw EEG data.
    - events_duration (np.ndarray): Array of event durations.

    Returns:
    - tuple[np.ndarray, dict[str, int], np.ndarray]: Events, event dictionary, and event timestamps.
    """
    events, event_dict = mne.events_from_annotations(raw_file)
    start_event = events[:, 0]
    events_timestamps = np.column_stack((start_event, events_duration))

    return events, event_dict, events_timestamps


def _discard_undesired_responses(
    events: np.ndarray,
    event_dict: dict[str, int],
    undesired_answer: str,
    events_timestamp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Discard events corresponding to undesired responses ('TIMEOUT' or specified answer).

    Parameters:
    - events (np.ndarray): Array of events.
    - event_dict (dict[str, int]): Dictionary of event types and their codes.
    - undesired_answer (str): Undesired response ('YES' or 'NO').
    - events_timestamp (np.ndarray): Array of event timestamps.

    Returns:
    - tuple[np.ndarray, np.ndarray]: Filtered events and event timestamps.
    """

    undesired_responses = []
    indices_to_remove = []

    for key in undesired_responses:
        if key in event_dict:
            event_index = event_dict[key]
            key_indices = np.where(events[:, 2] == event_index)[0]
            indices_to_remove.extend(key_indices)
            indices_to_remove.extend(key_indices - 1)

    unique_indices_to_remove = np.unique(indices_to_remove)
    filtered_events = events[
        ~np.isin(np.arange(events.shape[0]), unique_indices_to_remove)
    ]
    filtered_events_timestamp = events_timestamp[
        ~np.isin(np.arange(events_timestamp.shape[0]), unique_indices_to_remove)
    ]

    return filtered_events, filtered_events_timestamp


def _extract_desired_values(event_dict: dict[str, int], personality: str) -> list[int]:
    """
    Extract the event IDs corresponding to the desired personality type.

    Parameters:
    - event_dict (dict[str, int]): Dictionary of event types and their codes.
    - personality (str): Desired personality type ('REAL' or 'FAKE').

    Returns:
    - list[int]: List of event IDs corresponding to the desired personality type.
    """
    real_values = []
    for key, value in event_dict.items():
        if "PersonalDataType" in key:
            real_values.append(value)

    return real_values


def _rename_events(
    events: np.ndarray, event_id: list[int], label: int
) -> tuple[np.ndarray, list[int]]:
    """
    Rename events to categorize them as honest or deceitful responses.

    Parameters:
    - events (np.ndarray): Array of events.
    - event_id (list[int]): List of desired event IDs.
    - label (int): Label for categorizing responses (1 for honest, 0 for deceitful).

    Returns:
    - tuple[np.ndarray, list[int]]: Renamed events and updated event IDs.
    """
    for i in range(len(events)):
        event = events[i]
        if event[2] in event_id:
            events[i, 2] = label
        else:
            events[i, 2] = 100  # for no important event
    event_id = [0, 1]
    return events, event_id


def _pad_epochs(
    constant_values: int, epochs_list: list[mne.Epochs]
) -> tuple[list[mne.Epochs], np.ndarray, np.ndarray]:
    """
    Pad epochs to ensure they have the same length for concatenation.

    Parameters:
    - constant_values : The values to set the padded values for each axis.
    - epochs_list (list[mne.Epochs]): List of epochs.

    Returns:
    - np.ndarray, np.ndarray]: data array (n_events, n_channels, n_timepoints) and labels array.
    """
    if not epochs_list:
        raise ValueError("The epochs list is empty.")

    max_length = max([len(epoch.times) for epoch in epochs_list])

    all_padded_array_epochs = []
    all_labels = []

    for epochs in epochs_list:
        current_length = len(epochs.times)
        if current_length < max_length:
            pad_amount = max_length - current_length
            pad_width = ((0, 0), (0, 0), (0, pad_amount))
            padded_epoch_array = np.pad(
                epochs.get_data(),
                pad_width,
                mode="constant",
                constant_values=constant_values,
            )
            label = epochs.events[:, -1]
            all_padded_array_epochs.append(padded_epoch_array)
            all_labels.append(label)

        else:
            epoch_array = epochs.get_data()
            all_padded_array_epochs.append(epoch_array)
            label = epochs.events[:, -1]
            all_labels.append(label)

    padded_epochs_array = np.concatenate(all_padded_array_epochs, axis=0)
    all_events_array = np.concatenate(all_labels, axis=0)

    return padded_epochs_array, all_events_array


def _create_variable_time_epochs(
    raw_file: mne.io.Raw,
    events: np.ndarray,
    event_id: list[int],
    event_timestamp: np.ndarray,
    event_dict,
    baseline: tuple[float, float],
) -> list[mne.Epochs]:
    """
    Create variable-time epochs for each event.

    Parameters:
    - raw_file (mne.io.Raw): Raw EEG data.
    - events (np.ndarray): Array of events.
    - event_id (list[int]): List of desired event IDs.
    - event_timestamp (np.ndarray): Array of event timestamps.
    - baseline (tuple[float, float]): Baseline correction period.

    Returns:
    - list[mne.Epochs]: List of variable-time epochs.
    """
    epochs_list = []

    for event_idx, event in enumerate(events):
        if event[2] in event_id:
            event_id_unique = event[0]
            single_event_id = event[2]
            for event2 in event_timestamp:
                if event2[0] == event_id_unique:
                    tmin = 0
                    tmax = event2[1]
                    event = np.reshape(event, (1, 3))
                    temp_epochs = mne.Epochs(
                        raw_file,
                        event,
                        single_event_id,
                        tmin,
                        tmax,
                        baseline=baseline,
                        preload=True,
                    )

                    event_anno = event[0][2]
                    field_str, dtype_str = [
                        k for k, v in event_dict.items() if v == event_anno
                    ][0].split(": ")
                    field = field_str.removeprefix("PersonalDataField.")
                    dtype = dtype_str.removeprefix("PersonalDataType.")
                    response_anno = events[event_idx + 1][2]
                    response_str = [
                        k for k, v in event_dict.items() if v == response_anno
                    ][0].removeprefix("ParticipantResponse.")

                    epochs_list.append(
                        EegData(temp_epochs, field, dtype, response_str, tmax)
                    )

    return epochs_list


def _create_fixed_time_epochs(
    raw_file: mne.io.Raw,
    events: np.ndarray,
    event_id: list[int],
    tmin: float,
    tmax: float,
    baseline: tuple[float, float],
) -> mne.Epochs:
    """
    Create fixed-time epochs for each event.

    Parameters:
    - raw_file (mne.io.Raw): Raw EEG data.
    - events (np.ndarray): Array of events.
    - event_id (list[int]): List of desired event IDs.
    - tmin (float): Start time before event for epoching.
    - tmax (float): End time after event for epoching.
    - baseline (tuple[float, float]): Baseline correction period.

    Returns:
    - mne.Epochs: Fixed-time epochs.
    """
    epochs = mne.Epochs(
        raw_file,
        events,  # created based on all events
        event_id,
        tmin,  # the time relative to each event at which to start each epoch
        tmax,
        preload=True,
        baseline=baseline,
        on_missing="warn",
    )

    return epochs


def _convert_to_numpy(all_epoches):
    """
    Convert epochs to numpy arrays.

    Parameters:
    - all_epochs (mne.Epochs): Concatenated epochs.

    Returns:
    - tuple[np.ndarray, np.ndarray]: Data array (n_events, n_channels, n_timepoints) and labels array.
    """
    X = all_epoches.get_data()  # Shape: (n_epochs, n_channels, n_times)
    y = all_epoches.events[:, -1]

    return X, y
