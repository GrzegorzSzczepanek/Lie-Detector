# IMPORTANT INFO
# This module is not used in final training process due to new data extraction methods.
# It's kept because of potentially useful funcitons for data augmentation and plotting correlations between channels.

import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import os
from collections import defaultdict
import os


def time_shift(epoch: mne.Epochs, shift_max: int) -> np.ndarray:
    """
    Shift the time axis of the epoch data.
    Args:
        epoch (mne.Epochs): Epoch data.
        shift_max (int): Maximum number of time points to shift.
    Returns:
        np.ndarray: Shifted epoch data
    """
    shift = np.random.randint(-shift_max, shift_max)
    shifted_data = np.roll(epoch.get_data(), shift, axis=-1)
    return shifted_data



def time_scaling(epoch: mne.Epochs, scaling_factor: float) -> np.ndarray:
    """
    Scale the time axis of the epoch data.
    Args:
        epoch (mne.Epochs): Epoch data.
        scaling_factor (float): Scaling factor for the time axis.
    Returns:
        np.ndarray: Scaled epoch data
    """



def amplitude_scaling(epoch: mne.Epochs, scaling_factor: float) -> np.ndarray:
    """
    Scale the amplitude of the epoch data.
    Args:
        epoch (mne.Epochs): Epoch data.
        scaling_factor (float): Scaling factor for the amplitude.
    Returns:
        np.ndarray: Scaled epoch data
    """

    return epoch * scaling_factor


def add_noise(epoch: mne.Epochs, noise_level: float) -> np.ndarray:
    """
    Add Gaussian noise to the epoch data.
    Args:
        epoch (mne.Epochs): Epoch data.
        noise_level (float): Standard deviation of the Gaussian noise.
    Returns:
        np.ndarray: Noisy epoch data
    """
    noise = np.random.normal(0, noise_level, epoch.shape)
    return epoch + noise



def create_epochs_for_users(
    files_paths: list[str], output_dir: str, to_ica=False
) -> None:
    """
    Create epochs for each user in the dataset.
    Args:
        files_paths (list[str]): List of file paths to the raw data files.
        output_dir (str): Path to the directory to save the epoch files.
        to_ica (bool): Whether to apply ICA to the raw data.
    Returns:
        None
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in files_paths if f.endswith(".fif")]
    print(files)

    for file_path in files:

        user_id = os.path.basename(os.path.dirname(file_path))


        raw = mne.io.read_raw_fif(file_path, preload=True)
        raw.apply_function(fun=lambda x: x * 10 ** (-6))

        raw.filter(l_freq=1, h_freq=40)
        events, event_dict = mne.events_from_annotations(raw)
        raw.pick_types(eeg=True)





        events, user_events = mne.events_from_annotations(raw)

        selected_event_dict = {
            key: value for key, value in event_dict.items() if key in user_events.keys()
        }

        if to_ica:
            ica = mne.preprocessing.ICA(n_components=16, random_state=97, max_iter=800)
            ica.fit(raw)
            raw_ica = ica.apply(raw)
            epochs = mne.Epochs(
                raw_ica,
                events,
                event_dict,
                tmin=-0.2,
                tmax=0.5,
                baseline=(None, 0),
                preload=True,
            )
            output_filename = (
                f"{user_id}_{os.path.splitext(os.path.basename(file_path))[0]}.epo.fif"
            )
            output_path = os.path.join(output_dir, output_filename)

            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))

            epochs.save(output_path, overwrite=True)
            print(f"Saved epochs for file {file_path} to {output_path}")
        else:
            try:
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id=selected_event_dict,
                    tmin=-0.2,
                    tmax=0.5,
                    baseline=(None, 0),
                )
                output_filename = f"{user_id}_{os.path.splitext(os.path.basename(file_path))[0]}_epo.fif"
                output_path = os.path.join(output_dir, output_filename)

                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))

                epochs.save(output_path, overwrite=True)
                print(f"Saved epochs for file {file_path} to {output_path}")
            except ValueError as e:
                print(f"Error processing file {file_path}: {e}")


def get_file_paths(dir_path: str) -> list[str]:
    """
    Get the list of file paths in a directory.
    Args:
        dir_path (str): Directory path.
    Returns:
        list[str]: List of file paths in the directory.
    """
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:

            file_path = os.path.join(root, file)

            file_paths.append(file_path)

    return file_paths


def organize_files_by_id(file_paths: list[str]) -> dict[str, list[str]]:
    """
    Organize file paths by directory ID.

    Args:
        file_paths (list[str]): A list of file paths.

    Returns:
        dict[str, list[str]]: A dictionary where the key is the directory ID and the value is a list of file paths.
    """
    file_dict = defaultdict(list)

    for path in file_paths:
        dir_name = os.path.basename(os.path.dirname(path))
        file_dict[dir_name].append(path)

    return dict(file_dict)


def calculate_euclidean_distances(
    data: np.ndarray, visualize: bool = False
) -> np.ndarray:
    """
    Calculate the Euclidean distance matrix between EEG channels.
    Args:
        data (np.ndarray): EEG data of shape (n_channels, n_samples).
        visualize (bool): Whether to visualize the distance matrix.
    Returns:
        np.ndarray: Euclidean distance matrix of shape (n_channels, n_channels).
    """
    n_channels = data.shape[0]
    distances = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            distance = euclidean(data[i], data[j])
            distances[i, j] = distance
            distances[j, i] = distance

    if visualize:

        plt.figure(figsize=(10, 8))
        sns.heatmap(distances, annot=False, cmap="viridis")
        plt.title("Euclidean Distance Matrix of EEG Channels")
        plt.xlabel("Channel Index")
        plt.ylabel("Channel Index")
        plt.show()


        print("Euclidean Distance Matrix:\n", distances)
    return distances


def plot_correlation_matrix(
    raw, title="Correlation Matrix", just_corr_matrix_data=False
):
    """
    Plot the correlation matrix for all channels in an mne.Raw object.

    Args:
        raw (mne.io.Raw): The raw EEG data.
    """

    data, times = raw.get_data(return_times=True)
    channel_names = raw.ch_names


    corr_matrix = np.corrcoef(data)


    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        xticklabels=channel_names,
        yticklabels=channel_names,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        square=True,
    )
    plt.title(title)
    plt.show()

    return corr_matrix if just_corr_matrix_data else None
