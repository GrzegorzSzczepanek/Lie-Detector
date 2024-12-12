import mne
import numpy as np
import os
import pandas as pd

first_block_path = "EEG_ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
second_block_path = "EEG_ExperimentBlock.DECEITFUL_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
third_block_path = "EEG_ExperimentBlock.DECEITFUL_RESPONSE_TO_FAKE_IDENTITY_raw.fif"
fourth_block_path = "EEG_ExperimentBlock.HONEST_RESPONSE_TO_FAKE_IDENTITY_raw.fif"

blocks = [first_block_path, second_block_path, third_block_path, fourth_block_path]

def _desired_answer(record):
    
    if record["data_type"] == "RANDO":
        return "NO"
    
    if record["data_type"] == "CELEBRITY":
        return "YES"
    
    if record["data_type"] == "REAL":
        if record["block_no"] == 1:
            return "YES"
        return "NO"
    
    if record["data_type"] == "FAKE":
        if record["block_no"] == 3:
            return "YES"
        return "NO"
    
def _extract_eeg_epoch(eeg, event, baseline, tmin, tmax):
    epoch = mne.Epochs(
        raw = eeg,
        events = np.reshape(event, (1, 3)),
        tmin = tmin,
        tmax = tmax,
        baseline = baseline
    )
    return epoch

def _load_one_subject(dir_path, lfreq, hfreq, notch_filter, baseline, tmin, tmax):
    records = []
    for i, block in enumerate(blocks):
        eeg = mne.io.read_raw_fif(os.path.join(dir_path, block))
        eeg.load_data()
        eeg = eeg.pick_types(eeg=True, stim=False, eog=False, exclude="bads") 
        eeg.apply_function(lambda x: x * 10 ** -6)
        eeg.filter(l_freq=lfreq, h_freq=hfreq) 
        eeg.notch_filter(notch_filter)
        events, event_dict = mne.events_from_annotations(eeg)
        event_dict = {v: k for k, v in event_dict.items()}
        for j in range(0, len(events), 2):
            if j+1 >= len(events):
                continue

            event_1 = events[j]
            event_2 = events[j+1]

            anno_1 = event_dict[event_1[2]]
            anno_2 = event_dict[event_2[2]]

            if not "ParticipantResponse" in anno_2 or not "PersonalDataField" in anno_1:
                continue

            record = {
                "subject": dir_path.split("/")[-1],
                "block_no": i+1,
                "duration": (event_2[0] - event_1[0])*0.004,
                'field': anno_1.split(":")[0].split(".")[-1],
                "data_type": anno_1.split(":")[-1].split(".")[-1],
                "answer": anno_2.split(".")[-1],
                "eeg": _extract_eeg_epoch(eeg, event_1, baseline, tmin, tmax)
            }

            record["desired_answer"] = _desired_answer(record)
            records.append(record)

    return records

def load_df(root_path, lfreq = 1, hfreq = 50, notch_filter = [50, 100], baseline = (0,0), tmin = 0, tmax = 1):
    records = []
    for dir in os.listdir(root_path):
        if dir == ".DS_Store":
            continue
        records += _load_one_subject(os.path.join(root_path, dir), lfreq, hfreq, notch_filter, baseline, tmin, tmax)
    return pd.DataFrame.from_records(records)

def extract_X_y_from_df(df: pd.DataFrame):
    if not 'label' in df.columns:
        raise ValueError("Add column label")
    if not 'eeg' in df.columns:
        raise ValueError("No column named eeg")
    X, y = [], []

    for i, row in df.iterrows():
        eeg = row['eeg'].get_data()
        if eeg.shape[0] == 0:
            continue
        X.append(eeg)
        y.append(row['label'])

    return np.row_stack(X), np.array(y)
