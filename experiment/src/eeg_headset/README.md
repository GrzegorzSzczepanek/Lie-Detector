# EEGHeadset Class

## Overview

The `EEGHeadset` class is designed to interface with the BrainAccess EEG device, manage data acquisition, and annotate and save EEG data during an experiment. This class handles the connection to the EEG headset, starts and stops data acquisition, and annotates the data with relevant information about the experiment.

## Components

### Initialization

The `EEGHeadset` class is initialized with a participant ID, which is used to create a directory for saving the EEG data.

```python
from .eeg_headset import EEGHeadset

eeg_headset = EEGHeadset(participant_id="12345")
```

### Key Methods

#### [`_connect()`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A31%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition")

Establishes a connection to the EEG headset.

```python
eeg_headset._connect()
```

#### [`_disconnect()`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A43%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition")

Disconnects from the EEG headset.

```python
eeg_headset._disconnect()
```

#### [`start_block(experiment_block)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A57%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition")

Starts a new block of the experiment and begins data acquisition.

```python
from ..common import ExperimentBlock

eeg_headset.start_block(ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY)
```

#### [`stop_and_save_block()`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A73%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition")

Stops the current block and saves the acquired EEG data to a file.

```python
eeg_headset.stop_and_save_block()
```

#### [`annotate_data_shown(shown_data_field, shown_data_type)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A95%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition")

Annotates the EEG data with information about the data shown to the participant.

```python
from ..personal_data import PersonalDataField, PersonalDataType

eeg_headset.annotate_data_shown(PersonalDataField.NAME, PersonalDataType.REAL)
```

#### [`annotate_response(participant_response)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A116%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition")

Annotates the EEG data with the participant's response.

```python
from ..common import ParticipantResponse

eeg_headset.annotate_response(ParticipantResponse.YES)
```

## Configuration

The configuration for the EEG headset is defined in [`eeg_config.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_config.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "/Users/gsk/documents/projects/Lie-Detector/experiment/src/eeg_headset/eeg_config.py"). Key configuration parameters include:

- **Channel Mapping**

:

- [`BRAINACCESS_EXTENDED_KIT_16_CHANNEL`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_config.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A2%2C%22character%22%3A0%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition"): Mapping of the 16 EEG channels.

- **Data Folder Path**:
  - [`DATA_FOLDER_PATH`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_config.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A22%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A10%2C%22character%22%3A24%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition"): Path to the folder where EEG data will be saved.
- **Used Device**:
  - [`USED_DEVICE`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_config.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A24%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A10%2C%22character%22%3A42%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition"): The EEG device configuration to be used.

## Example Usage

1. **Initialization**:

   - Initialize the [`EEGHeadset`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2F__init__.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A0%2C%22character%22%3A25%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A13%2C%22character%22%3A6%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") with a participant ID.

   ```python
   eeg_headset = EEGHeadset(participant_id="12345")
   ```

2. **Start a Block**:

   - Start a new block of the experiment and begin data acquisition.

   ```python
   from ..common import ExperimentBlock

   eeg_headset.start_block(ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY)
   ```

3. **Annotate Data**:

   - Annotate the data shown to the participant.

   ```python
   from ..personal_data import PersonalDataField, PersonalDataType

   eeg_headset.annotate_data_shown(PersonalDataField.NAME, PersonalDataType.REAL)
   ```

   - Annotate the participant's response.

   ```python
   from ..common import ParticipantResponse

   eeg_headset.annotate_response(ParticipantResponse.YES)
   ```

4. **Stop and Save Block**:

   - Stop the current block and save the acquired EEG data.

   ```python
   eeg_headset.stop_and_save_block()
   ```

5. **Disconnect**:

   - Disconnect from the EEG headset.

   ```python
   eeg_headset._disconnect()
   ```

## Detailed Description

### Initialization

The [`EEGHeadset`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2F__init__.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A0%2C%22character%22%3A25%7D%7D%2C%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A13%2C%22character%22%3A6%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") class is initialized with a participant ID, which is used to create a directory for saving the EEG data. The BrainAccess library is initialized, and directories for saving data are created if they do not exist.

### Connection Management

The [`_connect`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A31%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") method establishes a connection to the EEG headset using the BrainAccess library. The [`_disconnect`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A43%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") method disconnects from the EEG headset and ensures that no block is currently being recorded.

### Data Acquisition

The [`start_block`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A57%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") method starts a new block of the experiment and begins data acquisition. The [`stop_and_save_block`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A73%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") method stops the current block, retrieves the acquired EEG data, and saves it to a file.

### Data Annotation

The [`annotate_data_shown`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A95%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") method annotates the EEG data with information about the data shown to the participant, such as the field (e.g., name, birth date) and the type of data (e.g., real, fake). The [`annotate_response`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fgsk%2Fdocuments%2Fprojects%2FLie-Detector%2Fexperiment%2Fsrc%2Feeg_headset%2Feeg_headset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A116%2C%22character%22%3A8%7D%7D%5D%2C%229ed467e3-ea9c-4f7e-a804-0b33eafd0cce%22%5D "Go to definition") method annotates the EEG data with the participant's response (e.g., yes, no).

### Directory Management

The `_create_dir_if_not_exist` "Go to definition" method ensures that the specified directory exists, creating it if necessary.

## Configuration File: [`eeg_config.py`](./eeg_config.py)

```python
# eeg_config.py

BRAINACCESS_EXTENDED_KIT_16_CHANNEL = {
    0: "Fp1",
    1: "Fp2",
    2: "F3",
    3: "F4",
    4: "C3",
    5: "C4",
    6: "P3",
    7: "P4",
    8: "O1",
    9: "O2",
    10: "T3",
    11: "T4",
    12: "T5",
    13: "T6",
    14: "F7",
    15: "F8",
}

# Path to the folder where EEG data will be saved
DATA_FOLDER_PATH = "eeg_data"

# Device configuration to be used
USED_DEVICE = BRAINACCESS_EXTENDED_KIT_16_CHANNEL
```
