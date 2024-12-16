# Personal Data Module

## Purpose and explanation

We employed for types of data that experiment participant has to memorize and answer with yes or no. We made sure data types have nothing in common (based on fileds mentioned in [identity generator](#identitygenerator) section) Data types consisted:

- `Real` - user real data
- `Celebrity` - a well known person that participant knew. Answer for this is always _yes_ and it serves purpose of balancing `yes` and `no` answers to have the amout of labels. Participant was explained to lie that celebrity's data is theirs. It was sort of impersonation.
- `Random` - Opposite of the celebrity. Participant was supposed to always respond that _Random_ data are not theirs.
- `Fake` - Fake data that has no connection to user Real identiry as every other identity we assign to user.

Experiment is separated into four blocks:

- Honest response to true identity (yes answer for real and celebrity, no for Random)
- Deceitful response to true identiry (no answer for real and random, yes for celebrity)
- Honest response to fake identity (no answer for fake and for Random, yes to Celebrity)
- Deceitful response to Fake identiry (yes answer for fake for celebrity, no for random)
  > after second block user real data is replaced ad we give them time to learn fake id.

## Overview

The Personal Data module is designed to manage and generate personal data for participants in a study aimed at detecting lies about one’s own identity through brainwave analysis. This module includes two main components: the `IdentityGenerator` and the `PersonalDataManager`.

## Components

### IdentityGenerator

The `IdentityGenerator` class is responsible for generating and managing identity data for the participant, a fake identity, a celebrity, and a random person. It loads data from YAML files and ensures that the generated identities do not overlap with the participant's real data. Data for every person (real, fake, celebrity, random) consist these fields:

- Name + Surname (as one field)
- Hometown
- Day of birth + month of date (as one field, f.e `1st of the January`)

#### Key Methods

- `get_real()`: Returns the participant's real identity data.
- `get_fake()`: Returns the generated fake identity data.
- `get_celebrity()`: Returns the selected celebrity identity data.
- `get_rando()`: Returns the generated random identity data.
- `get_id()`: Returns the participant's ID.

### PersonalDataManager

The `PersonalDataManager` class manages the flow of personal data during the experiment. It interacts with the EEG headset to annotate data and responses, and it keeps track of the participant's responses.

#### Key Methods

- `start_block(experiment_block, is_practice)`: Starts a new block of the experiment, generating data for practice or test trials.
- `stop_block()`: Stops the current block and saves EEG data if it is not a practice block.
- `has_next()`: Checks if there is more data to get.
- `get_next()`: Retrieves the next piece of personal data.
- `get_feedback_for_practice_participant_response(participant_response)`: Provides feedback on the participant's response during practice trials.
- `register_participant_response(participant_response)`: Registers the participant's response and annotates it in the EEG recording.
- `get_response_counts()`: Returns the counts of correct and incorrect responses for the current block.
- `cleanup()`: Erases user data.

## How It Works

1. **Initialization**:

   - The `PersonalDataManager` initializes the `IdentityGenerator`, which loads and validates the participant's real data, fake data, celebrity data, and random data from YAML files.
   - The `EEGHeadset` is initialized and connected to the participant's EEG device.

2. **Experiment Blocks**:

   - The experiment is divided into four blocks, each with different instructions for the participant.
   - The `PersonalDataManager` starts a block by generating data for practice or test trials.
   - The GUI displays the data, and the participant responds with 'Yes' or 'No'.
   - The `PersonalDataManager` checks the participant's response and provides feedback during practice trials.
   - The `EEGHeadset` annotates the data shown and the participant's response.

3. **Data Annotation**:

   - During each block, the `EEGHeadset` annotates the type of data shown (e.g., real, fake, celebrity, random) and the participant's response (e.g., 'Yes', 'No', 'Timeout').

4. **Cleanup**:
   - After the experiment, the `PersonalDataManager` erases the participant's data and disconnects the EEG headset.

> NOTE participant data is not stored. After successful exam what's left after them is just assigned UUID and annotated EED data.

## Configuration

The configuration for the Personal Data module is defined in `config.py`. Key configuration parameters include:

- **File Paths**:

  - `CELEBRITIES_FILE_PATH`: Path to the YAML file containing celebrity data.
  - `FAKE_AND_RANDO_DATA_FILE_PATH`: Path to the YAML file containing fake and random identity data.
  - `USER_DATA_FILE_PATH`: Path to the YAML file containing user data.

- **Trial Multipliers**:

  - `TEST_TRIALS_MULTIPLIER`: Multiplier for the number of test trials for fake and real data.
  - `BIGGER_CATCH_TRIALS_MULTIPLIER`: Multiplier for the number of catch trials for celebrity or random data.
  - `SMALLER_CATCH_TRIALS_MULTIPLIER`: Multiplier for the number of catch trials for celebrity or random data.

- **Block Expected Identity**:

  - `BLOCK_EXPECTED_IDENTITY`: Dictionary mapping each `ExperimentBlock` to the expected identity types and trial multipliers.

- **YAML Template**:
  - `YAML_TEMPLATE`: Template for user data in YAML format.

### Explanation of Trial Multipliers

Multiplying trial data by these values will give us the exact amount of probes we need for each block:

- `TEST_TRIALS_MULTIPLIER = 15`: Used for fake and real data. Each probe (e.g., full name, hometown, birthdate) will be shown 15 times, resulting in 45 trials in our example.
- `BIGGER_CATCH_TRIALS_MULTIPLIER = 18`: Used for celebrity or random data to balance the number of 'Yes' and 'No' answers and to reduce the participant's ability to get used to the same data. Every probe is being shown 3 times during the experiment meaning 54 probes during bigger catch trial.
- `SMALLER_CATCH_TRIALS_MULTIPLIER = 3`: Also used for celebrity or random data for the same balancing purpose. Every probe is being shown 3 times during the experiment so that gives nine probes for smaller catch trial.

> If participant is in block where they are supposed to lie about their identity we give celebrity a bigger catch trial number and less to random identity to balance `yes` and `no` answers. For truth blocks we do opposite.

### Block Expected Identity

Depending on the block, the participant was instructed to respond as follows:

- Honest Response to True Identity :

  - `yes` to Real (user's identity)
  - `yes` to Celebrity
  - `no` to Random

- Honest Response to True Identity :

  - `no` to Real (user's identity)
  - `yes` to Celebrity
  - `no` to Random

- Honest Response to Fake Identity :

  - `no` to Fake (user's identity)
  - `yes` to Celebrity
  - `no` to Random

- Deceitful Response to Fake Identity :
  - `yes` to Fake (user's identity)
  - `yes` to Celebrity
  - `no` to Random

The `BLOCK_EXPECTED_IDENTITY` configuration maps each `ExperimentBlock` to the expected identity types and trial multipliers:

- `ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY`: (REAL, SMALL, BIG)
- `ExperimentBlock.DECEITFUL_RESPONSE_TO_TRUE_IDENTITY`: (REAL, BIG, SMALL)
- `ExperimentBlock.HONEST_RESPONSE_TO_FAKE_IDENTITY`: (FAKE, BIG, SMALL)
- `ExperimentBlock.DECEITFUL_RESPONSE_TO_FAKE_IDENTITY`: (FAKE, SMALL, BIG)
