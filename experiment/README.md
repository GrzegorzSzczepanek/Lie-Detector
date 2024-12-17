# Lie Detector

## Overview

This project aims to detect lies about one’s own identity through brainwave analysis. The experiment is divided into four blocks, each with different instructions for the participant. The participant's brainwave data is recorded using an EEG headset, and the data is analyzed to determine the truthfulness of their responses.

## Experiment Process

1. **Initialization**:

   - The participant's real and fake identity data is loaded.
   - The EEG headset is connected and initialized.

2. **Experiment Blocks**:

   - The experiment consists of four blocks:
     - Honest response to true identity
     - Deceitful response to true identity
     - Honest response to fake identity
     - Deceitful response to fake identity
   - Each block has specific instructions for the participant on how to respond to different types of data.

3. **Data Annotation**:

   - During each block, the EEG data is annotated with information about the data shown and the participant's response.

4. **Cleanup**:
   - After the experiment, the participant's data is erased, and the EEG headset is disconnected.

## GUI Instructions

First 5 screens - navigate to the next/previous screen using the arrow keys.

"End of block" screen - proceed to the next block by pressing Enter.

Last screen, with the number of correct answers - exit by pressing Q.

## Setup Instructions

For experiment part of the project, Python 3.11 was used.

1. Create a virtual environment from the terminal: `py -m venv venv` (if you are using an environment manager like conda, do it your way).

2. Activate the virtual environment: `.\venv\Scripts\activate`.

3. Install Poetry: `pip install poetry`.

4. Install all dependencies: `poetry install`.

5. If you want to add any dependencies: `poetry add <name>`, e.g., `poetry add requests`.

6. Then to start the experiment you have to run `main.py` file.

## Identities

We employed four types of data that the experiment participant has to memorize and answer with yes or no. We made sure data types have nothing in common (based on fields mentioned in [identity generator](#identitygenerator) section). Data types consisted of:

- **Real** (user real data)
- **Celebrity** - a well-known person that the participant knew. The answer for this is always _yes_ and it serves the purpose of balancing _yes_ and _no_ answers to have an equal amount of labels. The participant was explained to lie that the celebrity's data is theirs. It was a sort of impersonation.
- **Random** - Opposite of the celebrity. The participant was supposed to always respond that **Random** data are not theirs.
- **Fake** - after the second block we do

## Experiment Blocks

The experiment is separated into four blocks:

- Honest response to true identity (yes answer for **Real** and **Celebrity**, no for **Random**)
- Deceitful response to true identity (no answer for **Real** and **Random**, yes for **Celebrity**)
- Honest response to fake identity (no answer for **Fake** and for **Random**, yes to **Celebrity**)
- Deceitful response to fake identity (yes answer for **Fake** and **Celebrity**, no for **Random**)

> After the second block, the user's real data is replaced and we give them time to learn the fake ID.

<!-- ## Screenshots -->

<!-- ![Screenshot 1](path/to/screenshot1.png)
![Screenshot 2](path/to/screenshot2.png) -->

## References to other modules

- [Personal Data Module README](/experiment/src/personal_data/README.md)
- [EEG Headset Module README](/experiment/src/eeg_headset/README.md)
- [GUI Module README](/experiment/src/gui/README.md)
- [AI Module README](/classificators_and_data/README.md)
