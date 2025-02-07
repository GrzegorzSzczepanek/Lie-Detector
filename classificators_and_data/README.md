- `poetry.lock`: Poetry lock file for dependencies.
- `pyproject.toml`: Poetry configuration file.

## Installation

> We used python 3.12 for the AI part

### Using Poetry

1. **Create a virtual environment**:

   ```sh
   python -m venv ai-venv
   ```

2. **Activate the virtual environment**:

   - On Windows:
     ```sh
     .\ai-venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source ai-venv/bin/activate
     ```

3. **Install Poetry**:

   ```sh
   pip install poetry
   ```

4. **Install all dependencies**:

   ```sh
   poetry install
   ```

5. **Add any dependencies**:
   ```sh
   poetry add <name>
   ```

### Using `requirements.txt`

1. **Create a virtual environment**:

   ```sh
   python -m venv ai-venv
   ```

2. **Activate the virtual environment**:

   - On Windows:
     ```sh
     .\ai-venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source ai-venv/bin/activate
     ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## `requirements.txt`

```txt
joblib==1.4.2
mne==1.8.0
pandas==2.2.3
seaborn==0.13.2
scikit-learn==1.6.0
torcheeg==1.1.2
mne==1.8.0
ipykernel==6.29.5
pandas==2.2.2
tensorboard==2.17.1
frozendict==2.4.6
torch==2.5.1+cu124
torchvision==0.20.1+cu124
torchaudio==2.5.1+cu124
```
