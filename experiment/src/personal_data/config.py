from pathlib import Path

from ..common import ExperimentBlock
from .personal_data import PersonalDataType

assets_dir = Path(__file__).parent.parent
CELEBRITIES_FILE_PATH = assets_dir / "assets/celebrities_data.yaml"
FAKE_AND_RANDO_DATA_FILE_PATH = assets_dir / "assets/identity_data.yaml"
USER_DATA_FILE_PATH = assets_dir / "assets/user_data.yaml"

# Multiplying trial data by these values will give us exact amount of probes we need for each block
TEST_TRIALS_MULTIPLIER = 15  # Used for fake and real
BIGGER_CATCH_TRIALS_MULTIPLIER = 18  # Used for celebrity or rando
SMALLER_CATCH_TRIALS_MULTIPLIER = 3  # Used for celebrity or rando

REAL = PersonalDataType.REAL
FAKE = PersonalDataType.FAKE

SMALL = SMALLER_CATCH_TRIALS_MULTIPLIER
BIG = BIGGER_CATCH_TRIALS_MULTIPLIER

BLOCK_EXPECTED_IDENTITY = {
    ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY: (REAL, SMALL, BIG),
    ExperimentBlock.DECEITFUL_RESPONSE_TO_TRUE_IDENTITY: (REAL, BIG, SMALL),
    ExperimentBlock.HONEST_RESPONSE_TO_FAKE_IDENTITY: (FAKE, BIG, SMALL),
    ExperimentBlock.DECEITFUL_RESPONSE_TO_FAKE_IDENTITY: (FAKE, SMALL, BIG),
}

YAML_TEMPLATE = """id: ABC123
# tylko pełne imię bez zdrobnień
name: Jan
surname: Kowalski
# w formacie dzień (liczba) miesiąc (słownie np. stycznia, lutego, marca, kwietnia, maja, czerwca, lipca, sierpnia, września, października, listopada, grudnia)
birth_date: 1 stycznia
# tylko nazwa miasta
hometown: Warszawa
# m lub k
gender: m
"""
