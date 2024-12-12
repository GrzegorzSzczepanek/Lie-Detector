from enum import Enum


class PersonalDataType(Enum):
    REAL = 1
    FAKE = 2
    CELEBRITY = 3
    RANDO = 4


class PersonalDataField(Enum):
    NAME = 1
    BIRTH_DATE = 2
    HOMETOWN = 3

    @classmethod
    def from_string(cls, value):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"No enum member with name '{value}'")
