from dataclasses import dataclass
from enum import Enum


class ExperimentBlock(Enum):
    HONEST_RESPONSE_TO_TRUE_IDENTITY = 1
    DECEITFUL_RESPONSE_TO_TRUE_IDENTITY = 2
    HONEST_RESPONSE_TO_FAKE_IDENTITY = 3
    DECEITFUL_RESPONSE_TO_FAKE_IDENTITY = 4


class ParticipantResponse(Enum):
    YES = 1
    NO = 2
    TIMEOUT = 3


@dataclass
class ResponseCounts:
    correct: int
    incorrect: int
