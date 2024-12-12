from typing import NamedTuple

from ..common import ExperimentBlock, ParticipantResponse


class Size(NamedTuple):
    width: int
    height: int


class ResponseKey(NamedTuple):
    response: ParticipantResponse
    response_translation: str
    key_description: str


class ExperimentSubblock(NamedTuple):
    block: ExperimentBlock
    is_practice: bool
    part: int
