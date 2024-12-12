from dataclasses import dataclass

from ..common import ParticipantResponse
from .personal_data import PersonalDataField, PersonalDataType


@dataclass 
class BlockData: 
    data: str  # f.ex. "kwiecień" 
    data_type: PersonalDataType  # f.ex. PersonalDataType.REAL 
    data_field: PersonalDataField  # f.ex. PersonalDataField.DATE_OF_BIRTH 
    expected_response: ParticipantResponse  # f.ex. ParticipantResponse.NO 
 