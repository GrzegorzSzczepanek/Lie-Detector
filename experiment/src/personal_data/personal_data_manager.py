import random
import time
from typing import Optional
from unittest.mock import Mock

from ..common import ExperimentBlock, ParticipantResponse, ResponseCounts
from ..config import DEBUG, DEBUG_WITHOUT_HEADSET
from ..eeg_headset.eeg_headset import EEGHeadset
from .common_types import BlockData
from .config import (
    BLOCK_EXPECTED_IDENTITY,
    TEST_TRIALS_MULTIPLIER,
    USER_DATA_FILE_PATH,
    YAML_TEMPLATE,
)
from .identity_generator import IdentityGenerator
from .personal_data import PersonalDataField, PersonalDataType


class PersonalDataManager:
    def __init__(self):
        self._current_block: Optional[ExperimentBlock] = None
        self._current_block_data: list[BlockData] = []
        self._is_practice = False

        self._response_counts = ResponseCounts(correct=0, incorrect=0)

        identity_generator = IdentityGenerator()

        self._real = identity_generator.get_real()
        self._fake = identity_generator.get_fake()
        self._celebrity = identity_generator.get_celebrity()
        self._rando = identity_generator.get_rando()

        self._eeg_headset = (
            Mock() if DEBUG_WITHOUT_HEADSET else EEGHeadset(identity_generator.get_id())
        )

    def cleanup(self) -> None:
        """
        Erases user data.
        """

        self._erase_user_data()

    def start_block(self, experiment_block: ExperimentBlock, is_practice: bool) -> None:
        """
        Args:
            experiment_block (ExperimentBlock): Experiment block to generate data for.
            is_practice (bool): If generated data is for practice trials.

        Raises:
            RuntimeError: If previous block was not stopped.
        """
        if self._current_block is not None:
            raise RuntimeError("Previous block was not stopped.")

        self._eeg_headset.start_block(experiment_block)
        self._is_practice = is_practice
        self._current_block = experiment_block
        self._prepare_block_data(is_practice)

    def stop_block(self) -> None:
        """
        If the current block is not a practice block, saves EEG data.

        Raises:
            RuntimeError: If block was not started.
        """
        if self._current_block is None:
            raise RuntimeError("Block was not started.")

        time.sleep(0.2)  # buffer for EEG annotations
        self._eeg_headset.stop_and_save_block()
        self._current_block = None

    def has_next(self) -> bool:
        """
        Returns:
            bool: If there is more data to get.
        """

        return len(self._current_block_data) > 0

    def get_next(self) -> str:
        """
        Returns:
            str: Next available piece of personal data.

        Raises:
            IndexError: If there is no more data to get.
        """
        if not self._is_practice:
            self._eeg_headset.annotate_data_shown(
                self._current_block_data[0].data_field,
                self._current_block_data[0].data_type,
            )

        if not self.has_next():
            raise IndexError("No more data to get.")
        return self._current_block_data[0].data

    def get_feedback_for_practice_participant_response(
        self, participant_response: ParticipantResponse
    ) -> bool:
        """
        Args:
            participant_response (ParticipantResponse): Response to the last returned piece of data in the practice part of a block.

        Returns:
            bool: If the participant's response was correct.
        """

        # remove the first piece of data from the queue and check if it was correct
        return participant_response == self._current_block_data.pop(0).expected_response

    def register_participant_response(
        self, participant_response: ParticipantResponse
    ) -> None:
        """
        Args:
            participant_response (ParticipantResponse): Response to the last returned piece of data that will be annotated in the EEG recording.
            If TIMEOUT, then the piece of data is not removed from the queue, but the queue is shuffled.
            Response is recorded as correct or incorrect.
        """
        if participant_response != ParticipantResponse.TIMEOUT:
            if participant_response == self._current_block_data[0].expected_response:
                self._response_counts.correct += 1
            else:
                self._response_counts.incorrect += 1
            self._current_block_data.pop(0)
        else:
            random.shuffle(self._current_block_data)

        self._eeg_headset.annotate_response(participant_response)

    def get_response_counts(self) -> ResponseCounts:
        """
        Returns:
            ResponseCounts: Object with correct and incorrect response counts for current block.
        """
        return self._response_counts

    def get_celebrity_data(self) -> dict[PersonalDataField, str]:
        """
        Returns:
            dict[PersonalDataField, str]: chosen celebrity data
        """

        return self._celebrity

    def get_rando_data(self) -> dict[PersonalDataField, str]:
        """
        Returns:
            dict[PersonalDataField, str]: chosen rando data
        """

        return self._rando

    def get_fake_identity_data(self) -> dict[PersonalDataField, str]:
        """
        Returns:
            dict[PersonalDataField, str]: generated fake identity data
        """

        return self._fake

    def _erase_user_data(self) -> None:
        if not DEBUG:
            with open(USER_DATA_FILE_PATH, "w", encoding="utf-8") as file:
                file.write(YAML_TEMPLATE)

    def _find_expected_response(
        self, person: dict[PersonalDataField, str]
    ) -> ParticipantResponse:
        """
        Args:
            person (dict[PersonalDataField, str]): Person data.
        Returns:
            ParticipantResponse: Expected response for the given person.
        Raises:
            RuntimeError: If person data is not equal to any existing identity.
        """

        if person == self._real:
            if (
                self._current_block
                == ExperimentBlock.DECEITFUL_RESPONSE_TO_TRUE_IDENTITY
            ):
                return ParticipantResponse.NO
            return ParticipantResponse.YES

        if person == self._fake:
            if self._current_block == ExperimentBlock.HONEST_RESPONSE_TO_FAKE_IDENTITY:
                return ParticipantResponse.NO
            return ParticipantResponse.YES

        if person == self._celebrity:
            return ParticipantResponse.YES

        if person == self._rando:
            return ParticipantResponse.NO

        raise RuntimeError("Person is not equal to any existing identity.")

    def _prepare_block_data(self, is_practice: bool):
        """
        Prepares data for the current block. Person is either real or fake.

        Raises:
            RuntimeError: If block was not started.
        """
        if self._current_block is None:
            raise RuntimeError("Block was not started.")

        current_block_data: list[BlockData] = []

        (
            current_personal_data_type,
            celebrity_trials_multiplier,
            rando_trials_multiplier,
        ) = BLOCK_EXPECTED_IDENTITY[self._current_block]
        current_identity = (
            self._real
            if current_personal_data_type == PersonalDataType.REAL
            else self._fake
        )
        identities = [current_identity, self._celebrity, self._rando]
        personal_data_types = [
            current_personal_data_type,
            PersonalDataType.CELEBRITY,
            PersonalDataType.RANDO,
        ]

        for person, personal_data_type in zip(identities, personal_data_types):
            trials: list[BlockData] = []
            expected_response = self._find_expected_response(person)

            for field, value in person.items():
                trials.append(
                    BlockData(
                        data=value,
                        data_type=personal_data_type,
                        data_field=field,
                        expected_response=expected_response,
                    )
                )
            if is_practice:
                current_block_data += trials  # 3 probes per identity and 9 probes for whole practice block
            else:
                # 120 probes per block consissting 72 probes of celebrity and rando (60 + 12) and 48 probes of real or fake
                if person == self._real or person == self._fake:
                    current_block_data += trials * TEST_TRIALS_MULTIPLIER

                elif person == self._celebrity:
                    current_block_data += trials * celebrity_trials_multiplier
                elif person == self._rando:
                    current_block_data += trials * rando_trials_multiplier

        random.shuffle(current_block_data)
        self._current_block_data = (
            current_block_data[:5] if DEBUG else current_block_data
        )
