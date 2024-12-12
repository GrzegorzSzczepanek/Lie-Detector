import itertools
import logging
import random
from typing import Optional, Sequence

import pygame

from ..common import ParticipantResponse
from ..personal_data import PersonalDataField, PersonalDataManager
from .common_types import ExperimentSubblock, ResponseKey, Size
from .config import (
    APP_TITLE,
    BACKGROUND_COLOR_INCORRECT,
    BACKGROUND_COLOR_PRIMARY,
    BLOCK_END_TEXT,
    BREAK_BETWEEN_BLOCKS_TEXT,
    BREAK_BETWEEN_PRACTICE_AND_PROPER_MILLIS,
    BREAK_BETWEEN_PRACTICE_AND_PROPER_TEXTS,
    BREAK_BETWEEN_TRIALS_TIME_RANGE_MILLIS,
    CELEBRITY_DATA_HINT_TEXT,
    CELEBRITY_DATA_TEXT,
    CONFIRMATION_KEY,
    EXPERIMENT_BLOCK_SEQUENCE_PART_1,
    EXPERIMENT_BLOCK_SEQUENCE_PART_2,
    EXPERIMENT_BLOCK_TRANSLATIONS,
    FAKE_IDENTITY_DATA_HINT_TEXT,
    FAKE_IDENTITY_DATA_TEXT,
    FIXATION_CROSS_COLOR,
    FIXATION_CROSS_LENGTH_AS_WIDTH_PERCENTAGE,
    FIXATION_CROSS_TIME_RANGE_MILLIS,
    FIXATION_CROSS_WIDTH_AS_WIDTH_PERCENTAGE,
    GO_BACK_KEY,
    GO_BACK_TEXT,
    GO_FORWARD_KEY,
    GO_FORWARD_TEXT,
    INCORRECT_RESPONSE_CORRECT_RESPONSE_TEXT,
    INCORRECT_RESPONSE_SHOWN_DATA_TEXT,
    INCORRECT_RESPONSE_TEXT,
    MARGIN_BETWEEN_DATA_FIELDS_AS_WIDTH_PERCENTAGE,
    NEURON_ICON_PATH,
    PARTICIPANT_RESPONSE_TRANSLATIONS,
    QUIT_KEY,
    RANDO_DATA_HINT_TEXT,
    RANDO_DATA_TEXT,
    RESPONSE_KEYS,
    RESULTS_TEXT,
    SCREEN_PARAMS,
    TEXT_COLOR,
    TEXT_FONT,
    TEXT_FONT_SIZE_AS_WIDTH_PERCENTAGE,
    TIMEOUT_MILLIS,
    TIMEOUT_TEXT,
)
from .iterator import Iterator
from .screen_sequence import ScreenSequence


class Gui:
    def __init__(self) -> None:
        self._init_experiment_blocks()
        self._init_screen_sequences()
        self._init_response_keys()
        self._init_events()
        self._running = False
        self._can_quit = False
        self._do_handle_participant_response = True
        self._wait_for_confirmation = False
        self._data_to_show: Optional[str] = None
        self._clock = pygame.time.Clock()

    def start(self) -> None:
        logging.debug("Starting...")

        self._personal_data_manager = PersonalDataManager()
        self._celebrity_data = self._sort_by_personal_data_field(
            self._personal_data_manager.get_celebrity_data()
        )
        self._rando_data = self._sort_by_personal_data_field(
            self._personal_data_manager.get_rando_data()
        )
        self._fake_identity_data = self._sort_by_personal_data_field(
            self._personal_data_manager.get_fake_identity_data()
        )

        pygame.init()
        pygame.display.set_icon(pygame.image.load(NEURON_ICON_PATH))
        pygame.display.set_caption(APP_TITLE)
        self._main_surface = pygame.display.set_mode(*SCREEN_PARAMS)
        self._main_screens_sequence.call_current()

        self._running = True
        self._mainloop()

    def _mainloop(self) -> None:
        while self._running:
            self._handle_events()
            self._clock.tick(60)
            pygame.display.update()

        self._personal_data_manager.cleanup()

    def _init_response_keys(self) -> None:
        responses = list(PARTICIPANT_RESPONSE_TRANSLATIONS.items())
        random.shuffle(responses)
        self._response_keys = {
            key: ResponseKey(response, response_translation, key_description)
            for (key, key_description), (response, response_translation) in zip(
                RESPONSE_KEYS.items(), responses
            )
        }

        logging.debug(
            f"key assignment: {', '.join(f'{response_key.key_description} = {response_key.response}' for response_key in self._response_keys.values())}"
        )

    def _init_experiment_blocks(self) -> None:
        iterator = [
            ExperimentSubblock(block, is_practice, part)
            for part, block_sequence in enumerate(
                (EXPERIMENT_BLOCK_SEQUENCE_PART_1, EXPERIMENT_BLOCK_SEQUENCE_PART_2),
                start=1,
            )
            for block, is_practice in itertools.product(block_sequence, (True, False))
        ]
        self._experiment_blocks_sequence = Iterator(iterator)

    def _init_screen_sequences(self) -> None:
        self._main_screens_sequence = ScreenSequence(
            [
                self._draw_keys_assignment,
                self._draw_celebrity_data_screen,
                self._draw_rando_data_screen,
                self._draw_block_info_screen,
                self._run_block,
                self._draw_keys_assignment,
                self._draw_celebrity_data_screen,
                self._draw_rando_data_screen,
                self._draw_fake_identity_data_screen,
                self._draw_block_info_screen,
                self._run_block,
            ],
            (4, 10),
            False,
        )

        self._experiment_block_parts_sequence = ScreenSequence(
            [
                self._draw_fixation_cross,
                self._draw_experiment_data,
                self._draw_blank_screen,
            ],
            1,
            True,
        )

    def _init_events(self) -> None:
        self._timeout_event = pygame.event.Event(pygame.event.custom_type())
        self._go_to_next_part_event = pygame.event.Event(pygame.event.custom_type())
        self._run_block_event = pygame.event.Event(pygame.event.custom_type())

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    self._running = False
                case pygame.KEYDOWN if event.key == QUIT_KEY and self._can_quit:
                    self._running = False
                case pygame.KEYDOWN:
                    if not self._main_screens_sequence.is_at_marked() and event.key in (
                        GO_BACK_KEY,
                        GO_FORWARD_KEY,
                    ):
                        self._handle_screen_change(event.key)
                    elif (
                        self._experiment_blocks_sequence.get_current().is_practice
                        and not self._do_handle_participant_response
                        and event.key == CONFIRMATION_KEY
                    ):
                        self._go_to_next_part()
                    elif (
                        self._main_screens_sequence.is_at_marked()
                        and self._experiment_block_parts_sequence.is_at_marked()
                        and self._do_handle_participant_response
                        and event.key in self._response_keys
                    ):
                        self._handle_participant_response(event.key)
                    elif self._wait_for_confirmation and event.key == CONFIRMATION_KEY:
                        self._wait_for_confirmation = False
                        self._run_block()
                case self._go_to_next_part_event.type:
                    self._go_to_next_part()
                case self._run_block_event.type:
                    self._run_block()
                case self._timeout_event.type:
                    self._register_response(ParticipantResponse.TIMEOUT)

    def _handle_screen_change(self, key: int) -> None:
        """
        Args:
            key (int): Pygame key code.

        Raises:
            ValueError: If key is not K_LEFT nor K_RIGHT.
        """

        if key == GO_BACK_KEY:
            self._main_screens_sequence.move_back_and_call()
        elif key == GO_FORWARD_KEY:
            self._main_screens_sequence.advance_and_call()
        else:
            raise ValueError(f"Invalid key code: {key}")

    def _handle_participant_response(self, key: int) -> None:
        """
        Args:
            key (int): Pygame key code.

        Raises:
            ValueError: If key is not a valid response key.
        """

        if key not in self._response_keys:
            raise ValueError(f"Invalid key code: {key}")

        self._clear_timeout(self._timeout_event)
        self._register_response(self._response_keys[key].response)

    def _register_response(self, response: ParticipantResponse) -> None:
        """
        Args:
            response (ParticipantResponse): Participant response to register.
        """

        logging.debug(f"registered participant response: {response}")

        if self._experiment_blocks_sequence.get_current().is_practice:
            feedback = self._personal_data_manager.get_feedback_for_practice_participant_response(
                response
            )

            if response == ParticipantResponse.TIMEOUT:
                self._draw_incorrect_response_screen(TIMEOUT_TEXT)
            elif not feedback:
                logging.debug("response was incorrect")

                if self._data_to_show is None:
                    raise ValueError("Data to show is somehow None.")

                correct_response = (
                    ParticipantResponse.YES
                    if response == ParticipantResponse.NO
                    else ParticipantResponse.NO
                )
                correct_response_text = PARTICIPANT_RESPONSE_TRANSLATIONS[
                    correct_response
                ]
                correct_key_description = [
                    key_info.key_description
                    for key_info in self._response_keys.values()
                    if key_info.response == correct_response
                ][0]

                self._draw_incorrect_response_screen(
                    [
                        INCORRECT_RESPONSE_TEXT,
                        "",
                        INCORRECT_RESPONSE_SHOWN_DATA_TEXT,
                        self._data_to_show,
                        INCORRECT_RESPONSE_CORRECT_RESPONSE_TEXT,
                        f"{correct_response_text} ({correct_key_description})",
                    ]
                )
            else:
                self._go_to_next_part()
        else:
            self._personal_data_manager.register_participant_response(response)
            self._go_to_next_part()

    def _draw_background(self, color: pygame.Color = BACKGROUND_COLOR_PRIMARY) -> None:
        """
        Args:
            color (pygame.Color, optional): Color of the background. Defaults to BACKGROUND_COLOR_PRIMARY.
        """

        self._main_surface.fill(color)

    def _draw_incorrect_response_screen(self, text: str | Sequence[str]) -> None:
        """
        Args:
            text (str): Info about the incorrect response type.
        """

        self._do_handle_participant_response = False
        self._draw_background(BACKGROUND_COLOR_INCORRECT)

        if isinstance(text, str):
            text = [text]
        self._draw_texts([*text, "", GO_FORWARD_TEXT])

    def _draw_blank_screen(self) -> None:
        self._draw_background()
        self._set_timeout(
            self._go_to_next_part_event,
            self._get_random_from_range(BREAK_BETWEEN_TRIALS_TIME_RANGE_MILLIS),
        )

    def _draw_texts(self, texts: str | Sequence[str]) -> None:
        """
        Args:
            texts (str | Sequence[str]): One string or an iterable of strings to render.
        """

        if isinstance(texts, str):
            texts = [texts]

        display_info = pygame.display.Info()
        screen_width = display_info.current_w
        screen_height = display_info.current_h

        font_size = int(screen_width * TEXT_FONT_SIZE_AS_WIDTH_PERCENTAGE)
        font = pygame.font.SysFont(TEXT_FONT, font_size)
        font_bold = pygame.font.SysFont(TEXT_FONT, font_size, bold=True)

        items_to_draw = []
        item_height_sum = 0
        for text in texts:
            font_to_render = font_bold if text.isupper() else font
            rendered_text = font_to_render.render(text, True, TEXT_COLOR)
            items_to_draw.append(rendered_text)

            text_height = rendered_text.get_size()[1]
            item_height_sum += text_height

        margin_between_items = int(
            MARGIN_BETWEEN_DATA_FIELDS_AS_WIDTH_PERCENTAGE * screen_height
        )
        all_margins_height = margin_between_items * len(texts)
        text_container_total_height = item_height_sum + all_margins_height

        y_coord = self._calculate_margin(text_container_total_height, screen_height)
        for item in items_to_draw:
            item_width, item_height = item.get_size()
            self._main_surface.blit(
                item,
                (self._calculate_margin(item_width, screen_width), y_coord),
            )
            y_coord += item_height + margin_between_items

    def _draw_data(self, data: str | Sequence[str]) -> None:
        """
        Args:
            data (str | Sequence[str]): One string or an iterable of strings to render on the default background.
        """

        self._draw_background()
        self._draw_texts(data)

    def _draw_keys_assignment(self) -> None:
        key_assignments = [
            f"{response_key.key_description}: {response_key.response_translation}"
            for response_key in self._response_keys.values()
        ] + ["", GO_FORWARD_TEXT]
        self._draw_data(key_assignments)

    def _draw_celebrity_data_screen(self) -> None:
        self._draw_data(
            [
                CELEBRITY_DATA_TEXT,
                *self._celebrity_data,
                "",
                CELEBRITY_DATA_HINT_TEXT,
                "",
                GO_BACK_TEXT,
                GO_FORWARD_TEXT,
            ]
        )

    def _draw_rando_data_screen(self) -> None:
        self._draw_data(
            [
                RANDO_DATA_TEXT,
                *self._rando_data,
                "",
                RANDO_DATA_HINT_TEXT,
                "",
                GO_BACK_TEXT,
                GO_FORWARD_TEXT,
            ]
        )

    def _draw_fake_identity_data_screen(self) -> None:
        self._draw_data(
            [
                FAKE_IDENTITY_DATA_TEXT,
                *self._fake_identity_data,
                "",
                FAKE_IDENTITY_DATA_HINT_TEXT,
                "",
                GO_BACK_TEXT,
                GO_FORWARD_TEXT,
            ]
        )

    def _draw_block_info_screen(self) -> None:
        next_block = self._experiment_blocks_sequence.get_current()
        break_texts = [
            BREAK_BETWEEN_BLOCKS_TEXT,
            *EXPERIMENT_BLOCK_TRANSLATIONS[next_block.block],
            "",
            GO_BACK_TEXT,
            GO_FORWARD_TEXT,
        ]
        self._draw_data(break_texts)

    def _draw_fixation_cross(self) -> None:
        screen_width, screen_height = self._get_screen_size()
        fixation_cross_length = int(
            FIXATION_CROSS_LENGTH_AS_WIDTH_PERCENTAGE * screen_width
        )
        fixation_cross_width = int(
            FIXATION_CROSS_WIDTH_AS_WIDTH_PERCENTAGE * screen_width
        )

        horizontal_x_coord = self._calculate_margin(fixation_cross_length, screen_width)
        horizontal_y_coord = self._calculate_margin(fixation_cross_width, screen_height)
        horizontal_rect = pygame.Rect(
            horizontal_x_coord,
            horizontal_y_coord,
            fixation_cross_length,
            fixation_cross_width,
        )

        vertical_x_coord = self._calculate_margin(fixation_cross_width, screen_width)
        vertical_y_coord = self._calculate_margin(fixation_cross_length, screen_height)
        vertical_rect = pygame.Rect(
            vertical_x_coord,
            vertical_y_coord,
            fixation_cross_width,
            fixation_cross_length,
        )

        self._draw_background()
        pygame.draw.rect(self._main_surface, FIXATION_CROSS_COLOR, horizontal_rect)
        pygame.draw.rect(self._main_surface, FIXATION_CROSS_COLOR, vertical_rect)
        self._set_timeout(
            self._go_to_next_part_event,
            self._get_random_from_range(FIXATION_CROSS_TIME_RANGE_MILLIS),
        )

    def _draw_experiment_data(self) -> None:
        self._data_to_show = self._personal_data_manager.get_next()
        self._draw_data(self._data_to_show)
        self._set_timeout(self._timeout_event, TIMEOUT_MILLIS)

    def _draw_break_between_practice_and_proper(self) -> None:
        self._draw_data(BREAK_BETWEEN_PRACTICE_AND_PROPER_TEXTS)
        self._set_timeout(
            self._run_block_event,
            BREAK_BETWEEN_PRACTICE_AND_PROPER_MILLIS,
        )

    def _draw_break_between_blocks(self) -> None:
        next_block = self._experiment_blocks_sequence.get_current()
        break_texts = [
            BLOCK_END_TEXT,
            "",
            BREAK_BETWEEN_BLOCKS_TEXT,
            *EXPERIMENT_BLOCK_TRANSLATIONS[next_block.block],
            "",
            GO_FORWARD_TEXT,
        ]
        self._draw_data(break_texts)
        self._wait_for_confirmation = True

    def _draw_results(self) -> None:
        self._can_quit = True
        results = self._personal_data_manager.get_response_counts()
        self._draw_data(
            [RESULTS_TEXT, f"{results.correct} / {results.correct + results.incorrect}"]
        )

    def _go_to_next_block(self) -> None:
        self._personal_data_manager.stop_block()

        current_block = self._experiment_blocks_sequence.get_current()
        was_practice = current_block.is_practice
        current_block_part = current_block.part

        self._experiment_blocks_sequence.advance()

        if not self._experiment_blocks_sequence.has_next():
            logging.debug("blocks ended")
            self._draw_results()
            return

        if self._experiment_blocks_sequence.get_current().part != current_block_part:
            self._main_screens_sequence.advance_and_call()
            return

        logging.debug("break between blocks")

        if was_practice:
            self._draw_break_between_practice_and_proper()
        else:
            self._draw_break_between_blocks()

    def _go_to_next_part(self) -> None:
        self._do_handle_participant_response = True

        if not self._personal_data_manager.has_next():
            self._experiment_block_parts_sequence.reset()
            self._go_to_next_block()
            return

        self._experiment_block_parts_sequence.advance_and_call()

    def _run_block(self) -> None:
        block_info = self._experiment_blocks_sequence.get_current()

        logging.debug(f"running block: {block_info}")

        block, is_practice, _ = block_info
        self._personal_data_manager.start_block(block, is_practice)
        self._experiment_block_parts_sequence.call_current()

    def _calculate_margin(self, element_size: int, screen_size: int) -> int:
        """
        Args:
            element_size (int): Size of the element to calculate margin for (width or height).
            screen_size (int): Size of the screen (width or height).

        Returns:
            int: Size of the margin.
        """

        return (screen_size - element_size) // 2

    def _get_screen_size(self) -> Size:
        """
        Returns:
            Size: Screen size.
        """

        display_info = pygame.display.Info()
        return Size(display_info.current_w, display_info.current_h)

    def _get_random_from_range(self, range: tuple[int, int]) -> int:
        """
        Args:
            range (tuple[int, int]): Range, inclusive on both ends.

        Returns:
            int: Random integer from range.
        """

        start, end = range
        return random.randrange(start, end + 1)

    def _set_timeout(self, event: pygame.event.Event, millis: int) -> None:
        """
        Args:
            event (pygame.event.Event): Event to set the timeout for.
            millis (int): After how many milliseconds to dispatch the event.
        """

        pygame.time.set_timer(event, millis, 1)

    def _clear_timeout(self, event: pygame.event.Event) -> None:
        """
        Args:
            event (pygame.event.Event): Event to clear the timeout for.
        """

        pygame.time.set_timer(event, 0)

    def _sort_by_personal_data_field(
        self, personal_data: dict[PersonalDataField, str]
    ) -> list[str]:
        personal_data_field_sequence = list(PersonalDataField)

        return [
            data
            for _, data in sorted(
                personal_data.items(),
                key=lambda k: personal_data_field_sequence.index(k[0]),
            )
        ]
