import pygame
from pygame.color import Color

from ..common import ExperimentBlock, ParticipantResponse
from ..config import DEBUG

SCREEN_PARAMS = ((1200, 600), 0) if DEBUG else ((0, 0), pygame.FULLSCREEN)

FIXATION_CROSS_TIME_RANGE_MILLIS = (400, 600)
TIMEOUT_MILLIS = 2000
BREAK_BETWEEN_TRIALS_TIME_RANGE_MILLIS = (1300, 1600)
BREAK_BETWEEN_PRACTICE_AND_PROPER_MILLIS = 2_000 if DEBUG else 10_000

NEURON_ICON_PATH = "src/gui/assets/neuron.ico"
APP_TITLE = "KN Neuron: Lie Detector"

MARGIN_BETWEEN_DATA_FIELDS_AS_WIDTH_PERCENTAGE = 0.01

BACKGROUND_COLOR_PRIMARY = Color("#f4f4f4")
BACKGROUND_COLOR_INCORRECT = Color("#e87060")

FIXATION_CROSS_COLOR = Color("#000000")
FIXATION_CROSS_LENGTH_AS_WIDTH_PERCENTAGE = 0.08
FIXATION_CROSS_WIDTH_AS_WIDTH_PERCENTAGE = 0.02

TEXT_COLOR = Color("#000000")
TEXT_FONT = "Helvetica"
TEXT_FONT_SIZE_AS_WIDTH_PERCENTAGE = 0.04

CELEBRITY_DATA_TEXT = "DANE CELEBRYTY"
CELEBRITY_DATA_HINT_TEXT = 'zawsze odpowiedź "tak"'
RANDO_DATA_TEXT = "DANE LOSOWEJ OSOBY"
RANDO_DATA_HINT_TEXT = 'zawsze odpowiedź "nie"'
FAKE_IDENTITY_DATA_TEXT = "TWOJE FAŁSZYWE DANE"
FAKE_IDENTITY_DATA_HINT_TEXT = "odpowiedź zależna od bloku"
INCORRECT_RESPONSE_TEXT = "Podano niepoprawną odpowiedź!"
INCORRECT_RESPONSE_SHOWN_DATA_TEXT = "WYŚWIETLONE DANE: "
INCORRECT_RESPONSE_CORRECT_RESPONSE_TEXT = "POPRAWNA ODPOWIEDŹ: "
TIMEOUT_TEXT = "Za długo!"
BLOCK_END_TEXT = "Koniec bloku."
BREAK_BETWEEN_BLOCKS_TEXT = "NASTĘPNY BLOK:"
BREAK_BETWEEN_PRACTICE_AND_PROPER_TEXTS = [
    "Koniec prób treningowych.",
    "Kolejne próby nie będą już treningowe.",
    "Przerwa potrwa 10 sekund.",
]
RESULTS_TEXT = "Poprawnych odpowiedzi:"

EXPERIMENT_BLOCK_SEQUENCE_PART_1 = [
    ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY,
    ExperimentBlock.DECEITFUL_RESPONSE_TO_TRUE_IDENTITY,
]
EXPERIMENT_BLOCK_SEQUENCE_PART_2 = [
    ExperimentBlock.HONEST_RESPONSE_TO_FAKE_IDENTITY,
    ExperimentBlock.DECEITFUL_RESPONSE_TO_FAKE_IDENTITY,
]

RESPONSE_KEYS = {pygame.K_LSHIFT: "lewy Shift", pygame.K_RSHIFT: "prawy Shift"}
CONFIRMATION_KEY = pygame.K_RSHIFT
GO_BACK_KEY = pygame.K_LSHIFT
GO_FORWARD_KEY = pygame.K_RSHIFT
QUIT_KEY = pygame.K_RSHIFT

GO_BACK_TEXT = f"Aby wrócić, naciśnij {RESPONSE_KEYS[GO_BACK_KEY]}."
GO_FORWARD_TEXT = f"Aby przejść dalej, naciśnij {RESPONSE_KEYS[GO_FORWARD_KEY]}."

EXPERIMENT_BLOCK_TRANSLATIONS = {
    ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY: [
        'dane celebryty → "tak"',
        'dane losowej osoby → "nie"',
        'Twoje dane → "tak"',
    ],
    ExperimentBlock.HONEST_RESPONSE_TO_FAKE_IDENTITY: [
        'dane celebryty → "tak"',
        'dane losowej osoby → "nie"',
        'dane osoby, pod którą się podszywasz → "nie"',
    ],
    ExperimentBlock.DECEITFUL_RESPONSE_TO_TRUE_IDENTITY: [
        'dane celebryty → "tak"',
        'dane losowej osoby → "nie"',
        'Twoje dane → "nie"',
    ],
    ExperimentBlock.DECEITFUL_RESPONSE_TO_FAKE_IDENTITY: [
        'dane celebryty → "tak"',
        'dane losowej osoby → "nie"',
        'dane osoby, pod którą się podszywasz → "tak"',
    ],
}
PARTICIPANT_RESPONSE_TRANSLATIONS = {
    ParticipantResponse.YES: '"tak"',
    ParticipantResponse.NO: '"nie"',
}
