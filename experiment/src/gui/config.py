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

CELEBRITY_DATA_TEXT = "CELEBRITY DATA"
CELEBRITY_DATA_HINT_TEXT = 'always answer "yes"'
RANDO_DATA_TEXT = "RANDOM PERSON DATA"
RANDO_DATA_HINT_TEXT = 'always answer "no"'
FAKE_IDENTITY_DATA_TEXT = "YOUR FAKE DATA"
FAKE_IDENTITY_DATA_HINT_TEXT = "response depends on the block"
INCORRECT_RESPONSE_TEXT = "Incorrect response provided!"
INCORRECT_RESPONSE_SHOWN_DATA_TEXT = "DISPLAYED DATA: "
INCORRECT_RESPONSE_CORRECT_RESPONSE_TEXT = "CORRECT RESPONSE: "
TIMEOUT_TEXT = "Too long!"
BLOCK_END_TEXT = "End of block."
BREAK_BETWEEN_BLOCKS_TEXT = "NEXT BLOCK:"
BREAK_BETWEEN_PRACTICE_AND_PROPER_TEXTS = [
    "End of practice trials.",
    "The next trials will no longer be practice.",
    "The break will last 10 seconds.",
]
RESULTS_TEXT = "Correct responses:"

EXPERIMENT_BLOCK_SEQUENCE_PART_1 = [
    ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY,
    ExperimentBlock.DECEITFUL_RESPONSE_TO_TRUE_IDENTITY,
]
EXPERIMENT_BLOCK_SEQUENCE_PART_2 = [
    ExperimentBlock.HONEST_RESPONSE_TO_FAKE_IDENTITY,
    ExperimentBlock.DECEITFUL_RESPONSE_TO_FAKE_IDENTITY,
]

RESPONSE_KEYS = {pygame.K_LSHIFT: "left Shift", pygame.K_RSHIFT: "right Shift"}
CONFIRMATION_KEY = pygame.K_RSHIFT
GO_BACK_KEY = pygame.K_LSHIFT
GO_FORWARD_KEY = pygame.K_RSHIFT
QUIT_KEY = pygame.K_RSHIFT

GO_BACK_TEXT = f"To go back, press {RESPONSE_KEYS[GO_BACK_KEY]}."
GO_FORWARD_TEXT = f"To go forward, press {RESPONSE_KEYS[GO_FORWARD_KEY]}."

EXPERIMENT_BLOCK_TRANSLATIONS = {
    ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY: [
        'celebrity data → "yes"',
        'random person data → "no"',
        'Your data → "yes"',
    ],
    ExperimentBlock.HONEST_RESPONSE_TO_FAKE_IDENTITY: [
        'celebrity data → "yes"',
        'random person data → "no"',
        'data of the person you are impersonating → "no"',
    ],
    ExperimentBlock.DECEITFUL_RESPONSE_TO_TRUE_IDENTITY: [
        'celebrity data → "yes"',
        'random person data → "no"',
        'Your data → "no"',
    ],
    ExperimentBlock.DECEITFUL_RESPONSE_TO_FAKE_IDENTITY: [
        'celebrity data → "yes"',
        'random person data → "no"',
        'data of the person you are impersonating → "yes"',
    ],
}
PARTICIPANT_RESPONSE_TRANSLATIONS = {
    ParticipantResponse.YES: '"yes"',
    ParticipantResponse.NO: '"no"',
}