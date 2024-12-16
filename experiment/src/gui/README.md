# GUI Module

## Overview

The `Gui` class is responsible for managing the graphical user interface of the Lie Detector experiment. It handles the display of instructions, experiment data, and feedback to the participant. The GUI is built using the Pygame library and includes various screens and sequences to guide the participant through the experiment.

## Components

### Initialization

The `Gui` class initializes various components required for the experiment, including experiment blocks, screen sequences, response keys, and events.

```python
from .gui import Gui

gui = Gui()
```

### Key Methods

#### `start()`

Starts the GUI and initializes the Pygame display. It also sets up the personal data manager and begins the main loop.

```python
gui.start()
```

#### `_mainloop()`

The main loop of the GUI, which handles events and updates the display.

```python
def _mainloop(self) -> None:
    while self._running:
        self._handle_events()
        self._clock.tick(60)
        pygame.display.update()

    self._personal_data_manager.cleanup()
```

#### `_handle_events()`

Handles Pygame events, such as key presses and custom events.

```python
def _handle_events(self) -> None:
    for event in pygame.event.get():
        match event.type:
            case pygame.QUIT:
                self._running = False
            case pygame.KEYDOWN if event.key == QUIT_KEY and self._can_quit:
                self._running = False
            case pygame.KEYDOWN:
                self._handle_keydown(event.key)
            case self._go_to_next_part_event.type:
                self._go_to_next_part()
            case self._run_block_event.type:
                self._run_block()
            case self._timeout_event.type:
                self._register_response(ParticipantResponse.TIMEOUT)
```

#### `_handle_keydown(key)`

Handles keydown events based on the current state of the GUI.

```python
def _handle_keydown(self, key: int) -> None:
    if not self._main_screens_sequence.is_at_marked() and key in (GO_BACK_KEY, GO_FORWARD_KEY):
        self._handle_screen_change(key)
    elif self._experiment_blocks_sequence.get_current().is_practice and not self._do_handle_participant_response and key == CONFIRMATION_KEY:
        self._go_to_next_part()
    elif self._main_screens_sequence.is_at_marked() and self._experiment_block_parts_sequence.is_at_marked() and self._do_handle_participant_response and key in self._response_keys:
        self._handle_participant_response(key)
    elif self._wait_for_confirmation and key == CONFIRMATION_KEY:
        self._go_to_next_part()
```

#### `_draw_fixation_cross()`

Draws a fixation cross on the screen.

```python
def _draw_fixation_cross(self) -> None:
    screen_width, screen_height = self._get_screen_size()
    fixation_cross_length = int(FIXATION_CROSS_LENGTH_AS_WIDTH_PERCENTAGE * screen_width)
    fixation_cross_width = int(FIXATION_CROSS_WIDTH_AS_WIDTH_PERCENTAGE * screen_width)

    horizontal_x_coord = self._calculate_margin(fixation_cross_length, screen_width)
    horizontal_y_coord = self._calculate_margin(fixation_cross_width, screen_height)
    horizontal_rect = pygame.Rect(horizontal_x_coord, horizontal_y_coord, fixation_cross_length, fixation_cross_width)

    vertical_x_coord = self._calculate_margin(fixation_cross_width, screen_width)
    vertical_y_coord = self._calculate_margin(fixation_cross_length, screen_height)
    vertical_rect = pygame.Rect(vertical_x_coord, vertical_y_coord, fixation_cross_width, fixation_cross_length)

    self._draw_background()
    pygame.draw.rect(self._main_surface, FIXATION_CROSS_COLOR, horizontal_rect)
    pygame.draw.rect(self._main_surface, FIXATION_CROSS_COLOR, vertical_rect)
    self._set_timeout(self._go_to_next_part_event, self._get_random_from_range(FIXATION_CROSS_TIME_RANGE_MILLIS))
```

#### `_draw_experiment_data()`

Draws the experiment data on the screen.

```python
def _draw_experiment_data(self) -> None:
    self._data_to_show = self._personal_data_manager.get_next()
    self._draw_data(self._data_to_show)
    self._set_timeout(self._timeout_event, TIMEOUT_MILLIS)
```

#### `_draw_break_between_practice_and_proper()`

Draws the break screen between practice and proper trials.

```python
def _draw_break_between_practice_and_proper(self) -> None:
    self._draw_data(BREAK_BETWEEN_PRACTICE_AND_PROPER_TEXTS)
    self._set_timeout(self._run_block_event, BREAK_BETWEEN_PRACTICE_AND_PROPER_MILLIS)
```

#### `_draw_break_between_blocks()`

Draws the break screen between experiment blocks.

```python
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
```

#### `_draw_results()`

Draws the results screen at the end of the experiment.

```python
def _draw_results(self) -> None:
    self._can_quit = True
    results = self._personal_data_manager.get_response_counts()
    self._draw_data([RESULTS_TEXT, f"{results.correct} / {results.correct + results.incorrect}"])
```

### Helper Methods

#### `_calculate_margin(element_size, screen_size)`

Calculates the margin for centering an element on the screen.

```python
def _calculate_margin(self, element_size: int, screen_size: int) -> int:
    return (screen_size - element_size) // 2
```

#### `_get_screen_size()`

Returns the size of the screen.

```python
def _get_screen_size(self) -> Size:
    display_info = pygame.display.Info()
    return Size(display_info.current_w, display_info.current_h)
```

#### `_get_random_from_range(range)`

Returns a random integer from the specified range.

```python
def _get_random_from_range(self, range: tuple[int, int]) -> int:
    start, end = range
    return random.randrange(start, end + 1)
```

#### `_set_timeout(event, millis)`

Sets a timeout for a Pygame event.

```python
def _set_timeout(self, event: pygame.event.Event, millis: int) -> None:
    pygame.time.set_timer(event, millis, 1)
```

#### `_clear_timeout(event)`

Clears a timeout for a Pygame event.

```python
def _clear_timeout(self, event: pygame.event.Event) -> None:
    pygame.time.set_timer(event, 0)
```

#### `_sort_by_personal_data_field(personal_data)`

Sorts personal data by the field.

```python
def _sort_by_personal_data_field(self, personal_data: dict[PersonalDataField, str]) -> list[str]:
    personal_data_field_sequence = list(PersonalDataField)
    return [
        data
        for _, data in sorted(
            personal_data.items(),
            key=lambda k: personal_data_field_sequence.index(k[0]),
        )
    ]
```

## Configuration

The configuration for the GUI is defined in `config.py`. Key configuration parameters include:

- **Screen Parameters**:
  - `SCREEN_PARAMS`: Screen size and mode (fullscreen or windowed).
- **Timing**:
  - `FIXATION_CROSS_TIME_RANGE_MILLIS`: Time range for displaying the fixation cross.
  - `TIMEOUT_MILLIS`: Timeout duration for participant responses.
  - `BREAK_BETWEEN_TRIALS_TIME_RANGE_MILLIS`: Time range for breaks between trials.
  - `BREAK_BETWEEN_PRACTICE_AND_PROPER_MILLIS`: Duration of the break between practice and proper trials.
- **Colors**:
  - `BACKGROUND_COLOR_PRIMARY`: Primary background color.
  - `BACKGROUND_COLOR_INCORRECT`: Background color for incorrect responses.
  - `FIXATION_CROSS_COLOR`: Color of the fixation cross.
  - `TEXT_COLOR`: Color of the text.
- **Text**:
  - `TEXT_FONT`: Font for the text.
  - `TEXT_FONT_SIZE_AS_WIDTH_PERCENTAGE`: Font size as a percentage of the screen width.
  - Various text strings for instructions, feedback, and results.

## Example Usage

1. **Initialization**:

   - Initialize the `Gui` class.

   ```python
   from .gui import Gui

   gui = Gui()
   ```

2. **Start the GUI**:

   - Start the GUI and begin the experiment.

   ```python
   gui.start()
   ```

3. **Handle Events**:

   - The GUI will handle events and update the display in the main loop.

   ```python
   def _mainloop(self) -> None:
       while self._running:
           self._handle_events()
           self._clock.tick(60)
           pygame.display.update()

       self._personal_data_manager.cleanup()
   ```

4. **Draw Experiment Data**:

   - Draw the experiment data on the screen.

   ```python
   def _draw_experiment_data(self) -> None:
       self._data_to_show = self._personal_data_manager.get_next()
       self._draw_data(self._data_to_show)
       self._set_timeout(self._timeout_event, TIMEOUT_MILLIS)
   ```

5. **Draw Results**:

   - Draw the results screen at the end of the experiment.

   ```python
   def _draw_results(self) -> None:
       self._can_quit = True
       results = self._personal_data_manager.get_response_counts()
       self._draw_data([RESULTS_TEXT, f"{results.correct} / {results.correct + results.incorrect}"])
   ```

## Conclusion

The `Gui` class provides a comprehensive interface for managing the graphical user interface of the Lie Detector experiment. It handles the display of instructions, experiment data, and feedback to the participant, ensuring a smooth and interactive experience.
