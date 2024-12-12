from typing import Callable, Sequence

ScreenDrawFn = Callable[[], None]


class ScreenSequence:
    def __init__(
        self,
        screen_draw_fn_sequence: Sequence[ScreenDrawFn],
        marked_screen_idxs: int | Sequence[int],
        is_circular: bool,
    ) -> None:
        """
        Args:
            screen_draw_fn_sequence (Sequence[Callable[[], None]]): Iterable of functions to draw screens.
            marked_screen_idx (int | Sequence[int]): Special screen(s) marked for comparison.
            is_circular (bool): Should it go back to the first element after the last one.
        """

        self._screen_draw_fn_sequence = screen_draw_fn_sequence
        self._pointer = 0
        self._marked_screen_idxs = (
            (marked_screen_idxs,)
            if isinstance(marked_screen_idxs, int)
            else marked_screen_idxs
        )
        self._is_circular = is_circular

    def reset(self) -> None:
        self._pointer = 0

    def is_at_marked(self) -> bool:
        """
        Returns:
            bool: If the pointer is currently pointing at the marked screen.
        """

        return self._pointer in self._marked_screen_idxs

    def advance_and_call(self) -> None:
        self._change_pointer_and_call(1)

    def move_back_and_call(self) -> None:
        if self._pointer - 1 not in self._marked_screen_idxs:
            self._change_pointer_and_call(-1)

    def call_current(self) -> None:
        self._screen_draw_fn_sequence[self._pointer]()

    def _change_pointer_and_call(self, to_add: int) -> None:
        """
        Args:
            to_add (int): Number to add to the pointer.
        """

        seq_length = len(self._screen_draw_fn_sequence)

        if self._is_circular:
            self._pointer += to_add
            self._pointer %= seq_length
        else:
            self._pointer = max(0, min(seq_length - 1, self._pointer + to_add))

        self.call_current()
