"""Menu behaviour, driven by a scripted key sequence instead of a terminal."""

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from lmshf import termui
from lmshf.termui import Choice


def run_menu(func, choices, keys, **kwargs):
    """Drive `func` with `keys`, returning (selection, printed output)."""
    buf = io.StringIO()
    with mock.patch.object(termui, "require_tty"), \
         mock.patch.object(termui, "get_key", side_effect=list(keys)), \
         redirect_stdout(buf):
        result = func(choices, header="test", **kwargs)
    return result, buf.getvalue()


ROWS = [Choice(label="alpha"), Choice(label="beta"), Choice(label="gamma")]


class SelectManyTest(unittest.TestCase):
    def test_space_toggles_and_enter_confirms(self):
        keys = [termui.KEY_SPACE, termui.KEY_DOWN, termui.KEY_DOWN,
                termui.KEY_SPACE, termui.KEY_ENTER]
        result, _ = run_menu(termui.select_many, ROWS, keys)
        self.assertEqual(result.indices, [0, 2])
        self.assertFalse(result.cancelled)

    def test_toggle_twice_deselects(self):
        keys = [termui.KEY_SPACE, termui.KEY_SPACE, termui.KEY_ENTER]
        result, _ = run_menu(termui.select_many, ROWS, keys)
        self.assertEqual(result.indices, [])

    def test_interrupt_reports_cancelled(self):
        result, _ = run_menu(termui.select_many, ROWS, [termui.KEY_INTERRUPT])
        self.assertTrue(result.cancelled)
        self.assertEqual(result.indices, [])

    def test_cursor_stops_at_the_ends(self):
        keys = [termui.KEY_UP, termui.KEY_UP, termui.KEY_SPACE, termui.KEY_ENTER]
        result, _ = run_menu(termui.select_many, ROWS, keys)
        self.assertEqual(result.indices, [0])


class SelectOneTest(unittest.TestCase):
    def test_enter_picks_the_row_under_the_cursor(self):
        result, _ = run_menu(termui.select_one, ROWS, [termui.KEY_DOWN, termui.KEY_ENTER])
        self.assertEqual(result.index, 1)

    def test_starting_cursor_is_honoured(self):
        result, _ = run_menu(termui.select_one, ROWS, [termui.KEY_ENTER], cursor=2)
        self.assertEqual(result.index, 2)

    def test_unselectable_row_refuses_enter(self):
        rows = [Choice(label="bad", selectable=False), Choice(label="good")]
        keys = [termui.KEY_ENTER, termui.KEY_DOWN, termui.KEY_ENTER]
        result, out = run_menu(termui.select_one, rows, keys)
        self.assertEqual(result.index, 1)
        self.assertIn(termui.NOT_SELECTABLE, out)

    def test_extra_key_returns_to_the_caller(self):
        result, _ = run_menu(termui.select_one, ROWS, [termui.KEY_DOWN, "a"], extra_keys="a")
        self.assertEqual(result.key, "a")
        self.assertEqual(result.cursor, 1)
        self.assertIsNone(result.index)

    def test_unbound_key_is_ignored(self):
        result, _ = run_menu(termui.select_one, ROWS, ["z", termui.KEY_ENTER])
        self.assertEqual(result.index, 0)

    def test_empty_choice_list_is_cancelled(self):
        result, _ = run_menu(termui.select_one, [], [])
        self.assertTrue(result.cancelled)


class RenderTest(unittest.TestCase):
    def test_detail_lines_and_marking_are_rendered(self):
        rows = [Choice(label="alpha", detail="q4_k_m", marked=True), Choice(label="beta")]
        _, out = run_menu(termui.select_one, rows, [termui.KEY_ENTER])
        self.assertIn("q4_k_m", out)
        self.assertIn(termui.ANSI_RED, out)

    def test_footer_is_rendered(self):
        _, out = run_menu(termui.select_one, ROWS, [termui.KEY_ENTER], footer="3 件中 2 件")
        self.assertIn("3 件中 2 件", out)


if __name__ == "__main__":
    unittest.main()
