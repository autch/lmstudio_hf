"""Menu behaviour, driven by a scripted key sequence instead of a terminal."""

import io
import re
import sys
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
        _, out = run_menu(termui.select_one, ROWS, [termui.KEY_ENTER], footer="2 of 3 shown")
        self.assertIn("2 of 3 shown", out)


class FitTest(unittest.TestCase):
    def test_short_text_is_untouched(self):
        self.assertEqual(termui.fit("abc", 10), "abc")

    def test_long_text_is_cut_to_width(self):
        cut = termui.fit("a" * 40, 10)
        self.assertLessEqual(termui.display_width(cut), 10)
        self.assertTrue(cut.endswith(termui.GLYPH_ELLIPSIS))

    def test_cjk_counts_as_two_columns(self):
        # Model names and repository names are ASCII, but a path can hold
        # anything, and a row that miscounts its width wraps and breaks the
        # fixed-height window. The literals here are test data, not UI text.
        wide = chr(0x4E00) * 5  # a full-width ideograph, two columns each
        self.assertEqual(termui.display_width(wide), 10)
        self.assertLessEqual(termui.display_width(termui.fit(wide * 3, 10)), 10)

    def test_rows_do_not_wrap(self):
        rows = [Choice(label="x" * 200, detail="y" * 200, marked=True)]
        _, out = run_menu(termui.select_one, rows, [termui.KEY_ENTER])
        for line in out.splitlines():
            # Colour codes take no columns, so measure what is actually shown.
            visible = re.sub(chr(27) + "[[][0-9;]*[A-Za-z]", "", line)
            self.assertLessEqual(termui.display_width(visible), 80)


class ConfigureOutputTest(unittest.TestCase):
    """A name the console cannot encode must not end the run."""

    HANZI = chr(0x7B80)  # simplified Chinese, absent from cp932
    ESCAPED = (chr(92) + "u7b80").encode("ascii")

    def console(self):
        raw = io.BytesIO()
        return raw, io.TextIOWrapper(raw, encoding="cp932", errors="strict", newline="")

    def test_a_strict_console_would_raise(self):
        raw, stream = self.console()
        with self.assertRaises(UnicodeEncodeError):
            stream.write(self.HANZI)
            stream.flush()

    def test_configured_output_escapes_instead(self):
        raw, stream = self.console()
        with mock.patch.object(sys, "stdout", stream), mock.patch.object(sys, "stderr", stream):
            termui.configure_output()
            print(f"model: {self.HANZI}.gguf")
            stream.flush()
        self.assertIn(self.ESCAPED, raw.getvalue())
        self.assertIn(b"model: ", raw.getvalue())

    def test_a_stream_that_cannot_be_reconfigured_is_left_alone(self):
        with mock.patch.object(sys, "stdout", io.StringIO()):
            termui.configure_output()  # must not raise


class CombiningMarkTest(unittest.TestCase):
    def test_a_combining_mark_adds_no_width(self):
        composed = chr(0x30AC)  # a single character
        decomposed = chr(0x30AB) + chr(0x3099)  # the same, as base + mark
        self.assertEqual(termui.display_width(composed), 2)
        self.assertEqual(termui.display_width(decomposed), 2)

    def test_fit_measures_decomposed_text_the_same_way(self):
        decomposed = (chr(0x30AB) + chr(0x3099)) * 10
        self.assertLessEqual(termui.display_width(termui.fit(decomposed, 10)), 10)
