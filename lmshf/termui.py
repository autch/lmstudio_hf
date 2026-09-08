"""Terminal helpers: single keypress input and a scrolling selection menu.

The menu knows nothing about models; callers hand it `Choice` rows and get
back the indices they picked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import shutil
import sys
import unicodedata

from . import IS_WINDOWS

# Logical keys returned by get_key(). Any other printable character is
# returned as itself (lowercased), so callers can offer extra commands.
KEY_UP = "UP"
KEY_DOWN = "DOWN"
KEY_SPACE = "SPACE"
KEY_ENTER = "ENTER"
KEY_INTERRUPT = "INTERRUPT"
KEY_OTHER = "OTHER"

ANSI_RED = "\033[31m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"

NOT_SELECTABLE = "That row cannot be selected."


def _pick_glyphs(preferred, fallback):
    """Return `preferred` if the current stdout encoding can render all of it.

    The glyphs are chosen as a group so a partially encodable set (cp932 has
    the empty circle but not the filled one, say) doesn't produce a mismatch.
    """
    encoding = sys.stdout.encoding or "ascii"
    try:
        "".join(preferred).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return fallback
    return preferred


# A Windows console on a legacy code page (e.g. cp932) cannot encode these, and
# neither can a redirected stdout.
GLYPH_PROMPT = _pick_glyphs("❯", ">")
GLYPH_CHECKED, GLYPH_UNCHECKED = _pick_glyphs(("◉", "○"), ("[x]", "[ ]"))
GLYPH_ARROWS = _pick_glyphs("↑/↓", "Up/Down")
GLYPH_ELLIPSIS = _pick_glyphs("…", "...")


def display_width(text):
    """Columns `text` occupies; CJK characters take two of them."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def fit(text, width):
    """Cut `text` to `width` columns so a row never wraps and breaks the layout."""
    if width <= 0 or display_width(text) <= width:
        return text
    budget = width - display_width(GLYPH_ELLIPSIS)
    out = []
    used = 0
    for ch in text:
        step = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if used + step > budget:
            break
        out.append(ch)
        used += step
    return "".join(out) + GLYPH_ELLIPSIS


def configure_output():
    """Stop a name the console cannot encode from killing the run.

    Model directories and repository names come from whoever published them,
    so a report can carry Chinese, Korean or accented characters that a cp932
    or cp1252 console has no encoding for. The default is to raise
    UnicodeEncodeError halfway through the output; escaping the characters
    instead keeps the rest of the report, and keeps the name recoverable.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            pass  # already replaced, or not a text stream we can reconfigure


def enable_ansi():
    """Enable VT escape sequence processing on legacy Windows consoles."""
    if not IS_WINDOWS:
        return
    import ctypes
    from ctypes import wintypes

    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    STD_OUTPUT_HANDLE = -11
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        if handle in (0, -1):
            return
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return  # not a console (piped or redirected)
        kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except (OSError, AttributeError):
        pass


def get_key():
    """Read a single keypress and return one of the KEY_* constants."""
    if IS_WINDOWS:
        return _get_key_windows()
    return _get_key_posix()


def _get_key_windows():
    import msvcrt

    try:
        ch = msvcrt.getwch()
    except KeyboardInterrupt:
        return KEY_INTERRUPT
    if ch in ("\x00", "\xe0"):  # extended key: the second call carries the code
        code = msvcrt.getwch()
        if code == "H":
            return KEY_UP
        if code == "P":
            return KEY_DOWN
        return KEY_OTHER
    return _classify(ch)


def _get_key_posix():
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    if ch == "\x1b[A":
        return KEY_UP
    if ch == "\x1b[B":
        return KEY_DOWN
    return _classify(ch)


def _classify(ch):
    if ch == " ":
        return KEY_SPACE
    if ch in ("\r", "\n"):
        return KEY_ENTER
    if ch == "\x03":
        return KEY_INTERRUPT
    if ch.isprintable():
        # A bare character, so callers can bind extra commands. The KEY_*
        # constants are multi-character, so they can never collide.
        return ch.lower()
    return KEY_OTHER


@dataclass
class Choice:
    """One row of a menu."""

    label: str
    detail: str = ""  # optional dimmed second line
    marked: bool = False  # red: already imported, incompatible, ...
    selectable: bool = True
    value: object = None


@dataclass
class Selection:
    """What a menu returned."""

    indices: list = field(default_factory=list)
    key: str = ""  # an extra key the caller asked to be told about
    cursor: int = 0
    cancelled: bool = False

    @property
    def index(self):
        return self.indices[0] if self.indices else None


def require_tty():
    # msvcrt reads the console directly, so without one it would block forever
    # rather than fail the way termios does.
    if not sys.stdin.isatty():
        print("This tool needs an interactive terminal.")
        sys.exit(1)


def select_many(choices, header, instructions=None, footer=None, cursor=0, extra_keys=()):
    """Multi-select menu; returns the indices whose rows were checked."""
    return _menu(choices, header, instructions, footer, True, extra_keys, cursor)


def select_one(choices, header, instructions=None, footer=None, cursor=0, extra_keys=()):
    """Single-select menu; ENTER picks the row under the cursor."""
    return _menu(choices, header, instructions, footer, False, extra_keys, cursor)


def _default_instructions(multi):
    # Callers that bind extra keys pass their own instruction line instead.
    pick = "SPACE to select, ENTER to confirm" if multi else "ENTER to choose"
    return f"{GLYPH_ARROWS} to move, {pick}, Ctrl+C to quit"


def _menu(choices, header, instructions, footer, multi, extra_keys, cursor):
    require_tty()
    enable_ansi()

    if not choices:
        return Selection(cancelled=True)

    selected = [False] * len(choices)
    idx = max(0, min(cursor, len(choices) - 1))
    note = ""
    if instructions is None:
        instructions = _default_instructions(multi)
    # Detail text can itself run to two lines (a label plus a warning).
    row_height = 1 + max(len(c.detail.splitlines()) for c in choices)

    while True:
        chrome = 3 + len(header.splitlines()) + (2 if footer else 0) + (1 if note else 0)
        # os.get_terminal_size() raises OSError on Windows when stdout is not a
        # console; shutil's variant falls back to a sane default instead.
        size = shutil.get_terminal_size()
        window = max(1, (size.lines - chrome) // row_height)
        _render(choices, header, instructions, footer, note, selected, idx, multi,
                window, size.columns)
        note = ""

        try:
            key = get_key()
        except KeyboardInterrupt:
            key = KEY_INTERRUPT

        if key == KEY_UP:
            idx = max(0, idx - 1)
        elif key == KEY_DOWN:
            idx = min(len(choices) - 1, idx + 1)
        elif key == KEY_SPACE and multi:
            if choices[idx].selectable:
                selected[idx] = not selected[idx]
            else:
                note = NOT_SELECTABLE
        elif key == KEY_ENTER:
            if multi:
                picked = [i for i, on in enumerate(selected) if on]
                return Selection(indices=picked, cursor=idx)
            if choices[idx].selectable:
                return Selection(indices=[idx], cursor=idx)
            note = NOT_SELECTABLE
        elif key == KEY_INTERRUPT:
            return Selection(cursor=idx, cancelled=True)
        elif key in extra_keys:
            return Selection(key=key, cursor=idx)


def _render(choices, header, instructions, footer, note, selected, idx, multi,
            window, columns=80):
    print("\033[H\033[J", end="")
    print(f"{GLYPH_PROMPT} {header}")
    print(instructions)
    print()

    start = max(0, min(idx - window + 3, len(choices) - window))
    for i in range(start, min(start + window, len(choices))):
        choice = choices[i]
        if multi:
            marker = GLYPH_CHECKED if selected[i] else GLYPH_UNCHECKED
        else:
            marker = GLYPH_CHECKED if i == idx else GLYPH_UNCHECKED
        # "> " + marker + " " sits in front of the label; the marker is
        # three columns wide when the ASCII fallback glyphs are in use.
        label = fit(choice.label, columns - 3 - display_width(marker))
        if choice.marked:
            label = f"{ANSI_RED}{label}{ANSI_RESET}"
        elif not choice.selectable:
            label = f"{ANSI_DIM}{label}{ANSI_RESET}"
        print(f"{'>' if i == idx else ' '} {marker} {label}")
        for line in choice.detail.splitlines():
            print(f"      {ANSI_DIM}{fit(line, columns - 6)}{ANSI_RESET}")

    if footer:
        print()
        print(f"{ANSI_DIM}{footer}{ANSI_RESET}")
    if note:
        print(note)
