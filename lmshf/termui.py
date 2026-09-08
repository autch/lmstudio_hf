import shutil
import sys

from . import IS_WINDOWS

# Logical keys returned by get_key()
KEY_UP = "UP"
KEY_DOWN = "DOWN"
KEY_SPACE = "SPACE"
KEY_ENTER = "ENTER"
KEY_INTERRUPT = "INTERRUPT"
KEY_OTHER = "OTHER"
def _pick_glyphs(preferred, fallback):
    """Return `preferred` if the current stdout encoding can render all of it.

    The glyphs are chosen as a group so a partially encodable set (cp932 has ○
    but not ◉, say) doesn't produce a mismatched pair.
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
    return KEY_OTHER


def select_models(model_choices):
    # msvcrt reads the console directly, so without one it would block forever
    # rather than fail the way termios does.
    if not sys.stdin.isatty():
        print("This tool needs an interactive terminal.")
        sys.exit(1)

    enable_ansi()
    # Don't pre-select any models
    selected = [False] * len(model_choices)
    idx = 0
    # os.get_terminal_size() raises OSError on Windows when stdout is not a
    # console; shutil's variant falls back to a sane default instead.
    window_size = max(1, shutil.get_terminal_size().lines - 5)

    while True:
        print("\033[H\033[J", end="")
        print(
            f"{GLYPH_PROMPT} lm-studio - Hugging Face Model Manager \n"
            f"Available models ({GLYPH_ARROWS} to navigate, SPACE to select, "
            "ENTER to confirm, Ctrl+C to quit):"
        )

        window_start = max(0, min(idx - window_size + 3, len(model_choices) - window_size))
        window_end = min(window_start + window_size, len(model_choices))

        for i in range(window_start, window_end):
            display_name, _, is_imported, _, _ = model_choices[i]
            # Use red color for already imported models
            if is_imported:
                color = "\033[31m"  # Red
                reset = "\033[0m"   # Reset color
                display_text = f"{color}{display_name}{reset}"
            else:
                display_text = display_name
            marker = GLYPH_CHECKED if selected[i] else GLYPH_UNCHECKED
            print(f"{'>' if i == idx else ' '} {marker} {display_text}")

        try:
            key = get_key()
        except KeyboardInterrupt:
            key = KEY_INTERRUPT

        if key == KEY_UP:
            idx = max(0, idx - 1)
        elif key == KEY_DOWN:
            idx = min(len(model_choices) - 1, idx + 1)
        elif key == KEY_SPACE:
            selected[idx] = not selected[idx]
        elif key == KEY_ENTER:
            break
        elif key == KEY_INTERRUPT:
            print("\nImport is cancelled. Do nothing.")
            sys.exit(0)

    return [choice for choice, is_selected in zip(model_choices, selected) if is_selected]
