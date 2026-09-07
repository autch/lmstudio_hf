import json
import os
from pathlib import Path
import sys
import shutil

IS_WINDOWS = os.name == "nt"

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


def hf_cache_dir():
    """Priority: HF_HOME > XDG_CACHE_HOME/huggingface > ~/.cache/huggingface"""
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"])
    if "XDG_CACHE_HOME" in os.environ:
        return Path(os.environ["XDG_CACHE_HOME"]) / "huggingface"
    return Path(os.path.expanduser("~/.cache/huggingface"))


def lm_studio_models_dir():
    """Priority: LMSTUDIO_HOME > settings.json downloadsFolder > platform default."""
    if "LMSTUDIO_HOME" in os.environ:
        return Path(os.environ["LMSTUDIO_HOME"])

    settings_path = Path(os.path.expanduser("~/.lmstudio/settings.json"))
    try:
        with open(settings_path, encoding="utf-8") as f:
            downloads_folder = json.load(f).get("downloadsFolder")
        if downloads_folder:
            return Path(downloads_folder)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        pass

    if IS_WINDOWS:
        return Path(os.path.expanduser("~/.lmstudio/models"))
    return Path(os.path.expanduser("~/.cache/lm-studio/models"))


def _is_reparse_point(path):
    """True for a Windows symlink or junction (Path.is_symlink misses junctions)."""
    if not IS_WINDOWS:
        return False
    try:
        return bool(getattr(os.lstat(path), "st_reparse_tag", 0))
    except OSError:
        return False


def remove_path(path):
    """Delete a file, directory, symlink or junction without following it."""
    if path.is_symlink() or _is_reparse_point(path):
        # Never rmtree a link: that would delete the content it points at.
        if os.path.isdir(path):
            os.rmdir(path)
        else:
            path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def link_into(src, dst):
    """Link `src` at `dst`, returning the mechanism used.

    Entries inside a Hugging Face snapshot are themselves symlinks into
    ../../blobs, so resolve them and point straight at the blob.

    Nothing here ever copies: the point of the tool is to avoid a second copy
    of the weights on disk. If no kind of link can be made, this raises.
    """
    target = src.resolve()
    target_is_dir = target.is_dir()
    try:
        os.symlink(target, dst, target_is_directory=target_is_dir)
        return "symlink"
    except OSError as exc:
        # 1314 = ERROR_PRIVILEGE_NOT_HELD, 5 = ERROR_ACCESS_DENIED
        if not IS_WINDOWS or getattr(exc, "winerror", None) not in (5, 1314):
            raise

    # Symlinks on Windows need Developer Mode or an elevated shell. Junctions
    # and hard links need neither, so fall back to those.
    if target_is_dir:
        import _winapi

        _winapi.CreateJunction(str(target), str(dst))
        return "junction"
    try:
        os.link(target, dst)
        return "hardlink"
    except OSError as exc:
        raise OSError(
            f"Cannot link {target} -> {dst}. Enable Developer Mode in Windows "
            "Settings so symlinks can be created, or keep the Hugging Face "
            "cache and the LM Studio models directory on the same NTFS volume "
            f"so hard links can be used. Original error: {exc}"
        ) from exc


def find_models(cache_dir):
    """Collect (model_type, model_name, snapshot_path) for every cached model."""
    found_models = set()
    # `hub` lives under cache_dir, so a single walk covers both layouts.
    prune = {"blobs", "refs", "snapshots", ".no_exist", "xet"}

    for root, dirs, _ in os.walk(cache_dir):
        for d in list(dirs):
            # Skip datasets directories
            if d.startswith("datasets--") or "datasets" in Path(root).parts:
                continue

            # Check for models--org--name pattern
            if d.startswith("models--") and "--" in d:
                model_dir = Path(root) / d
                # Extract model name from models--org--name format
                parts = d.replace("models--", "").split("--")
                model_name = "/".join(parts)
            elif "--" in d:  # Check for other HF naming patterns
                model_dir = Path(root) / d
                parts = d.split("--")
                model_name = "/".join(parts)
            else:
                continue

            # Look for snapshots directory
            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.exists():
                continue

            # Search for snapshots - with or without config.json
            model_type = "unknown"
            snapshot_path = None

            # Walk through snapshots directory
            for config_root, snapshot_dirs, config_files in os.walk(snapshots_dir):
                # Skip the snapshots directory itself, look in subdirectories
                if config_root == str(snapshots_dir) and snapshot_dirs:
                    continue

                # If we found files in a snapshot subdirectory, this is a valid model
                if config_files or snapshot_dirs:
                    snapshot_path = Path(config_root)

                    # Try to find config.json for model type
                    if "config.json" in config_files:
                        config_path = Path(config_root) / "config.json"
                        try:
                            with open(config_path, encoding="utf-8") as f:
                                config = json.load(f)
                                model_type = config.get("model_type", "unknown").lower()
                        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                            pass
                    break

            if snapshot_path and model_name:
                # Store model even if no config.json was found
                found_models.add((model_type, model_name, snapshot_path))

        # Don't descend into the bulky internals of a model repo.
        dirs[:] = [d for d in dirs if d not in prune]

    return found_models


def manage_models():
    "Import models from the Hugging Face cache."
    cache_dir = hf_cache_dir()
    lm_studio_dir = lm_studio_models_dir()

    # Importing into the Hugging Face cache itself would have the tool link
    # models on top of their own source files.
    try:
        overlaps = lm_studio_dir.resolve().is_relative_to(cache_dir.resolve())
    except (OSError, ValueError):
        overlaps = False
    if overlaps:
        print(
            f"The LM Studio models directory ({lm_studio_dir}) is inside the "
            f"Hugging Face cache ({cache_dir}).\n"
            "Point LM Studio's models directory somewhere else, or set "
            "LMSTUDIO_HOME, and run again."
        )
        return

    found_models = find_models(cache_dir)

    if not found_models:
        print("No models found in Hugging Face cache")
        return

    # First, scan all existing models in LM Studio directory recursively
    existing_lm_models = {}  # Maps normalized names to actual paths
    if lm_studio_dir.exists():
        # Check for org/model structure (subdirectories)
        for org_dir in lm_studio_dir.iterdir():
            if org_dir.is_dir():
                # Check if this is an org directory with model subdirectories
                has_subdirs = False
                try:
                    for model_dir in org_dir.iterdir():
                        if model_dir.is_dir():
                            has_subdirs = True
                            # This is org/model format
                            model_name = f"{org_dir.name}/{model_dir.name}"
                            existing_lm_models[model_name] = model_dir
                except OSError:
                    pass  # Handle permission errors

                # If no subdirectories, this might be a direct model directory
                if not has_subdirs:
                    existing_lm_models[org_dir.name] = org_dir

    # Create list of models with their current import status
    model_choices = []
    for model_type, model, snapshot_path in sorted(found_models):
        is_imported = False
        actual_target_path = None

        # Check for the exact model path as it would be created
        if model in existing_lm_models:
            is_imported = True
            actual_target_path = existing_lm_models[model]

        # If not found, use default path for new imports or removals
        if actual_target_path is None:
            actual_target_path = lm_studio_dir.joinpath(*model.split("/"))

        status = " (already imported)" if is_imported else ""
        display_name = f"({model_type}) {model}{status}"
        model_choices.append((display_name, model, is_imported, snapshot_path, actual_target_path))

    # Show interactive selection menu
    selected = select_models(model_choices)
    print("\nImporting models...\n")

    for display_name, model_name, is_imported, snapshot_path, target_path in selected:

        if is_imported:
            # Remove existing directory or symlink
            try:
                remove_path(target_path)
            except OSError as exc:
                print(f"Failed to remove {model_name}: {exc}")
                continue
            print(f"Removed {model_name}")

        else:
            # Create parent directories and target directory
            target_path.mkdir(parents=True, exist_ok=True)

            # Link every file in the snapshot directory
            method = "symlink"
            try:
                for item in snapshot_path.iterdir():
                    method = link_into(item, target_path / item.name)
            except OSError as exc:
                print(f"Failed to import {model_name}: {exc}")
                # Don't leave a half-linked model behind for LM Studio to find.
                try:
                    remove_path(target_path)
                except OSError:
                    pass
                continue

            print(f"Imported {model_name} ({method}ed files)")


if __name__ == "__main__":
    manage_models()
