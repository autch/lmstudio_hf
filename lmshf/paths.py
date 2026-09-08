import json
import os
from pathlib import Path

from . import IS_WINDOWS


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
