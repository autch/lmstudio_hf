"""The LM Studio side: what is already in the models directory."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from . import gguf, mmproj

# A split model is "name-00002-of-00005.gguf"; only the first part carries
# the full header, and every part is about the same size.
_SPLIT_PART = re.compile(r"-([0-9]{5})-of-[0-9]{5}[.]gguf$", re.IGNORECASE)


def existing_models(lm_studio_dir):
    """Map "org/name" (and bare "name") to the directory holding that model."""
    found = {}
    if not lm_studio_dir.exists():
        return found

    # Check for org/model structure (subdirectories)
    for org_dir in lm_studio_dir.iterdir():
        if not org_dir.is_dir():
            continue
        # Check if this is an org directory with model subdirectories
        has_subdirs = False
        try:
            for model_dir in org_dir.iterdir():
                if model_dir.is_dir():
                    has_subdirs = True
                    found[f"{org_dir.name}/{model_dir.name}"] = model_dir
        except OSError:
            pass  # Handle permission errors

        # If no subdirectories, this might be a direct model directory
        if not has_subdirs:
            found[org_dir.name] = org_dir

    return found


@dataclass
class LmModel:
    """One model directory as LM Studio sees it."""

    name: str
    path: object
    text: object = None  # the primary text GGUF, or None
    projectors: list = field(default_factory=list)
    extras: list = field(default_factory=list)  # MTP/draft modules, other quants

    @property
    def has_projector(self):
        return bool(self.projectors)

    @property
    def attached(self):
        """The projector record this tool wrote for the directory, if any."""
        return (mmproj.read_sidecar(self.path).get("mmproj") or {})


def _is_split_continuation(name):
    match = _SPLIT_PART.search(name)
    return bool(match) and int(match.group(1)) != 1


def describe(name, path):
    """Read every GGUF in one model directory and sort out what it holds."""
    model = LmModel(name=name, path=path)
    try:
        paths = sorted(path.glob("*.gguf"))
    except OSError:
        return model

    models = []
    for gguf_path in paths:
        info = gguf.inspect(gguf_path)
        if info.is_projector:
            model.projectors.append(info)
        elif not _is_split_continuation(gguf_path.name):
            models.append(info)

    if models:
        # The largest file is the model proper; anything else in the folder is
        # a companion, such as the MTP/draft module Gemma 4 repos ship.
        models.sort(key=lambda info: info.size, reverse=True)
        model.text, model.extras = models[0], models[1:]
    return model


def scan(lm_studio_dir):
    """Describe every model directory under the LM Studio models directory."""
    return [describe(name, path) for name, path in sorted(existing_models(lm_studio_dir).items())]
