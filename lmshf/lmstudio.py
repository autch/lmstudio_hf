"""The LM Studio side: what is already in the models directory."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from . import gguf, mmproj

# A split model is "name-00002-of-00005.gguf"; only the first part carries
# the full header, and every part is about the same size.
_SPLIT_PART = re.compile(r"^(?P<stem>.+)-(?P<part>[0-9]{5})-of-[0-9]{5}[.]gguf$",
                         re.IGNORECASE)


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
    nested: list = field(default_factory=list)  # GGUFs found below the model directory
    size: int = 0  # the text model, counting every part of a split one

    @property
    def has_projector(self):
        return bool(self.projectors)

    @property
    def attached(self):
        """The projector record this tool wrote for the directory, if any."""
        return (mmproj.read_sidecar(self.path).get("mmproj") or {})


def _is_split_continuation(name):
    match = _SPLIT_PART.match(name)
    return bool(match) and int(match.group("part")) != 1


def _split_stem(name):
    match = _SPLIT_PART.match(name)
    return match.group("stem") if match else None


def _set_size(infos, primary):
    """The size of a model, counting every part when it is a split one."""
    stem = _split_stem(primary.path.name)
    if stem is None:
        return primary.size
    return sum(info.size for info in infos if _split_stem(info.path.name) == stem)


def gguf_files(path):
    """Every GGUF in a model directory, including one level of subdirectory.

    A directory per quantisation is a layout LM Studio itself will index, so
    a model laid out that way has to be visible here even though importing
    now flattens it.
    """
    try:
        return sorted(path.glob("*.gguf")) + sorted(path.glob("*/*.gguf"))
    except OSError:
        return []


def describe(name, path, paths=None, progress=None):
    """Read every GGUF in one model directory and sort out what it holds."""
    model = LmModel(name=name, path=path)
    if paths is None:
        paths = gguf_files(path)
    model.nested = [p for p in paths if p.parent != path]

    models = []
    parts = []
    for gguf_path in paths:
        if progress is not None:
            progress.step(f"{name}  {gguf_path.name}")
        info = gguf.inspect(gguf_path)
        parts.append(info)
        if info.is_projector:
            model.projectors.append(info)
        elif not _is_split_continuation(gguf_path.name):
            models.append(info)

    if models:
        # The largest file is the model proper; anything else in the folder is
        # a companion, such as the MTP/draft module Gemma 4 repos ship.
        models.sort(key=lambda info: info.size, reverse=True)
        model.text, model.extras = models[0], models[1:]
        model.size = _set_size(parts, model.text)
    return model


def scan(lm_studio_dir, progress=None):
    """Describe every model directory under the LM Studio models directory."""
    # Listing the files is cheap; reading their headers is what takes time,
    # so the count is known before any of the waiting starts.
    entries = [(name, path, gguf_files(path))
               for name, path in sorted(existing_models(lm_studio_dir).items())]
    if progress is not None:
        progress.start(sum(len(paths) for _, _, paths in entries))
    return [describe(name, path, paths, progress) for name, path, paths in entries]
