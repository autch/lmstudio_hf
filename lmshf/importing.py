"""Importing a cached Hugging Face model into LM Studio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .hfcache import find_models
from .links import link_into, remove_path
from .lmstudio import existing_models
from .paths import hf_cache_dir, lm_studio_models_dir
from .termui import Choice, select_many


@dataclass
class ImportCandidate:
    """A cached Hugging Face model and where it would live in LM Studio."""

    model_type: str
    name: str
    imported: bool
    snapshot: Path
    target: Path


def _candidates(found_models, lm_studio_dir):
    existing = existing_models(lm_studio_dir)
    candidates = []
    for model_type, name, snapshot_path in sorted(found_models):
        # Check for the exact model path as it would be created
        target = existing.get(name)
        imported = target is not None
        if target is None:
            target = lm_studio_dir.joinpath(*name.split("/"))
        candidates.append(ImportCandidate(model_type, name, imported, snapshot_path, target))
    return candidates


def import_plan(snapshot):
    """The (source, link name) pairs to create for one snapshot.

    Some repositories give each quantisation its own subdirectory. LM Studio
    indexes a model by its path under the models directory and expects
    publisher/repo/file.gguf, so linking such a directory as it stands puts
    the model one level too deep and LM Studio names it after the directory
    ("iq4_xs") instead of after the model. Those GGUF files are flattened
    into the model directory; every other entry is linked as it is.
    """
    try:
        items = sorted(snapshot.iterdir())
    except OSError:
        return []

    entries = []
    nested = []
    for item in items:
        if item.is_dir():
            contained = sorted(item.glob("*.gguf"))
            if contained:
                nested.append((item, contained))
                continue
        entries.append((item, item.name))

    taken = {name for _, name in entries}
    claims = {}
    for directory, contained in nested:
        for path in contained:
            claims.setdefault(path.name, []).append((directory, path))

    for name, holders in sorted(claims.items()):
        # Two quantisation directories can hold the same file name. Prefix
        # every one of them rather than only the later ones, so the same
        # repository never yields "model.gguf" next to "Q5_K_M-model.gguf".
        prefix = len(holders) > 1 or name in taken
        for directory, path in holders:
            entries.append((path, f"{directory.name}-{name}" if prefix else name))
    return entries


def _import_model(candidate):
    """Link every file of the snapshot into the LM Studio models directory."""
    candidate.target.mkdir(parents=True, exist_ok=True)
    method = "symlink"
    try:
        for source, name in import_plan(candidate.snapshot):
            method = link_into(source, candidate.target / name)
    except OSError as exc:
        print(f"Failed to import {candidate.name}: {exc}")
        # Don't leave a half-linked model behind for LM Studio to find.
        try:
            remove_path(candidate.target)
        except OSError:
            pass
        return
    print(f"Imported {candidate.name} ({method}ed files)")


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

    candidates = _candidates(found_models, lm_studio_dir)
    choices = [
        Choice(
            label=f"({c.model_type}) {c.name}" + (" (already imported)" if c.imported else ""),
            marked=c.imported,
            value=c,
        )
        for c in candidates
    ]

    result = select_many(
        choices,
        header="lm-studio - Hugging Face Model Manager",
        instructions=None,
    )
    if result.cancelled:
        print("\nImport is cancelled. Do nothing.")
        return
    print("\nImporting models...\n")

    for i in result.indices:
        candidate = candidates[i]
        if candidate.imported:
            # Selecting an already imported model removes it again.
            try:
                remove_path(candidate.target)
            except OSError as exc:
                print(f"Failed to remove {candidate.name}: {exc}")
                continue
            print(f"Removed {candidate.name}")
        else:
            _import_model(candidate)
