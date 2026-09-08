"""The LM Studio side: what is already in the models directory."""

from __future__ import annotations


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
