from .hfcache import find_models
from .links import link_into, remove_path
from .paths import hf_cache_dir, lm_studio_models_dir
from .termui import select_models


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

def main():
    manage_models()
