import json
import os
from pathlib import Path


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
