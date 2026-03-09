"""Utility functions for mapping file paths to collection names."""

import os
from pathlib import Path

from private_gpt.settings.settings import Settings, settings


def get_collection_for_path(
    file_path: Path | str, settings_instance: Settings | None = None
) -> str:
    """Determine which collection to use based on file path.

    Args:
        file_path: Path to the file (can be absolute or relative)
        settings_instance: Optional Settings instance. If not provided, uses global settings.

    Returns:
        Collection name to use for this path.

    The function checks if the file path is under the persistent_path or temporary_path
    configured in settings. If neither matches, returns the default collection name.
    """
    if settings_instance is None:
        settings_instance = settings()

    path = Path(file_path)
    # Normalize the path to handle both Windows and Linux paths
    # Convert to absolute path for comparison
    if not path.is_absolute():
        # If relative, we can't determine the collection, use default
        return settings_instance.vectorstore.default_collection_name

    # Normalize paths for comparison (handle D:/ vs D:\ on Windows)
    abs_path = path.resolve()
    abs_path_str = str(abs_path)

    # Normalize persistent and temporary paths
    persistent_path = Path(settings_instance.data.paths.persistent_path).resolve()
    temporary_path = Path(settings_instance.data.paths.temporary_path).resolve()

    persistent_path_str = str(persistent_path)
    temporary_path_str = str(temporary_path)

    # Check if path is under temporary_path (E:/)
    # Use case-insensitive comparison on Windows
    if os.name == "nt":  # Windows
        if abs_path_str.lower().startswith(temporary_path_str.lower()):
            return settings_instance.data.paths.temporary_collection_name
        if abs_path_str.lower().startswith(persistent_path_str.lower()):
            return settings_instance.data.paths.persistent_collection_name
    else:  # Linux/Unix
        if abs_path_str.startswith(temporary_path_str):
            return settings_instance.data.paths.temporary_collection_name
        if abs_path_str.startswith(persistent_path_str):
            return settings_instance.data.paths.persistent_collection_name

    # Default to default collection if path doesn't match either
    return settings_instance.vectorstore.default_collection_name

