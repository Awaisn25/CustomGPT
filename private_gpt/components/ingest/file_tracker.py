"""File tracker for maintaining file path to document ID mappings.

This module provides a thread-safe tracking mechanism for mapping file paths
to their ingested document IDs, enabling efficient lookup for deletion operations.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any

from private_gpt.paths import local_data_path

logger = logging.getLogger(__name__)


class FileTracker:
    """Thread-safe tracker for file path to document ID mappings.

    Stores mappings persistently on disk to survive application restarts.
    Each collection has its own tracking file.
    """

    def __init__(self, tracker_dir: Path | None = None) -> None:
        """Initialize the file tracker.

        Args:
            tracker_dir: Directory to store tracking files. Defaults to local_data_path/file_tracker.
        """
        self._tracker_dir = tracker_dir or (local_data_path / "file_tracker")
        self._tracker_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # In-memory cache: {collection_name: {file_path: [doc_ids]}}
        self._cache: dict[str, dict[str, list[str]]] = {}

    def _get_tracker_file(self, collection_name: str) -> Path:
        """Get the tracker file path for a collection."""
        # Sanitize collection name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in collection_name)
        return self._tracker_dir / f"{safe_name}_tracker.json"

    def _load_collection_data(self, collection_name: str) -> dict[str, list[str]]:
        """Load tracking data for a collection from disk."""
        if collection_name in self._cache:
            return self._cache[collection_name]

        tracker_file = self._get_tracker_file(collection_name)
        if tracker_file.exists():
            try:
                with open(tracker_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._cache[collection_name] = data
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    f"Failed to load tracker file for collection {collection_name}: {e}. "
                    "Starting with empty tracker."
                )

        self._cache[collection_name] = {}
        return self._cache[collection_name]

    def _save_collection_data(self, collection_name: str) -> None:
        """Save tracking data for a collection to disk."""
        tracker_file = self._get_tracker_file(collection_name)
        try:
            data = self._cache.get(collection_name, {})
            with open(tracker_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.error(
                f"Failed to save tracker file for collection {collection_name}: {e}"
            )

    def _normalize_path(self, file_path: Path | str) -> str:
        """Normalize a file path for consistent lookup."""
        path = Path(file_path)
        # Resolve to absolute path and normalize
        try:
            resolved = path.resolve()
            return str(resolved)
        except (OSError, ValueError):
            # Fallback to string representation if resolution fails
            return str(path)

    def track_file(
        self,
        file_path: Path | str,
        doc_ids: list[str],
        collection_name: str,
    ) -> None:
        """Track a file's document IDs for a collection.

        Args:
            file_path: Path to the file
            doc_ids: List of document IDs associated with this file
            collection_name: Name of the collection
        """
        normalized_path = self._normalize_path(file_path)

        with self._lock:
            data = self._load_collection_data(collection_name)
            data[normalized_path] = doc_ids
            self._save_collection_data(collection_name)
            logger.debug(
                f"Tracked {len(doc_ids)} doc(s) for file {normalized_path} "
                f"in collection {collection_name}"
            )

    def get_doc_ids(
        self,
        file_path: Path | str,
        collection_name: str,
    ) -> list[str]:
        """Get document IDs for a file in a collection.

        Args:
            file_path: Path to the file
            collection_name: Name of the collection

        Returns:
            List of document IDs, or empty list if not found
        """
        normalized_path = self._normalize_path(file_path)

        with self._lock:
            data = self._load_collection_data(collection_name)
            return data.get(normalized_path, [])

    def untrack_file(
        self,
        file_path: Path | str,
        collection_name: str,
    ) -> list[str]:
        """Remove tracking for a file and return its document IDs.

        Args:
            file_path: Path to the file
            collection_name: Name of the collection

        Returns:
            List of document IDs that were tracked, or empty list if not found
        """
        normalized_path = self._normalize_path(file_path)

        with self._lock:
            data = self._load_collection_data(collection_name)
            doc_ids = data.pop(normalized_path, [])
            if doc_ids:
                self._save_collection_data(collection_name)
                logger.debug(
                    f"Untracked {len(doc_ids)} doc(s) for file {normalized_path} "
                    f"from collection {collection_name}"
                )
            return doc_ids

    def get_all_tracked_files(self, collection_name: str) -> dict[str, list[str]]:
        """Get all tracked files for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary mapping file paths to their document IDs
        """
        with self._lock:
            data = self._load_collection_data(collection_name)
            return dict(data)  # Return a copy

    def is_file_tracked(
        self,
        file_path: Path | str,
        collection_name: str,
    ) -> bool:
        """Check if a file is tracked in a collection.

        Args:
            file_path: Path to the file
            collection_name: Name of the collection

        Returns:
            True if the file is tracked, False otherwise
        """
        normalized_path = self._normalize_path(file_path)

        with self._lock:
            data = self._load_collection_data(collection_name)
            return normalized_path in data

    def clear_collection(self, collection_name: str) -> None:
        """Clear all tracking data for a collection.

        Args:
            collection_name: Name of the collection
        """
        with self._lock:
            self._cache[collection_name] = {}
            self._save_collection_data(collection_name)
            logger.info(f"Cleared all tracking data for collection {collection_name}")


# Global file tracker instance
_file_tracker: FileTracker | None = None


def get_file_tracker() -> FileTracker:
    """Get the global file tracker instance."""
    global _file_tracker
    if _file_tracker is None:
        _file_tracker = FileTracker()
    return _file_tracker

