"""Watched Path Manager for automatic file ingestion and deletion.

This module provides a service that manages file watchers for temporary paths,
automatically ingesting new files and removing deleted files from the vector store.
"""

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from injector import inject, singleton

from private_gpt.components.ingest.file_tracker import FileTracker, get_file_tracker
from private_gpt.server.ingest.ingest_watcher import TemporaryPathWatcher
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.server.ingest.ingest_service import IngestService

logger = logging.getLogger(__name__)


@singleton
class WatchedPathManager:
    """Manager for watched paths with automatic ingestion and deletion.

    This service:
    - Manages file watchers for configured paths
    - Automatically ingests new files into the appropriate collection
    - Automatically removes documents when files are deleted
    - Tracks file-to-document mappings for efficient deletion
    """

    @inject
    def __init__(self, settings: Settings) -> None:
        """Initialize the watched path manager.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._watchers: dict[str, TemporaryPathWatcher] = {}
        self._lock = threading.Lock()
        self._file_tracker: FileTracker = get_file_tracker()
        self._ingest_service: "IngestService | None" = None
        self._started = False

    def set_ingest_service(self, ingest_service: "IngestService") -> None:
        """Set the ingest service reference.

        This is called during startup to avoid circular dependency issues.

        Args:
            ingest_service: The ingest service to use for ingestion/deletion
        """
        self._ingest_service = ingest_service

    def _on_file_created(self, file_path: Path, collection_name: str) -> None:
        """Handle file creation event by ingesting the file.

        Args:
            file_path: Path to the created file
            collection_name: Collection to ingest into
        """
        if self._ingest_service is None:
            logger.error(
                "IngestService not set, cannot process file creation. "
                "This is a configuration error."
            )
            return

        try:
            # Check if file is already tracked (avoid re-ingestion)
            if self._file_tracker.is_file_tracked(file_path, collection_name):
                logger.debug(
                    f"File already tracked, skipping ingestion: {file_path}"
                )
                return

            # Ingest the file
            ingested_docs = self._ingest_service.ingest_file(
                file_name=file_path.name,
                file_data=file_path,
                collection_name=collection_name,
            )

            # Track the file-to-document mapping
            doc_ids = [doc.doc_id for doc in ingested_docs]
            self._file_tracker.track_file(file_path, doc_ids, collection_name)

            logger.info(
                f"Successfully ingested {len(doc_ids)} document(s) from {file_path} "
                f"into collection {collection_name}"
            )
        except Exception as e:
            logger.error(
                f"Failed to ingest file {file_path} into collection {collection_name}: {e}",
                exc_info=True,
            )

    def _on_file_deleted(self, file_path: Path, collection_name: str) -> None:
        """Handle file deletion event by removing documents from the index.

        Args:
            file_path: Path to the deleted file
            collection_name: Collection to remove from
        """
        if self._ingest_service is None:
            logger.error(
                "IngestService not set, cannot process file deletion. "
                "This is a configuration error."
            )
            return

        try:
            # Get tracked document IDs for this file
            doc_ids = self._file_tracker.untrack_file(file_path, collection_name)

            if not doc_ids:
                logger.debug(
                    f"No tracked documents found for {file_path} in collection {collection_name}"
                )
                return

            # Delete each document from the collection
            deleted_count = 0
            for doc_id in doc_ids:
                try:
                    self._ingest_service.delete(doc_id, collection_name=collection_name)
                    deleted_count += 1
                except ValueError as e:
                    logger.warning(
                        f"Document {doc_id} may have already been deleted: {e}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to delete document {doc_id} from collection {collection_name}: {e}"
                    )

            logger.info(
                f"Successfully deleted {deleted_count}/{len(doc_ids)} document(s) "
                f"for file {file_path} from collection {collection_name}"
            )
        except Exception as e:
            logger.error(
                f"Failed to handle file deletion for {file_path}: {e}",
                exc_info=True,
            )

    def _on_file_modified(self, file_path: Path, collection_name: str) -> None:
        """Handle file modification event by re-ingesting the file.

        This deletes the old documents and ingests the new content.

        Args:
            file_path: Path to the modified file
            collection_name: Collection to update
        """
        # Delete old documents
        self._on_file_deleted(file_path, collection_name)
        # Re-ingest the file
        self._on_file_created(file_path, collection_name)

    def start_watcher(
        self,
        watch_path: Path | str,
        collection_name: str,
        watch_modifications: bool = False,
    ) -> bool:
        """Start a watcher for a specific path.

        Args:
            watch_path: Path to watch
            collection_name: Collection name for ingested documents
            watch_modifications: If True, also re-ingest on file modifications

        Returns:
            True if watcher started successfully, False otherwise
        """
        watch_path = Path(watch_path)
        path_key = str(watch_path.resolve())

        with self._lock:
            if path_key in self._watchers:
                logger.warning(f"Watcher already exists for path: {watch_path}")
                return True

            try:
                watcher = TemporaryPathWatcher(
                    watch_path=watch_path,
                    collection_name=collection_name,
                    on_file_created=self._on_file_created,
                    on_file_deleted=self._on_file_deleted,
                    on_file_modified=self._on_file_modified if watch_modifications else None,
                )
                watcher.start(blocking=False)
                self._watchers[path_key] = watcher
                logger.info(f"Started watcher for {watch_path} -> collection {collection_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to start watcher for {watch_path}: {e}", exc_info=True)
                return False

    def stop_watcher(self, watch_path: Path | str) -> bool:
        """Stop a watcher for a specific path.

        Args:
            watch_path: Path to stop watching

        Returns:
            True if watcher stopped successfully, False if not found
        """
        watch_path = Path(watch_path)
        path_key = str(watch_path.resolve())

        with self._lock:
            watcher = self._watchers.pop(path_key, None)
            if watcher is not None:
                watcher.stop()
                logger.info(f"Stopped watcher for {watch_path}")
                return True
            return False

    def start_configured_watchers(self) -> None:
        """Start watchers for all configured paths.

        Reads configuration from settings.data.paths and starts watchers
        for the temporary path if enabled.
        """
        if self._started:
            logger.warning("Watchers already started")
            return

        paths_settings = self.settings.data.paths

        # Check if watcher is enabled
        if not paths_settings.watch_enabled:
            logger.info("File watching is disabled in settings")
            return

        # Start watcher for temporary path
        temp_path = Path(paths_settings.temporary_path)
        temp_collection = paths_settings.temporary_collection_name

        if temp_path.exists() or paths_settings.create_paths_if_missing:
            if not temp_path.exists():
                try:
                    temp_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created temporary path: {temp_path}")
                except OSError as e:
                    logger.error(f"Failed to create temporary path {temp_path}: {e}")
                    return

            success = self.start_watcher(
                watch_path=temp_path,
                collection_name=temp_collection,
                watch_modifications=paths_settings.watch_modifications,
            )
            if success:
                logger.info(
                    f"Started automatic file watching for temporary path: {temp_path}"
                )
            else:
                logger.error(
                    f"Failed to start file watching for temporary path: {temp_path}"
                )
        else:
            logger.warning(
                f"Temporary path does not exist and create_paths_if_missing is False: {temp_path}"
            )

        self._started = True

    def stop_all_watchers(self) -> None:
        """Stop all running watchers."""
        with self._lock:
            for path_key, watcher in list(self._watchers.items()):
                try:
                    watcher.stop()
                except Exception as e:
                    logger.error(f"Error stopping watcher for {path_key}: {e}")
            self._watchers.clear()
            self._started = False
            logger.info("Stopped all file watchers")

    def get_active_watchers(self) -> dict[str, str]:
        """Get information about active watchers.

        Returns:
            Dictionary mapping watched paths to their collection names
        """
        with self._lock:
            return {
                path: watcher.collection_name
                for path, watcher in self._watchers.items()
                if watcher.is_running
            }

    def sync_existing_files(
        self,
        watch_path: Path | str,
        collection_name: str,
    ) -> tuple[int, int]:
        """Sync existing files in a watched path.

        Ingests any files that exist in the path but are not tracked.

        Args:
            watch_path: Path to sync
            collection_name: Collection to ingest into

        Returns:
            Tuple of (ingested_count, error_count)
        """
        if self._ingest_service is None:
            logger.error("IngestService not set, cannot sync files")
            return 0, 0

        watch_path = Path(watch_path)
        ingested_count = 0
        error_count = 0

        if not watch_path.exists():
            logger.warning(f"Watch path does not exist: {watch_path}")
            return 0, 0

        logger.info(f"Syncing existing files in {watch_path} to collection {collection_name}")

        for file_path in watch_path.rglob("*"):
            if file_path.is_dir():
                continue

            if self._file_tracker.is_file_tracked(file_path, collection_name):
                continue

            try:
                self._on_file_created(file_path, collection_name)
                ingested_count += 1
            except Exception as e:
                logger.error(f"Failed to sync file {file_path}: {e}")
                error_count += 1

        logger.info(
            f"Sync complete for {watch_path}: ingested {ingested_count} files, "
            f"{error_count} errors"
        )
        return ingested_count, error_count

