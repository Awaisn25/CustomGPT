"""File system watcher for automatic document ingestion and deletion."""

import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


# Debounce time in seconds to avoid duplicate events
DEBOUNCE_TIME = 1.0


class IngestWatcher:
    """Basic file watcher for ingestion (backward compatible)."""

    def __init__(
        self, watch_path: Path, on_file_changed: Callable[[Path], None]
    ) -> None:
        self.watch_path = watch_path
        self.on_file_changed = on_file_changed

        class Handler(FileSystemEventHandler):
            def on_modified(self, event: FileSystemEvent) -> None:
                if isinstance(event, FileModifiedEvent):
                    on_file_changed(Path(event.src_path))

            def on_created(self, event: FileSystemEvent) -> None:
                if isinstance(event, FileCreatedEvent):
                    on_file_changed(Path(event.src_path))

        event_handler = Handler()
        observer: Any = Observer()
        self._observer = observer
        self._observer.schedule(event_handler, str(watch_path), recursive=True)

    def start(self) -> None:
        self._observer.start()
        while self._observer.is_alive():
            try:
                self._observer.join(1)
            except KeyboardInterrupt:
                break

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()


class TemporaryPathWatcher:
    """Enhanced file watcher for temporary paths with create/delete support.

    This watcher:
    - Monitors a directory for new files and automatically ingests them
    - Monitors for file deletions and removes associated documents
    - Uses debouncing to avoid duplicate events
    - Handles errors gracefully without stopping the watcher
    """

    def __init__(
        self,
        watch_path: Path | str,
        collection_name: str,
        on_file_created: Callable[[Path, str], None],
        on_file_deleted: Callable[[Path, str], None],
        on_file_modified: Callable[[Path, str], None] | None = None,
        supported_extensions: set[str] | None = None,
    ) -> None:
        """Initialize the temporary path watcher.

        Args:
            watch_path: Path to watch for changes
            collection_name: Name of the collection for this path
            on_file_created: Callback when a file is created
            on_file_deleted: Callback when a file is deleted
            on_file_modified: Optional callback when a file is modified
            supported_extensions: Set of file extensions to watch (e.g., {'.pdf', '.txt'}).
                                  If None, watches all files.
        """
        self.watch_path = Path(watch_path)
        self.collection_name = collection_name
        self.on_file_created = on_file_created
        self.on_file_deleted = on_file_deleted
        self.on_file_modified = on_file_modified
        self.supported_extensions = supported_extensions
        self._observer: Any = None
        self._running = False
        self._lock = threading.Lock()
        # Track recent events for debouncing: {path: last_event_time}
        self._recent_events: dict[str, float] = {}

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if a file should be processed based on extension."""
        if self.supported_extensions is None:
            return True
        return file_path.suffix.lower() in self.supported_extensions

    def _should_process_event(self, file_path: Path) -> bool:
        """Check if an event should be processed (debouncing)."""
        path_str = str(file_path)
        current_time = time.time()

        with self._lock:
            last_time = self._recent_events.get(path_str, 0)
            if current_time - last_time < DEBOUNCE_TIME:
                return False
            self._recent_events[path_str] = current_time
            return True

    def _handle_created(self, file_path: Path) -> None:
        """Handle file creation event."""
        if not self._is_supported_file(file_path):
            return

        if not self._should_process_event(file_path):
            return

        # Wait a bit for file write to complete
        time.sleep(0.5)

        if not file_path.exists():
            logger.debug(f"File no longer exists, skipping: {file_path}")
            return

        if file_path.is_dir():
            return

        try:
            logger.info(
                f"File created in watched path, ingesting: {file_path} "
                f"into collection {self.collection_name}"
            )
            self.on_file_created(file_path, self.collection_name)
        except Exception as e:
            logger.error(
                f"Error handling file creation for {file_path}: {e}",
                exc_info=True,
            )

    def _handle_deleted(self, file_path: Path) -> None:
        """Handle file deletion event."""
        if not self._is_supported_file(file_path):
            return

        # Clear from recent events
        with self._lock:
            self._recent_events.pop(str(file_path), None)

        try:
            logger.info(
                f"File deleted from watched path, removing from index: {file_path} "
                f"from collection {self.collection_name}"
            )
            self.on_file_deleted(file_path, self.collection_name)
        except Exception as e:
            logger.error(
                f"Error handling file deletion for {file_path}: {e}",
                exc_info=True,
            )

    def _handle_modified(self, file_path: Path) -> None:
        """Handle file modification event."""
        if self.on_file_modified is None:
            return

        if not self._is_supported_file(file_path):
            return

        if not self._should_process_event(file_path):
            return

        if file_path.is_dir():
            return

        try:
            logger.info(f"File modified in watched path: {file_path}")
            self.on_file_modified(file_path, self.collection_name)
        except Exception as e:
            logger.error(
                f"Error handling file modification for {file_path}: {e}",
                exc_info=True,
            )

    def _handle_moved(self, src_path: Path, dest_path: Path) -> None:
        """Handle file move event (treat as delete + create)."""
        # Handle as delete from source
        self._handle_deleted(src_path)

        # Handle as create at destination (if still in watched path)
        try:
            if dest_path.is_relative_to(self.watch_path):
                self._handle_created(dest_path)
        except ValueError:
            # dest_path is not relative to watch_path
            pass

    def start(self, blocking: bool = False) -> None:
        """Start the watcher.

        Args:
            blocking: If True, blocks until stop() is called or KeyboardInterrupt.
                      If False, starts in background and returns immediately.
        """
        if self._running:
            logger.warning(f"Watcher already running for {self.watch_path}")
            return

        if not self.watch_path.exists():
            logger.warning(
                f"Watch path does not exist, creating: {self.watch_path}"
            )
            try:
                self.watch_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create watch path {self.watch_path}: {e}")
                return

        watcher = self

        class Handler(FileSystemEventHandler):
            def on_created(self, event: FileSystemEvent) -> None:
                if isinstance(event, FileCreatedEvent) and not event.is_directory:
                    # Run in thread to avoid blocking the observer
                    threading.Thread(
                        target=watcher._handle_created,
                        args=(Path(event.src_path),),
                        daemon=True,
                    ).start()

            def on_deleted(self, event: FileSystemEvent) -> None:
                if isinstance(event, FileDeletedEvent) and not event.is_directory:
                    threading.Thread(
                        target=watcher._handle_deleted,
                        args=(Path(event.src_path),),
                        daemon=True,
                    ).start()

            def on_modified(self, event: FileSystemEvent) -> None:
                if isinstance(event, FileModifiedEvent) and not event.is_directory:
                    threading.Thread(
                        target=watcher._handle_modified,
                        args=(Path(event.src_path),),
                        daemon=True,
                    ).start()

            def on_moved(self, event: FileSystemEvent) -> None:
                if isinstance(event, FileMovedEvent) and not event.is_directory:
                    threading.Thread(
                        target=watcher._handle_moved,
                        args=(Path(event.src_path), Path(event.dest_path)),
                        daemon=True,
                    ).start()

        event_handler = Handler()
        self._observer = Observer()
        self._observer.schedule(event_handler, str(self.watch_path), recursive=True)
        self._observer.start()
        self._running = True

        logger.info(
            f"Started watching temporary path: {self.watch_path} "
            f"for collection: {self.collection_name}"
        )

        if blocking:
            while self._observer.is_alive():
                try:
                    self._observer.join(1)
                except KeyboardInterrupt:
                    self.stop()
                    break

    def stop(self) -> None:
        """Stop the watcher."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._running = False
            logger.info(f"Stopped watching: {self.watch_path}")

    @property
    def is_running(self) -> bool:
        """Check if the watcher is running."""
        return self._running
