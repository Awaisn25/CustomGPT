"""FastAPI app creation, logger configuration and main API routes."""

import atexit
import logging

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from injector import Injector
from llama_index.core.callbacks import CallbackManager
from llama_index.core.settings import Settings as LlamaIndexSettings

from private_gpt.server.chat.chat_router import chat_router
from private_gpt.server.chunks.chunks_router import chunks_router
from private_gpt.server.completions.completions_router import completions_router
from private_gpt.server.embeddings.embeddings_router import embeddings_router
from private_gpt.server.health.health_router import health_router
from private_gpt.server.ingest.ingest_router import ingest_router
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.watched_path_manager import WatchedPathManager
from private_gpt.server.recipes.summarize.summarize_router import summarize_router
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


def create_app(root_injector: Injector) -> FastAPI:

    # Start the API
    async def bind_injector_to_request(request: Request) -> None:
        request.state.injector = root_injector

    app = FastAPI(dependencies=[Depends(bind_injector_to_request)])

    app.include_router(completions_router)
    app.include_router(chat_router)
    app.include_router(chunks_router)
    app.include_router(ingest_router)
    app.include_router(summarize_router)
    app.include_router(embeddings_router)
    app.include_router(health_router)

    LlamaIndexSettings.callback_manager = CallbackManager([])

    settings = root_injector.get(Settings)
    if settings.server.cors.enabled:
        logger.debug("Setting up CORS middleware")
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=settings.server.cors.allow_credentials,
            allow_origins=settings.server.cors.allow_origins,
            allow_origin_regex=settings.server.cors.allow_origin_regex,
            allow_methods=settings.server.cors.allow_methods,
            allow_headers=settings.server.cors.allow_headers,
        )

    if settings.ui.enabled:
        logger.debug("Importing the UI module")
        try:
            from private_gpt.ui.ui import PrivateGptUi
        except ImportError as e:
            raise ImportError(
                "UI dependencies not found, install with `poetry install --extras ui`"
            ) from e

        ui = root_injector.get(PrivateGptUi)
        ui.mount_in_app(app, settings.ui.path)

    # Initialize file watchers for automatic ingestion/deletion
    _initialize_file_watchers(root_injector, settings)

    return app


def _initialize_file_watchers(root_injector: Injector, settings: Settings) -> None:
    """Initialize file watchers for automatic ingestion and deletion.

    This starts watchers for configured paths if enabled in settings.
    Watchers are automatically stopped when the application shuts down.
    """
    if not settings.data.paths.watch_enabled:
        logger.debug("File watching is disabled in settings")
        return

    try:
        # Get the services
        watched_path_manager = root_injector.get(WatchedPathManager)
        ingest_service = root_injector.get(IngestService)

        # Set up the ingest service reference
        watched_path_manager.set_ingest_service(ingest_service)

        # Start configured watchers
        watched_path_manager.start_configured_watchers()

        # Sync existing files if enabled
        if settings.data.paths.sync_on_startup:
            logger.info("Syncing existing files in watched paths...")
            watched_path_manager.sync_existing_files(
                watch_path=settings.data.paths.temporary_path,
                collection_name=settings.data.paths.temporary_collection_name,
            )

        # Register cleanup on application shutdown
        def cleanup_watchers() -> None:
            logger.info("Shutting down file watchers...")
            watched_path_manager.stop_all_watchers()

        atexit.register(cleanup_watchers)

        logger.info("File watchers initialized successfully")
    except Exception as e:
        logger.error(
            f"Failed to initialize file watchers: {e}. "
            "File watching will be disabled.",
            exc_info=True,
        )
