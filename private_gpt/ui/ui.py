"""This file should be imported if and only if you want to run the UI locally."""

import base64
import logging
import time
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.core.types import TokenGen
from pydantic import BaseModel

from private_gpt.components.ingest.file_tracker import get_file_tracker
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.watched_path_manager import WatchedPathManager
from private_gpt.server.recipes.summarize.summarize_service import SummarizeService
from private_gpt.settings.settings import Settings, settings
from private_gpt.ui.images import logo_svg
from private_gpt.utils.model_config import ModelConfig, get_available_models

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/avatar-bot.ico"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"

UI_TAB_TITLE = "My Private GPT"

SOURCES_SEPARATOR = "<hr>Sources: \n"


class Modes(str, Enum):
    RAG_MODE = "RAG"
    SEARCH_MODE = "Search"
    BASIC_CHAT_MODE = "Basic"
    SUMMARIZE_MODE = "Summarize"


MODES: list[Modes] = [
    Modes.RAG_MODE,
    Modes.SEARCH_MODE,
    Modes.BASIC_CHAT_MODE,
    Modes.SUMMARIZE_MODE,
]


class Source(BaseModel):
    file: str
    page: str
    text: str
    doc_id: str

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[Chunk]) -> list["Source"]:
        curated_sources = []

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata

            file_name = doc_metadata.get("file_name", "-") if doc_metadata else "-"
            page_label = doc_metadata.get("page_label", "-") if doc_metadata else "-"
            doc_id = chunk.document.doc_id

            source = Source(
                file=file_name, page=page_label, text=chunk.text, doc_id=doc_id
            )
            curated_sources.append(source)
            curated_sources = list(
                dict.fromkeys(curated_sources).keys()
            )  # Unique sources only

        return curated_sources


@singleton
class PrivateGptUi:
    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
        summarizeService: SummarizeService,
        llm_component: LLMComponent,
        settings: Settings,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service
        self._summarize_service = summarizeService
        self._llm_component = llm_component
        self._settings = settings

        # Cache the UI blocks
        self._ui_block = None

        self._selected_filename = None
        self._selected_collection: str | None = None  # Track selected collection

        # Initialize system prompt based on default mode
        default_mode_map = {mode.value: mode for mode in Modes}
        self._default_mode = default_mode_map.get(
            settings.ui.default_mode, Modes.RAG_MODE
        )
        self._system_prompt = self._get_default_system_prompt(self._default_mode)

        # Get available models and current model
        self._available_models = get_available_models()
        self._current_model = self._get_current_model_display_name()

        # Collection names from settings
        self._default_collection = settings.vectorstore.default_collection_name
        self._persistent_collection = settings.data.paths.persistent_collection_name
        self._temporary_collection = settings.data.paths.temporary_collection_name

    def _get_collection_choices(self) -> list[str]:
        """Get list of available collections for dropdown."""
        return [
            f"All Collections",
            f"Default ({self._default_collection})",
            f"Persistent ({self._persistent_collection})",
            f"Temporary ({self._temporary_collection})",
        ]

    def _get_collection_name_from_choice(self, choice: str) -> str | None:
        """Extract collection name from dropdown choice."""
        if choice.startswith("All"):
            return None
        elif choice.startswith("Default"):
            return self._default_collection
        elif choice.startswith("Persistent"):
            return self._persistent_collection
        elif choice.startswith("Temporary"):
            return self._temporary_collection
        return None

    def _chat(
        self, message: str, history: list[list[str]], mode: Modes, *_: Any
    ) -> Any:
        def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
            full_response: str = ""
            stream = completion_gen.response
            for delta in stream:
                if isinstance(delta, str):
                    full_response += str(delta)
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                yield full_response
                time.sleep(0.02)

            if completion_gen.sources:
                full_response += SOURCES_SEPARATOR
                cur_sources = Source.curate_sources(completion_gen.sources)
                sources_text = "\n\n\n"
                used_files = set()
                for index, source in enumerate(cur_sources, start=1):
                    if f"{source.file}-{source.page}" not in used_files:
                        # Create clickable link to the document
                        file_url = f"/v1/ingest/{source.doc_id}/file?collection_name={self._selected_collection}"
                        # Add page anchor for PDFs
                        if source.page and source.page != "-":
                            file_url += f"#page={source.page}"

                        # Format as markdown link
                        sources_text = (
                            sources_text
                            + f"{index}. [{source.file}]({file_url}) (page {source.page}) \n\n"
                        )
                        used_files.add(f"{source.file}-{source.page}")
                sources_text += "<hr>\n\n"
                full_response += sources_text
            yield full_response

        def yield_tokens(token_gen: TokenGen) -> Iterable[str]:
            full_response: str = ""
            for token in token_gen:
                full_response += str(token)
                yield full_response

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = []

            for interaction in history:
                history_messages.append(
                    ChatMessage(content=interaction[0], role=MessageRole.USER)
                )
                if len(interaction) > 1 and interaction[1] is not None:
                    history_messages.append(
                        ChatMessage(
                            # Remove from history content the Sources information
                            content=interaction[1].split(SOURCES_SEPARATOR)[0],
                            role=MessageRole.ASSISTANT,
                        )
                    )

            # max 20 messages to try to avoid context overflow
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]
        # If a system prompt is set, add it as a system message
        if self._system_prompt:
            all_messages.insert(
                0,
                ChatMessage(
                    content=self._system_prompt,
                    role=MessageRole.SYSTEM,
                ),
            )
        match mode:
            case Modes.RAG_MODE:
                # Use only the selected file for the query
                context_filter = None
                if self._selected_filename is not None:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested(
                        collection_name=self._selected_collection
                    ):
                        if (
                            ingested_document.doc_metadata
                            and ingested_document.doc_metadata.get("file_name")
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                query_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                    context_filter=context_filter,
                    collection_name=self._selected_collection,
                )
                yield from yield_deltas(query_stream)
            case Modes.BASIC_CHAT_MODE:
                llm_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=False,
                )
                yield from yield_deltas(llm_stream)

            case Modes.SEARCH_MODE:
                context_filter = None
                if self._selected_filename is not None:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested(
                        collection_name=self._selected_collection
                    ):
                        if (
                            ingested_document.doc_metadata
                            and ingested_document.doc_metadata.get("file_name")
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                response = self._chunks_service.retrieve_relevant(
                    text=message,
                    limit=4,
                    prev_next_chunks=0,
                    context_filter=context_filter,
                    collection_name=self._selected_collection,
                )

                sources = Source.curate_sources(response)

                # Build search results with clickable links
                search_results = []
                for index, source in enumerate(sources, start=1):
                    # Create clickable link to the document
                    file_url = f"/v1/ingest/{source.doc_id}/file?collection_name={self._selected_collection}"
                    # Add page anchor for PDFs
                    if source.page and source.page != "-":
                        file_url += f"#page={source.page}"

                    search_results.append(
                        f"{index}. **[{source.file}]({file_url}) "
                        f"(page {source.page})**\n "
                        f"{source.text}"
                    )

                yield "\n\n\n".join(search_results)
            case Modes.SUMMARIZE_MODE:
                # Summarize the given message, optionally using selected files
                context_filter = None
                if self._selected_filename:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested(
                        collection_name=self._selected_collection
                    ):
                        if (
                            ingested_document.doc_metadata
                            and ingested_document.doc_metadata.get("file_name")
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                summary_stream = self._summarize_service.stream_summarize(
                    use_context=True,
                    context_filter=context_filter,
                    instructions=message,
                )
                yield from yield_tokens(summary_stream)

    # On initialization and on mode change, this function set the system prompt
    # to the default prompt based on the mode (and user settings).
    @staticmethod
    def _get_default_system_prompt(mode: Modes) -> str:
        p = ""
        match mode:
            # For query chat mode, obtain default system prompt from settings
            case Modes.RAG_MODE:
                p = settings().ui.default_query_system_prompt
            # For chat mode, obtain default system prompt from settings
            case Modes.BASIC_CHAT_MODE:
                p = settings().ui.default_chat_system_prompt
            # For summarization mode, obtain default system prompt from settings
            case Modes.SUMMARIZE_MODE:
                p = settings().ui.default_summarization_system_prompt
            # For any other mode, clear the system prompt
            case _:
                p = ""
        return p

    @staticmethod
    def _get_default_mode_explanation(mode: Modes) -> str:
        match mode:
            case Modes.RAG_MODE:
                return "Get contextualized answers from selected files."
            case Modes.SEARCH_MODE:
                return "Find relevant chunks of text in selected files."
            case Modes.BASIC_CHAT_MODE:
                return "Chat with the LLM using its training data. Files are ignored."
            case Modes.SUMMARIZE_MODE:
                return "Generate a summary of the selected files. Prompt to customize the result."
            case _:
                return ""

    def _set_system_prompt(self, system_prompt_input: str) -> None:
        logger.info(f"Setting system prompt to: {system_prompt_input}")
        self._system_prompt = system_prompt_input

    def _set_explanatation_mode(self, explanation_mode: str) -> None:
        self._explanation_mode = explanation_mode

    def _set_current_mode(self, mode: Modes) -> Any:
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
        self._set_explanatation_mode(self._get_default_mode_explanation(mode))
        interactive = self._system_prompt is not None
        return [
            gr.update(placeholder=self._system_prompt, interactive=interactive),
            gr.update(value=self._explanation_mode),
        ]

    def _list_ingested_files(self) -> list[list[str]]:
        """List ingested files, optionally filtered by selected collection."""
        files: dict[str, str] = {}  # file_name -> collection_name
        for ingested_document in self._ingest_service.list_ingested(
            collection_name=self._selected_collection
        ):
            if ingested_document.doc_metadata is None:
                # Skipping documents without metadata
                continue
            file_name = ingested_document.doc_metadata.get(
                "file_name", "[FILE NAME MISSING]"
            )
            collection = ingested_document.doc_metadata.get(
                "collection_name", "default"
            )
            # Track file with collection info
            if file_name not in files:
                files[file_name] = collection
        # Return as list with file name and collection
        return [[name, coll] for name, coll in files.items()]

    def _on_collection_change(self, collection_choice: str) -> Any:
        """Handle collection dropdown change."""
        self._selected_collection = self._get_collection_name_from_choice(collection_choice)
        logger.info(f"Selected collection: {self._selected_collection}")
        # Refresh the file list
        return gr.List(self._list_ingested_files())

    def _upload_file(self, files: list[str]) -> Any:
        """Upload files to the selected collection."""
        logger.debug("Loading count=%s files into collection=%s", len(files), self._selected_collection)
        paths = [Path(file) for file in files]

        # Determine collection to use (default if not selected)
        collection = self._selected_collection or self._default_collection

        # remove all existing Documents with name identical to a new file upload:
        file_names = [path.name for path in paths]
        doc_ids_to_delete = []
        for ingested_document in self._ingest_service.list_ingested(
            collection_name=collection
        ):
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata.get("file_name") in file_names
            ):
                doc_ids_to_delete.append(ingested_document.doc_id)
        if len(doc_ids_to_delete) > 0:
            logger.info(
                "Uploading file(s) which were already ingested: %s document(s) will be replaced.",
                len(doc_ids_to_delete),
            )
            for doc_id in doc_ids_to_delete:
                self._ingest_service.delete(doc_id, collection_name=collection)

        self._ingest_service.bulk_ingest(
            [(str(path.name), path) for path in paths],
            collection_name=collection,
        )
        # Return updated file list
        return gr.List(self._list_ingested_files())

    def _delete_all_files(self) -> Any:
        """Delete all files from the selected collection (or all collections)."""
        ingested_files = self._ingest_service.list_ingested(
            collection_name=self._selected_collection
        )
        logger.debug("Deleting count=%s files from collection=%s", len(ingested_files), self._selected_collection)
        for ingested_document in ingested_files:
            self._ingest_service.delete(
                ingested_document.doc_id,
                collection_name=self._selected_collection,
            )
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _delete_selected_file(self) -> Any:
        """Delete selected file from the selected collection."""
        logger.debug("Deleting selected %s from collection=%s", self._selected_filename, self._selected_collection)
        # Note: keep looping for pdf's (each page became a Document)
        for ingested_document in self._ingest_service.list_ingested(
            collection_name=self._selected_collection
        ):
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata.get("file_name")
                == self._selected_filename
            ):
                self._ingest_service.delete(
                    ingested_document.doc_id,
                    collection_name=self._selected_collection,
                )
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _deselect_selected_file(self) -> Any:
        self._selected_filename = None
        return [
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _selected_a_file(self, select_data: gr.SelectData) -> Any:
        self._selected_filename = select_data.value
        return [
            gr.components.Button(interactive=True),
            gr.components.Button(interactive=True),
            gr.components.Textbox(self._selected_filename),
        ]

    def _get_current_model_display_name(self) -> str:
        """Get the display name of the currently active Ollama model."""
        config_settings = settings()
        if config_settings is None:
            return "Unknown"

        model_name = config_settings.ollama.llm_model
        return f"Ollama: {model_name}"

    def _get_model_options(self) -> list[str]:
        """Get list of model display names for the dropdown."""
        return [model.display_name for model in self._available_models]

    def _swap_model(self, model_display_name: str) -> tuple[Any, Any]:
        """Swap to a different model."""
        try:
            # Find the model config matching the display name
            model_config = None
            for model in self._available_models:
                if model.display_name == model_display_name:
                    model_config = model
                    break

            if model_config is None:
                logger.error(f"Model not found: {model_display_name}")
                return gr.update(), gr.update()

            # Swap the LLM
            self._llm_component.swap_llm(model_config)
            self._current_model = model_display_name

            logger.info(f"Successfully swapped to model: {model_display_name}")
            return (
                gr.update(value=model_display_name),
                gr.update(label=f"Model: {model_display_name}"),
            )
        except Exception as e:
            logger.error(f"Failed to swap model: {e}")
            # Return current model to keep dropdown in sync
            return (
                gr.update(value=self._current_model),
                gr.update(),
            )

    def _get_watcher_status(self) -> str:
        """Get the current status of file watchers."""
        try:
            watched_path_manager = global_injector.get(WatchedPathManager)
            active_watchers = watched_path_manager.get_active_watchers()
            
            if not active_watchers:
                return "🔴 No active watchers"
            
            status_lines = ["🟢 Active watchers:"]
            for path, collection in active_watchers.items():
                status_lines.append(f"  • {path} → {collection}")
            return "\n".join(status_lines)
        except Exception as e:
            logger.debug(f"Error getting watcher status: {e}")
            return "⚪ Watcher status unavailable"

    def _get_tracked_files_count(self) -> str:
        """Get count of tracked files per collection."""
        try:
            file_tracker = get_file_tracker()
            counts = []
            for coll_name in [self._default_collection, self._persistent_collection, self._temporary_collection]:
                tracked = file_tracker.get_all_tracked_files(coll_name)
                if tracked:
                    counts.append(f"{coll_name}: {len(tracked)} files")
            return ", ".join(counts) if counts else "No tracked files"
        except Exception as e:
            logger.debug(f"Error getting tracked files count: {e}")
            return "Tracking info unavailable"

    def _refresh_watcher_status(self) -> str:
        """Refresh and return the watcher status."""
        return self._get_watcher_status()

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(primary_hue=slate),
            css=".logo { "
            "display:flex;"
            "background-color: #C7BAFF;"
            "height: 80px;"
            "border-radius: 8px;"
            "align-content: center;"
            "justify-content: center;"
            "align-items: center;"
            "}"
            ".logo img { height: 25% }"
            ".contain { display: flex !important; flex-direction: column !important; }"
            "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
            "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
            "#col { height: calc(100vh - 112px - 16px) !important; }"
            "hr { margin-top: 1em; margin-bottom: 1em; border: 0; border-top: 1px solid #FFF; }"
            ".avatar-image { background-color: antiquewhite; border-radius: 2px; }"
            ".footer { text-align: center; margin-top: 20px; font-size: 14px; display: flex; align-items: center; justify-content: center; }"
            ".footer-zylon-link { display:flex; margin-left: 5px; text-decoration: auto; color: var(--body-text-color); }"
            ".footer-zylon-link:hover { color: #C7BAFF; }"
            ".footer-zylon-ico { height: 20px; margin-left: 5px; background-color: antiquewhite; border-radius: 2px; }"
            ".watcher-status { font-size: 12px; padding: 8px; border-radius: 4px; background: var(--background-fill-secondary); }"
            ".collection-info { font-size: 11px; color: var(--body-text-color-subdued); }",
        ) as blocks:
            # with gr.Row():
                # gr.HTML(f"<div class='logo'/><img src={logo_svg} alt=PrivateGPT></div")

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    # Model selection dropdown
                    model_options = self._get_model_options()
                    current_model_value = (
                        self._current_model
                        if self._current_model in model_options
                        else (model_options[0] if model_options else "No models available")
                    )
                    model_dropdown = gr.Dropdown(
                        choices=model_options,
                        label="Model",
                        value=current_model_value,
                        interactive=True,
                    )

                    default_mode = self._default_mode
                    mode = gr.Radio(
                        [mode.value for mode in MODES],
                        label="Mode",
                        value=default_mode,
                    )
                    explanation_mode = gr.Textbox(
                        placeholder=self._get_default_mode_explanation(default_mode),
                        show_label=False,
                        max_lines=3,
                        interactive=False,
                    )

                    # Collection selector dropdown
                    collection_dropdown = gr.Dropdown(
                        choices=self._get_collection_choices(),
                        label="📁 Collection",
                        value="All Collections",
                        interactive=True,
                        info="Select which collection to query/upload to",
                    )

                    upload_button = gr.components.UploadButton(
                        "Upload File(s)",
                        type="filepath",
                        file_count="multiple",
                        size="sm",
                    )
                    ingested_dataset = gr.List(
                        self._list_ingested_files,
                        headers=["File name", "Collection"],
                        label="Ingested Files",
                        col_count=2,
                        # height=235,
                        interactive=False,
                        render=False,  # Rendered under the button
                    )
                    # Collection change handler
                    collection_dropdown.change(
                        self._on_collection_change,
                        inputs=collection_dropdown,
                        outputs=ingested_dataset,
                    )
                    upload_button.upload(
                        self._upload_file,
                        inputs=upload_button,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.change(
                        self._list_ingested_files,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.render()
                    deselect_file_button = gr.components.Button(
                        "De-select selected file", size="sm", interactive=False
                    )
                    selected_text = gr.components.Textbox(
                        "All files", label="Selected for Query or Deletion", max_lines=1
                    )
                    delete_file_button = gr.components.Button(
                        "🗑️ Delete selected file",
                        size="sm",
                        visible=self._settings.ui.delete_file_button_enabled,
                        interactive=False,
                    )
                    delete_files_button = gr.components.Button(
                        "⚠️ Delete ALL files",
                        size="sm",
                        visible=self._settings.ui.delete_all_files_button_enabled,
                    )
                    deselect_file_button.click(
                        self._deselect_selected_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    ingested_dataset.select(
                        fn=self._selected_a_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    delete_file_button.click(
                        self._delete_selected_file,
                        outputs=[
                            ingested_dataset,
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    delete_files_button.click(
                        self._delete_all_files,
                        outputs=[
                            ingested_dataset,
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )

                    # File Watcher Status Panel
                    with gr.Accordion("📂 File Watcher Status", open=False):
                        watcher_status = gr.Textbox(
                            value=self._get_watcher_status,
                            label="Watcher Status",
                            interactive=False,
                            lines=3,
                            elem_classes=["watcher-status"],
                        )
                        refresh_watcher_btn = gr.Button(
                            "🔄 Refresh Status", size="sm"
                        )
                        refresh_watcher_btn.click(
                            self._refresh_watcher_status,
                            outputs=watcher_status,
                        )
                        # Path info
                        gr.Markdown(
                            f"""
                            **Configured Paths:**
                            - Persistent: `{self._settings.data.paths.persistent_path}`
                            - Temporary: `{self._settings.data.paths.temporary_path}`
                            
                            **Watch Settings:**
                            - Watch Enabled: `{self._settings.data.paths.watch_enabled}`
                            - Watch Modifications: `{self._settings.data.paths.watch_modifications}`
                            """,
                            elem_classes=["collection-info"],
                        )

                    system_prompt_input = gr.Textbox(
                        placeholder=self._system_prompt,
                        label="System Prompt",
                        lines=2,
                        interactive=True,
                        render=False,
                    )
                    # When mode changes, set default system prompt, and other stuffs
                    mode.change(
                        self._set_current_mode,
                        inputs=mode,
                        outputs=[system_prompt_input, explanation_mode],
                    )
                    # On blur, set system prompt to use in queries
                    system_prompt_input.blur(
                        self._set_system_prompt,
                        inputs=system_prompt_input,
                    )

                    def get_model_label() -> str | None:
                        """Get model label from llm mode setting YAML.

                        Raises:
                            ValueError: If an invalid 'llm_mode' is encountered.

                        Returns:
                            str: The corresponding model label.
                        """
                        # Get model label from llm mode setting YAML
                        # Labels: local, openai, openailike, sagemaker, mock, ollama
                        config_settings = settings()
                        if config_settings is None:
                            raise ValueError("Settings are not configured.")

                        # Get llm_mode from settings
                        llm_mode = config_settings.llm.mode

                        # Mapping of 'llm_mode' to corresponding model labels
                        model_mapping = {
                            "llamacpp": config_settings.llamacpp.llm_hf_model_file,
                            "openai": config_settings.openai.model,
                            "openailike": config_settings.openai.model,
                            "azopenai": config_settings.azopenai.llm_model,
                            "sagemaker": config_settings.sagemaker.llm_endpoint_name,
                            "mock": llm_mode,
                            "ollama": config_settings.ollama.llm_model,
                            "gemini": config_settings.gemini.model,
                        }

                        if llm_mode not in model_mapping:
                            print(f"Invalid 'llm mode': {llm_mode}")
                            return None

                        return model_mapping[llm_mode]

                with gr.Column(scale=7, elem_id="col"):
                    # Determine the model label based on the value of PGPT_PROFILES
                    model_label = get_model_label()
                    if model_label is not None:
                        label_text = (
                            f"LLM: {settings().llm.mode} | Model: {model_label}"
                        )
                    else:
                        label_text = f"LLM: {settings().llm.mode}"

                    chatbot_component = gr.Chatbot(
                        label=label_text,
                        show_copy_button=True,
                        elem_id="chatbot",
                        render=False,
                        avatar_images=(
                            None,
                            AVATAR_BOT,
                        ),
                    )

                    _ = gr.ChatInterface(
                        self._chat,
                        chatbot=chatbot_component,
                        additional_inputs=[mode, upload_button, system_prompt_input],
                    )

                    # Set up model swap handler after both components are defined
                    model_dropdown.change(
                        self._swap_model,
                        inputs=model_dropdown,
                        outputs=[model_dropdown, chatbot_component],
                    )

            with gr.Row():
                avatar_byte = AVATAR_BOT.read_bytes()
                f_base64 = f"data:image/png;base64,{base64.b64encode(avatar_byte).decode('utf-8')}"
                # gr.HTML(
                #     f"<div class='footer'><a class='footer-zylon-link' href='https://zylon.ai/'>Maintained by Zylon <img class='footer-zylon-ico' src='{f_base64}' alt=Zylon></a></div>"
                # )

        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI, at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path, favicon_path=AVATAR_BOT)


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)
