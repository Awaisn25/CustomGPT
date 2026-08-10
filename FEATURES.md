# CustomGPT — Feature Inventory

A reference for porting this project's capabilities into another codebase. Split into
**Part A: Custom features** (built on top of PrivateGPT for this project) and
**Part B: Stock PrivateGPT foundation** (inherited base capabilities).

> Notes carried from planning:
> - The `is_temporary` document flag is **vestigial** — threaded through ingestion metadata
>   but not meaningfully consumed. Skip it when porting.
> - The dual-path defaults (`D:/`, `E:/`) are Windows drive letters and will be fixed to
>   Linux directories separately — not covered here.
> - Several features are **Ollama-specific** (vision OCR, model hot-swap, summarize LLM
>   serialization). A provider check (`if settings.llm.mode == "ollama"`) should gate these
>   but does **not** exist yet — to be added later.

---

# Part A — Custom Features

## A1. Multi-Collection Vector Store
**What:** One running instance manages multiple named vector collections instead of a single
global index. Every ingest/list/delete/chat operation is scoped to a `collection_name`.

**How:**
- `components/vector_store/vector_store_component.py` — `VectorStoreComponent` holds
  `_vector_stores: dict[str, BasePydanticVectorStore]` and a factory.
  `get_vector_store(collection_name)` lazily creates + caches one store per collection.
  Backends: chroma, qdrant, postgres, milvus, clickhouse. Postgres uses
  `embeddings_{collection}` table names; chroma/qdrant/milvus use the collection name natively.
- `server/ingest/ingest_service.py` — `IngestService` mirrors this with `_storage_contexts`
  and `_ingest_components` dicts keyed by collection.
- **Key subtlety:** the *docstore* (node store) is **shared** across all collections; only the
  *vector store* is per-collection. `list_ingested(collection_name)` pulls all ref-docs from
  the shared docstore and filters by a `collection_name` value stored in each document's metadata.

## A2. Dual-Path System + Path→Collection Auto-Mapping
**What:** Two configured filesystem roots — "persistent" and "temporary" — each auto-map to
their own collection. Drop a file under a path and it lands in the correct collection.

**How:**
- `settings/settings.py` — `DataPathsSettings`: `persistent_path`, `temporary_path`,
  `persistent_collection_name`, `temporary_collection_name`, `watch_enabled`,
  `watch_modifications`, `create_paths_if_missing`, `sync_on_startup`.
- `utils/collection_mapper.py:get_collection_for_path()` resolves an absolute path and checks
  `startswith` against each configured root (case-insensitive on Windows) to pick the
  collection; falls back to default. Used during ingest when no explicit collection is passed.

## A3. Automatic File-Watching Ingestion/Deletion
**What:** Background watchers monitor the configured paths. New file → auto-ingest; deleted
file → auto-remove from index; (optional) modified file → re-ingest. Survives restarts and
re-syncs on startup.

**How:**
- `server/ingest/ingest_watcher.py:TemporaryPathWatcher` — watchdog `Observer`; handles
  create/delete/modify/move; 1s debounce; 0.5s settle-wait for writes to finish; each event on
  a daemon thread; optional extension filtering.
- `server/ingest/watched_path_manager.py:WatchedPathManager` (singleton) — starts/stops
  watchers per path, wires callbacks to `IngestService`, dedups via the FileTracker,
  `sync_existing_files()` (rglob + ingest untracked). Modify = delete-then-create.
- `launcher.py:_initialize_file_watchers()` wires the ingest service in (avoids circular DI),
  starts configured watchers, runs startup sync if `sync_on_startup`, registers `atexit` cleanup.

## A4. Persistent File→Doc-ID Tracker
**What:** A disk-backed map of file path → list of doc IDs, per collection, so deletions can
find the right docs even after a restart (watchdog delete events only give a path).

**How:** `components/ingest/file_tracker.py:FileTracker` — thread-safe (RLock), in-memory cache
+ one JSON file per collection under `local_data_path/file_tracker/` (sanitized filename).
API: `track_file`, `get_doc_ids`, `untrack_file`, `is_file_tracked`, `get_all_tracked_files`,
`clear_collection`. Global singleton via `get_file_tracker()`. Deletion falls back to scanning
the docstore by `file_name` when a file isn't tracked (e.g. UI uploads).

## A5. Auto-Convert Office Docs to PDF Before Ingestion
**What:** `.docx/.doc/.pptx/.ppt/.xlsx/.odt/.epub/.rtf` etc. are converted to PDF at ingest
time (so the original source is browser-viewable), with dual-ingestion prevention.

**How:** `components/ingest/pdf_converter.py` — `needs_conversion()` checks a suffix set;
`convert_to_pdf()` shells out to LibreOffice (`soffice --headless --convert-to pdf`), reuses an
existing sibling PDF, 120s timeout, gracefully degrades if soffice missing.
`IngestService.ingest_file` swaps to the converted PDF path/name. The watcher pre-tracks the
produced `.pdf` sibling so its own create-event doesn't re-ingest it.

## A6. Scanned-PDF + Image OCR via Ollama Vision Model
**What:** PDFs are read page-by-page; text pages use embedded text (fast), scanned/image-only
pages are rendered and OCR'd through a local vision model. Standalone images
(`.jpg/.png/.jpeg`) use the same vision model.

**How:** `components/readers/custom_readers.py`:
- `CustomImageReader` — sends image to Ollama (`ollama.chat`, default `qwen2.5vl:7b`,
  `think=False`) with a prompt returning raw text or a brief description.
- `ScannedPDFReader` — PyMuPDF (`fitz`); per page tries `get_text()`, else renders at
  configurable DPI (150) → temp PNG → `CustomImageReader`. Preserves PDF page labels into
  `page_label` metadata for source attribution.
- Wired in `ingest_helper.py` `FILE_READER_CLS`: `.pdf → ScannedPDFReader`, images →
  `CustomImageReader`.
- **Ollama-specific** — needs a provider gate before porting to non-Ollama setups.

## A7. Enhanced Ingestion Metadata + Sanitization
**What:** Every document gets `file_name`, `source_path`, `collection_name`, `is_temporary`
(vestigial), `doc_id`, `page_label` metadata, with control over what the embedder vs the LLM
sees, plus encoding sanitization.

**How:** `ingest_helper.py:transform_file_into_documents` — adds metadata, strips UTF-8
surrogates and NUL bytes (needed for Postgres), sets `excluded_embed_metadata_keys` /
`excluded_llm_metadata_keys` so bookkeeping fields don't pollute retrieval or context.

## A8. Batch Summarize (parallel, map-reduce)
**What:** A UI mode that summarizes many selected documents at once, in parallel, streaming
each result as it completes. Large single documents use chunked map-reduce so they don't blow
the context window.

**How:** `server/recipes/summarize/summarize_service.py`:
- `summarize_batch()` — `ThreadPoolExecutor` (workers from `settings.summarize.max_workers`),
  one `ContextFilter(docs_ids=...)` per file, yields `SummaryResult(filename, doc_id, summary)`
  via `as_completed`.
- Map-reduce: when nodes exceed `max_nodes_per_chunk` (default 150), splits into chunks,
  `TREE_SUMMARIZE` each, then a final combine pass. Streaming only on the final reduction.
- **Concurrency guard:** module-level `_llm_lock` serializes all LLM calls because a local
  Ollama serves one request at a time (parallel calls → HTTP 500). Batching parallelizes
  orchestration but LLM calls are serialized.
- `summarize.request_timeout` can override the provider timeout for summarization only
  (rebuilds the Ollama/OpenAI LLM with a new timeout).
- Settings: `SummarizeSettings{use_async, max_workers, max_nodes_per_chunk, request_timeout}`.

## A9. Ollama Model Hot-Swapping (UI dropdown)
**What:** A dropdown lets you switch the active Ollama model live without restarting.

**How:** `utils/model_config.py:get_available_models()` reads `ui.dropdown_models`.
`LLMComponent.swap_llm()` / `_create_ollama_llm()` builds a fresh `Ollama` instance (auto-adds
`:latest`, carries over sampling kwargs, optional autopull). UI `_swap_model()` wires the
dropdown `.change`. **Ollama-specific.**

## A10. Source File Serving Endpoint
**What:** `GET /v1/ingest/{doc_id}/file?collection_name=...` returns the original source file
inline (viewable in-browser), with PDF `#page=N` deep-linking. Chat/search sources render as
clickable links.

**How:** `ingest_router.py:get_document_file` + `IngestService.get_document_file_path` — looks
up `source_path` from doc metadata, **security-validates** the path is inside the allowed
persistent/temporary roots (403 otherwise), checks existence, sets Content-Type by extension,
`Content-Disposition: inline`. UI builds these links in RAG and Search results.

## A11. UI Enhancements
`ui/ui.py`:
- **Collection filter dropdown** — "All Collections / Default / Persistent / Temporary";
  `_on_collection_change` sets `_selected_collection`, filters the file list, scopes all ops.
- **File search bar** — `_list_ingested_files(query)` filters the ingested list;
  `_refresh_file_list` preserves the active query.
- **Batch Summarize mode** — `CheckboxGroup` of documents + **Select All / Deselect All**
  buttons (shown only in batch mode via `_set_current_mode` visibility toggles).
- **File Watcher Status panel** — `_get_watcher_status` (active watchers → collections) and
  `_get_tracked_files_count`.
- Modes: RAG / Search / Basic / Summarize / Batch Summarize; `UISettings.default_mode` extended.

## A12. Multi-Collection-Aware Delete Robustness
**What:** Deletion tolerates the shared-index-store desync that occurs with multiple
collections after a restart.

**How:** `ingest_component.py:BaseIngestComponentWithIndex.delete` — on `KeyError` (doc_id
missing from `index_struct.nodes_dict`), falls back to deleting directly from the vector store
+ docstore instead of crashing.

---

# Part B — Stock PrivateGPT Foundation

Inherited base capabilities the custom work builds on. Port these first (or ensure the target
already has equivalents).

## B1. OpenAI-Compatible API Server
FastAPI app exposing OpenAI-shaped endpoints so existing OpenAI clients work unchanged:
`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, plus PrivateGPT-specific
`/v1/ingest/*`, `/v1/chunks`, `/v1/summarize`, `/health`. Streaming (SSE) supported. Optional
bearer-token auth (`server/utils/auth.py`).

## B2. RAG Pipeline
Ingestion: file → LlamaIndex readers → `SentenceWindowNodeParser` chunking →
`EmbeddingComponent` → `VectorStoreComponent` + `NodeStoreComponent`.
Chat-with-context: query → embed → vector similarity search → top-k chunks → LLM with context
→ response + source attribution. `RagSettings`: `similarity_top_k`, `similarity_value`,
optional reranker (`RerankSettings`).

## B3. Swappable LLM Backends
`LLMSettings.mode`: `llamacpp | openai | openailike | azopenai | sagemaker | mock | ollama |
gemini`. Each constructed in `components/llm/llm_component.py`. Ollama path adds keep-alive,
autopull, connection retry (`utils/ollama_utils.py`, `utils/retry.py`).

## B4. Swappable Embedding Backends
`EmbeddingSettings.mode`: `huggingface | openai | azopenai | sagemaker | ollama | mock |
gemini | mistralai`. Ingest modes: `simple | batch | parallel | pipeline`
(`get_ingestion_component`), with multiprocessing/threaded pipelines for throughput.

## B5. Swappable Vector Stores
`VectorstoreSettings.database`: `chroma | qdrant | postgres | clickhouse | milvus`. Batched
Chroma wrapper (`components/vector_store/batched_chroma.py`) to respect max batch size.

## B6. Node/Doc Store
`components/node_store/node_store_component.py` — docstore + index store (Simple/persistent).
Shared across collections in this project (see A1).

## B7. Chunks API
`/v1/chunks` — retrieve top-k relevant chunks for a query without generating an LLM answer
(`server/chunks/`). Useful for custom retrieval clients.

## B8. Summarize Recipe
`server/recipes/summarize/` — `/v1/summarize` endpoint, single-doc/context summarization with
`TREE_SUMMARIZE`. (Extended by A8 for batch + map-reduce.)

## B9. Multi-Format Document Readers
`ingest_helper.py:FILE_READER_CLS` — pdf, docx, pptx, hwp, epub, md, csv, json, ipynb, mbox,
images, audio/video (whisper). (Extended by A5/A6.)

## B10. Gradio Web UI
`ui/ui.py` mounted at a configurable path — chat interface, mode selector, file upload,
ingested-file list, system-prompt editor. (Extended by A9/A11.)

## B11. Configuration System
`settings/` — layered YAML profiles merged by `PGPT_PROFILES` (comma-separated), `settings.yaml`
always the base, `${VAR:default}` env substitution, parsed into Pydantic models. Profiles:
local, ollama, ollama-pg, openai, azopenai, gemini, sagemaker, vllm, docker, mock, test.

## B12. Dependency Injection Architecture
`injector` library; global injector in `di.py`; request handlers use `request.state.injector`.
Router/Service separation per API; services depend on abstractions (`LLM`, `BaseEmbedding`,
`VectorStore`), implementations swapped via Components.

## B13. Utility Scripts
`scripts/ingest_folder.py` (bulk folder ingest), `scripts/utils.py` (`stats`, `wipe`),
`scripts/extract_openapi.py` (OpenAPI spec generation).
