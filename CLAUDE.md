# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PrivateGPT is a production-ready AI project that enables users to ask questions about their documents using LLMs in completely offline, privacy-focused environments. It provides an OpenAI-compatible API with RAG (Retrieval Augmented Generation) capabilities built on FastAPI and LlamaIndex.

## Development Commands

This project uses **`uv`** for dependency management (not Poetry — no `poetry.lock` exists). The `Makefile` still invokes `poetry run`, so when running commands manually use `uv run` instead:

```bash
# Quality checks (required before commits)
uv run pytest tests                      # Run all tests
uv run pytest tests/path/to/test.py     # Run single test file
uv run black .                           # Format code
uv run ruff check private_gpt tests --fix
uv run mypy private_gpt                 # Type checking

# Running the app
uv run python -m private_gpt            # Production
PGPT_PROFILES=local uv run python -m uvicorn private_gpt.main:app --reload --port 8001  # Dev

# Utilities
PGPT_PROFILES=mock uv run python scripts/extract_openapi.py private_gpt.main:app --out fern/openapi/openapi.json
uv run python scripts/ingest_folder.py <path>
uv run python scripts/utils.py stats
uv run python scripts/utils.py wipe
```

**Pre-commit requirement**: run `black` + `ruff` + `mypy` before committing. CI enforces these via `.github/workflows/tests.yml`.

## Architecture

### Core Patterns

1. **Dependency Injection** — uses `injector` library. Global injector in `private_gpt/di.py`. Always use `request.state.injector` in request handlers, not the global reference (harder to test).

2. **LlamaIndex abstractions** — services depend on `LLM`, `BaseEmbedding`, `VectorStore` base types. Implementations are swappable via Components.

3. **Router/Service separation** — each API has a `*_router.py` (FastAPI/HTTP layer) and `*_service.py` (business logic) in `private_gpt/server/<api>/`.

4. **Component system** — `private_gpt/components/<component>/` provides concrete DI-injected implementations for LLM, embedding, vector store, node store, and ingest.

### Key Directories

```
private_gpt/
├── components/          # DI components (llm/, embedding/, vector_store/, node_store/, ingest/)
├── server/              # FastAPI routers & services (chat/, completions/, embeddings/, chunks/, ingest/, recipes/, health/)
├── open_ai/             # OpenAI API compatibility layer and ContextFilter extension
├── settings/            # Pydantic settings models + YAML loader
├── ui/                  # Gradio web interface
├── utils/               # Shared utilities (collection_mapper, retry, eta, etc.)
├── launcher.py          # FastAPI app factory + file watcher init
├── di.py                # Global injector creation
└── paths.py             # Filesystem path constants
```

### Application Startup Flow

1. `__main__.py` → starts Uvicorn
2. `launcher.py:create_app(root_injector)` → mounts all routers, configures CORS, mounts Gradio UI, initializes file watchers
3. `di.py:create_application_injector()` → creates global `Injector(auto_bind=True)` with `Settings` bound

### Multi-Collection Vector Store

`VectorStoreComponent` maintains a cache of vector stores by collection name (`_vector_stores: dict[str, ...]`). Use `get_vector_store(collection_name)` to obtain a named collection. `IngestService` also caches `StorageContext` and ingest components per collection.

`utils/collection_mapper.py:get_collection_for_path()` maps a file path to a collection name based on `data.paths.persistent_path` / `data.paths.temporary_path` settings.

### File Watching System

`server/ingest/watched_path_manager.py` (`WatchedPathManager`) manages watchdog-based file watchers started at app startup. `components/ingest/file_tracker.py` (`FileTracker`) persists file-path→doc-id mappings to disk (under `local_data_path/file_tracker/`) so deletions survive restarts.

## Configuration

Settings are YAML files loaded via `settings/settings_loader.py`, merged in order, and parsed into Pydantic models in `settings/settings.py`.

**Active profiles**: set `PGPT_PROFILES=<name>` (comma-separated); `settings.yaml` is always loaded as base.

Available profile files: `settings-local.yaml`, `settings-ollama.yaml`, `settings-ollama-pg.yaml`, `settings-openai.yaml`, `settings-azopenai.yaml`, `settings-gemini.yaml`, `settings-sagemaker.yaml`, `settings-vllm.yaml`, `settings-docker.yaml`, `settings-mock.yaml`, `settings-test.yaml`.

**Key settings classes** (all in `settings/settings.py`):
- `LLMSettings.mode`: `llamacpp | openai | openailike | azopenai | sagemaker | mock | ollama | gemini`
- `EmbeddingSettings.mode`: `huggingface | openai | azopenai | sagemaker | ollama | mock | gemini | mistralai`
- `VectorstoreSettings.database`: `chroma | qdrant | postgres | clickhouse | milvus`
- `DataPathsSettings`: persistent/temporary dual-path system with file watching config
- `RagSettings`: `similarity_top_k`, `similarity_value`, reranker config

Environment variable substitution in YAML values uses `${VAR:default}` syntax.

## RAG Pipeline

**Ingestion**: file/text → LlamaIndex readers → `SentenceWindowNodeParser` chunking → `EmbeddingComponent` → `VectorStoreComponent` + `NodeStoreComponent`

**Chat with context**: query → embedding → vector similarity search → top-k chunks → LLM with context → response + source attribution

## Testing

Tests in `tests/`. Fixtures auto-loaded from `tests/fixtures/` via `conftest.py`. Key fixtures: `fast_api_test_client.py`, `mock_injector.py`, `auto_close_qdrant.py`.

Run with `PYTHONPATH=.` set (conftest also enforces the correct working directory).

## Important Constraints

- Python 3.11+ required (3.12 in use per `.python-version`)
- All services must depend on abstractions (`LLM`, `BaseEmbedding`), not concrete implementations
- MyPy typing is required for all new code
- Use `request.state.injector` (not `global_injector`) in request handlers
