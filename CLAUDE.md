# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PrivateGPT is a production-ready AI project that enables users to ask questions about their documents using Large Language Models (LLMs) in completely offline, privacy-focused environments. It provides an OpenAI-compatible API with both high-level RAG (Retrieval Augmented Generation) capabilities and low-level primitives for building custom AI pipelines.

**Key Characteristics:**
- 100% private - no data leaves the execution environment
- Enterprise-ready for data-sensitive domains (healthcare, legal)
- OpenAI API compatible with streaming support
- Built on FastAPI and LlamaIndex
- Multi-backend support (LLMs, embeddings, vector stores)

## Architecture & Design Patterns

### Core Architectural Principles

1. **Dependency Injection Pattern**
   - Uses `injector` library for DI throughout the codebase
   - Global injector created in [private_gpt/di.py](private_gpt/di.py)
   - Per-request injector access via `request.state.injector`
   - Components are auto-bound and resolved via `@inject` decorator
   - Prefer using request-scoped injector over global reference for testability

2. **LlamaIndex Abstractions**
   - Services use base abstractions: `LLM`, `BaseEmbedding`, `VectorStore`
   - Implementations are swappable via Components (e.g., LLMComponent, EmbeddingComponent)
   - This decouples service logic from specific LLM/embedding/database providers

3. **Service/Router Separation**
   - APIs defined in `private_gpt/server/<api>/`
   - Each API has: `<api>_router.py` (FastAPI layer) and `<api>_service.py` (business logic)
   - Routers handle HTTP concerns; Services implement RAG/AI logic

4. **Component System**
   - Components in `private_gpt/components/<component>/`
   - Each Component provides concrete implementations of abstractions
   - Examples: LLMComponent (LlamaCPP, Ollama, OpenAI), VectorStoreComponent (Qdrant, Postgres)

### Directory Structure

```
private_gpt/
├── components/          # Dependency injection components
│   ├── llm/            # LLM implementations (LlamaCPP, Ollama, OpenAI, etc.)
│   ├── embedding/      # Embedding models (HuggingFace, OpenAI, Ollama)
│   ├── vector_store/   # Vector DBs (Qdrant, Postgres, Chroma)
│   ├── node_store/     # Document/index storage
│   └── ingest/         # Document ingestion pipeline
├── server/             # FastAPI routers & services
│   ├── chat/           # Chat completions with RAG
│   ├── completions/    # Prompt completions
│   ├── embeddings/     # Embeddings API
│   ├── chunks/         # Contextual chunk retrieval
│   ├── ingest/         # Document upload/management
│   └── health/         # Health check endpoint
├── open_ai/            # OpenAI API compatibility layer
├── settings/           # Pydantic-based configuration
├── ui/                 # Gradio web interface
├── launcher.py         # FastAPI app initialization
└── di.py              # Dependency injection setup
```

### Key API Endpoints

Base path: `/v1/`

- `POST /v1/chat/completions` - Chat with RAG context
- `POST /v1/completions` - Prompt completions
- `POST /v1/embeddings` - Generate embeddings
- `POST /v1/ingest/file` - Upload and ingest document
- `POST /v1/ingest/text` - Ingest raw text
- `GET /v1/ingest/list` - List ingested documents
- `DELETE /v1/ingest/{doc_id}` - Delete document
- `GET /v1/chunks` - Retrieve contextual chunks
- `POST /v1/recipes/summarize` - Document summarization
- `GET /v1/health` - Health check

## Development Commands

All commands use Poetry for dependency management. Key commands from [Makefile](Makefile):

### Quality Checks (Required Before Commits)
```bash
make check          # Run format + mypy (REQUIRED before commits)
make test           # Run pytest tests
make format         # Auto-fix formatting (black + ruff --fix)
make mypy           # Type checking
make test-coverage  # Tests with HTML/XML coverage reports
```

### Running the Application
```bash
make run            # Production: python -m private_gpt
make dev            # Dev mode with auto-reload (PGPT_PROFILES=local, port 8001)
make dev-windows    # Windows-compatible dev mode
```

### Utilities
```bash
make ingest <path>  # Bulk document ingestion from folder
make stats          # Show storage statistics
make wipe           # Clean all storage data
make api-docs       # Generate OpenAPI spec to fern/openapi/openapi.json
```

## Configuration System

Settings are managed via YAML files and Pydantic models ([private_gpt/settings/settings.py](private_gpt/settings/settings.py)).

### Configuration Profiles

Set via `PGPT_PROFILES` environment variable:
- `settings.yaml` - Default configuration
- `settings-ollama.yaml` - Ollama backend
- `settings-local.yaml` - Local LlamaCPP
- `settings-mock.yaml` - Mock models for testing
- `settings-openai.yaml` - OpenAI API

### Key Settings Categories

- **Server**: Port (default 8001), CORS, basic auth
- **LLM**: Mode (llamacpp/ollama/openai/etc), temperature, context window, tokenizer
- **Embedding**: Model selection, cache directory
- **Vector Store**: Backend (qdrant/postgres/chroma), collection names
- **Data Paths**: Persistent/temporary storage, file watching configuration
- **UI**: Gradio interface, default mode (RAG/Search/Basic/Summarize)

### Environment Variables

Key environment variables:
- `PGPT_PROFILES` - Configuration profile to load
- `PORT` - Server port (default: 8001)
- `APP_ENV` - Environment name (prod/staging/local)

## Testing & Code Quality

### Pre-Commit Requirements

**CRITICAL**: Before every commit, run:
```bash
make check    # Formats code and runs mypy
make test     # Runs all tests
```

These checks are enforced in CI/CD (.github/workflows/tests.yml).

### Testing Infrastructure

- Tests located in `tests/` directory
- Fixtures in `tests/fixtures/` (auto-loaded via conftest.py)
- Key fixtures: FastAPI test client, mock injector, Qdrant management
- Run individual tests: `PYTHONPATH=. poetry run pytest tests/path/to/test.py`

### Code Quality Tools

- **Black**: Code formatting (line length, style)
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking (required for all code)

## Current Branch Context

**Branch**: `feature/parametrization`

This branch implements per-API-call parametrization of collections and paths. Key changes:

### New Components
- [private_gpt/components/ingest/file_tracker.py](private_gpt/components/ingest/file_tracker.py) - Track ingested files to avoid re-ingestion
- [private_gpt/server/ingest/watched_path_manager.py](private_gpt/server/ingest/watched_path_manager.py) - Manage file watchers for automatic ingestion

### Modified Behavior
- Ingest service now supports per-call collection names
- File watching system automatically ingests new files from configured paths
- Settings include new `DataPathsSettings` for persistent/temporary storage
  - `watch_enabled`: Enable automatic file watching
  - `watch_modifications`: Re-ingest on file modifications
  - `sync_on_startup`: Sync existing files on startup
  - Dual-path system: persistent_path (D:/) and temporary_path (E:/)

### Integration Points
- [private_gpt/launcher.py:73-110](private_gpt/launcher.py#L73-L110) - File watcher initialization during app startup
- Vector store component supports multiple collections via factory method
- Settings now use Pydantic for validation and environment variable injection

## Application Initialization Flow

1. `__main__.py` - Entry point, launches Uvicorn server
2. [launcher.py](private_gpt/launcher.py) - `create_app(root_injector)` sets up FastAPI:
   - Binds injector to each request via dependency
   - Mounts all routers (chat, ingest, chunks, etc.)
   - Configures CORS middleware
   - Initializes Gradio UI (if enabled)
   - Starts file watchers for automatic ingestion
   - Registers shutdown handlers
3. [di.py](private_gpt/di.py) - Creates global injector with auto-binding

## RAG Pipeline Flow

### Document Ingestion
1. API receives document (file/text/binary)
2. IngestService parses content via LlamaIndex readers
3. Document split into chunks (SentenceWindowNodeParser)
4. EmbeddingComponent generates vector embeddings
5. Stored in VectorStoreComponent + NodeStoreComponent
6. Returns doc_id and metadata

### Chat with Context
1. User question → EmbeddingComponent generates query embedding
2. VectorStoreComponent performs similarity search
3. ChunksService retrieves top-k relevant chunks
4. Context combined with system prompt + user message
5. LLMComponent generates response
6. Returns answer + source attribution

## Important Notes

- Python 3.8+ required
- Primary documentation: https://docs.privategpt.dev/
- Use `request.state.injector` (not global_injector) in request handlers
- All services should depend on abstractions (LLM, BaseEmbedding), not implementations
- File paths in settings use environment variable substitution: `${VAR:default}`
- Collection names are now parameterizable per API call (new feature)
