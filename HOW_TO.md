# PrivateGPT — How-To Guide

## Table of Contents

1. [Running the App](#1-running-the-app)
2. [Understanding Collections](#2-understanding-collections)
3. [Uploading Files to a Collection](#3-uploading-files-to-a-collection)
4. [File Watching & Auto-Ingestion](#4-file-watching--auto-ingestion)
5. [Querying with Collections](#5-querying-with-collections)
6. [Precautions & Common Pitfalls](#6-precautions--common-pitfalls)

---

## 1. Running the App

### Prerequisites

- [Ollama](https://ollama.com) must be running locally on port `11434`
- The required LLM and embedding models must already be pulled in Ollama:
  - LLM: `gemma3:4b-it-qat`
  - Embedding: `embeddinggemma:latest`
- A Qdrant vector database must be running on port `6333`
- Python dependencies must be installed via Poetry

### Starting the Application

```bash
PGPT_PROFILES=ollama make run
```

This loads the `settings-ollama.yaml` profile on top of the base `settings.yaml`, which configures:
- Ollama as the LLM and embedding backend
- Qdrant as the vector store (at `http://localhost:6333`)
- The web UI at `http://localhost:8001`

### Development Mode (with auto-reload)

```bash
PGPT_PROFILES=ollama make dev
```

> **Note:** Dev mode runs on port `8001` with live reload enabled — useful during development but not suitable for production.

---

## 2. Understanding Collections

The app uses three distinct Qdrant collections to organize your documents:

| Collection | Name in settings | Purpose |
|---|---|---|
| **Default** | `make_this_parameterizable_per_api_call` | Backward-compatible fallback collection |
| **Persistent** | `persistent_docs` | Long-term, stable documents (e.g., reference material) |
| **Temporary** | `temporary_docs` | Short-lived documents for ad-hoc analysis |

### Physical Paths

| Collection | Filesystem Path |
|---|---|
| Persistent | `persistent_docs/` (project root) |
| Temporary | `temp_docs/` (project root) |

The **Temporary** collection path (`temp_docs/`) is actively **file-watched** — any file placed in or removed from that folder is automatically ingested or de-indexed.

The **Persistent** collection path (`persistent_docs/`) is **not file-watched** — documents must be ingested explicitly through the UI upload button.

---

## 3. Uploading Files to a Collection

> **Critical rule: always select the target collection *before* uploading files.**

The collection dropdown in the UI controls both which files are listed in the file panel and which collection a newly uploaded file is ingested into. Uploading without selecting the correct collection will send the file to whichever collection is currently active.

### Step-by-Step: Upload to the Persistent Collection

1. Open the UI at `http://localhost:8001`
2. In the **Collection** dropdown (left panel), select **Persistent (persistent_docs)**
3. Use the **Upload File** button to select and upload your document
4. Wait for ingestion to complete — the file will appear in the file list
5. The document is now stored in the `persistent_docs` Qdrant collection

> **Persistent collection only supports upload via the Upload button.** Do not copy files directly into `persistent_docs/` on disk and expect them to appear — that folder is not watched and no auto-ingestion occurs for it.

### Step-by-Step: Upload to the Temporary Collection

The temporary collection offers **two ingestion methods**:

**Method A — Upload Button (UI)**

1. In the **Collection** dropdown, select **Temporary (temporary_docs)**
2. Use the **Upload File** button to select your document
3. The file is ingested directly into `temporary_docs` Qdrant collection

**Method B — File Manager (copy to `temp_docs/` folder)**

1. Copy or move your file into the `temp_docs/` directory on disk
2. The file watcher detects the new file automatically
3. Ingestion into the `temporary_docs` collection happens in the background — no UI action required
4. Refresh the UI file list to confirm the document appears

> **Both methods work for the temporary collection.** Use whichever fits your workflow — the end result is the same.

### Step-by-Step: Upload to the Default Collection

1. In the **Collection** dropdown, select **Default (make_this_parameterizable_per_api_call)**
2. Use the **Upload File** button to upload your document
3. The document is ingested into the default collection

---

## 4. File Watching & Auto-Ingestion

The `temp_docs/` folder is monitored by a background file watcher. The relevant settings (in `settings.yaml`) are:

```yaml
data:
  paths:
    watch_enabled: true          # Master switch for file watching
    watch_modifications: false   # If true, re-ingests files when they are modified
    sync_on_startup: true        # Ingests any existing files in temp_docs/ at startup
    create_paths_if_missing: true
```

### What the watcher does

| Event | Action |
|---|---|
| File added to `temp_docs/` | Automatically ingested into `temporary_docs` collection |
| File deleted from `temp_docs/` | Automatically removed from `temporary_docs` collection |
| File modified (if `watch_modifications: true`) | Old document deleted and re-ingested with new content |

### Startup sync

With `sync_on_startup: true`, any files already present in `temp_docs/` at launch time are ingested automatically if they are not already tracked. This means the collection stays consistent across restarts without manual re-uploading.

---

## 5. Querying with Collections

The collection dropdown also controls which collection is searched during chat:

- **All Collections** — searches across every available collection
- **Default** — searches only the default collection
- **Persistent** — searches only persistent documents
- **Temporary** — searches only temporary documents

To scope your chat or search to a specific set of documents, select the matching collection in the dropdown **before** asking your question.

In **RAG** and **Search** modes, you can further narrow context to a single file by clicking that file in the file list panel.

---

## 6. Precautions & Common Pitfalls

### Always select the collection before uploading

Uploading a file without first switching the collection dropdown will ingest the file into the wrong collection. There is no automatic move between collections — you would need to delete the document and re-upload it to the correct one.

### Do not copy files directly into `persistent_docs/`

That folder is not watched. Files placed there manually will **not** be ingested. Use the Upload button in the UI with the Persistent collection selected instead.

### Deleting from `temp_docs/` de-indexes the file

If you delete a file from the `temp_docs/` directory, the watcher will automatically remove it from the `temporary_docs` Qdrant collection. This is by design — do not use `temp_docs/` as a general storage location if you want the documents to persist in the index.

### File modifications are not re-ingested by default

`watch_modifications` is set to `false`. If you update a file in `temp_docs/`, the vector store will still hold the old version. To force a refresh, delete and re-add the file, or set `watch_modifications: true` in `settings.yaml` (note: this increases CPU load as every save triggers re-embedding).

### Qdrant and Ollama must be running before starting PrivateGPT

If either service is down at startup, the app will fail to initialize embeddings or the vector store. Start them first:

```bash
# Start Qdrant (Docker example)
docker run -p 6333:6333 qdrant/qdrant

# Start Ollama
ollama serve
```

### Model availability in Ollama

The configured models (`gemma3:4b-it-qat` and `embeddinggemma:latest`) must be pulled before running:

```bash
ollama pull gemma3:4b-it-qat
ollama pull embeddinggemma:latest
```

If a model is missing, PrivateGPT will fail to answer queries or generate embeddings.

### Context window limits

The LLM context window is set to `3900` tokens. If you upload very large documents and ask complex questions, responses may be truncated or incomplete. For large documents, prefer **Summarize** mode or narrow context by selecting a specific file.

### Duplicate ingestion is prevented

The file tracker prevents the same file from being ingested twice into the same collection. However, if you upload the same file through the UI upload button while the file watcher already tracked it via the filesystem, the UI upload may create a duplicate entry. Check the file list after uploading if you are unsure.
