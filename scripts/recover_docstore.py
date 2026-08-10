"""Rebuild a lost/empty docstore.json from the Qdrant vector store.

Background
----------
With `vectorstore.database: qdrant` and `nodestore.database: simple`, LlamaIndex
stores the *entire* serialized node (text + metadata + relationships) inside each
Qdrant point's payload under the `_node_content` key -- not just the embedding.
That makes `docstore.json` a derived cache of what already lives in Qdrant, so it
can be fully reconstructed if it gets truncated/emptied (e.g. a crash mid-ingest).

The docstore is SHARED across all collections (only the vector store is
per-collection), so by default this script scrolls EVERY Qdrant collection and
rebuilds one combined docstore.

Usage
-----
    # dry run first -- counts points, writes nothing
    PGPT_PROFILES=local uv run python scripts/recover_docstore.py --dry-run

    # rebuild all collections into local_data/docstore.json (backs up any existing file)
    PGPT_PROFILES=local uv run python scripts/recover_docstore.py

    # only certain collections
    PGPT_PROFILES=local uv run python scripts/recover_docstore.py -c my_collection -c other

Run it on the affected machine (same env/profile as the app) so it uses the same
Qdrant connection and the same local_data path.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from qdrant_client import QdrantClient

from private_gpt.paths import local_data_path
from private_gpt.settings.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("recover_docstore")

SCROLL_BATCH = 5000


def _make_client() -> QdrantClient:
    cfg = settings()
    if cfg.qdrant is None:
        logger.info("No qdrant settings found; connecting to localhost:6333")
        return QdrantClient()
    return QdrantClient(**cfg.qdrant.model_dump(exclude_none=True))


def _iter_points(client: QdrantClient, collection: str):
    """Yield all point payloads for a collection (no vectors, payload only)."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=False,
            limit=SCROLL_BATCH,
            offset=offset,
        )
        for p in points:
            yield p.payload or {}
        if offset is None:
            break


def rebuild(collections: list[str] | None, dry_run: bool) -> None:
    client = _make_client()

    all_collections = [c.name for c in client.get_collections().collections]
    targets = collections or all_collections
    missing = [c for c in targets if c not in all_collections]
    if missing:
        logger.warning("Requested collections not found in Qdrant: %s", missing)
    targets = [c for c in targets if c in all_collections]
    if not targets:
        logger.error("No matching collections to recover. Available: %s", all_collections)
        sys.exit(1)

    logger.info("Recovering from collections: %s", targets)

    docstore = SimpleDocumentStore()
    total_seen = 0
    total_ok = 0
    total_failed = 0
    start = time.time()

    for collection in targets:
        count = client.count(collection_name=collection, exact=True).count
        logger.info("Collection %r has %d points", collection, count)
        batch: list[BaseNode] = []
        seen = 0
        for payload in _iter_points(client, collection):
            seen += 1
            total_seen += 1
            try:
                node = metadata_dict_to_node(payload)
                batch.append(node)
            except Exception as e:  # noqa: BLE001
                total_failed += 1
                if total_failed <= 10:
                    logger.warning("Skipping a point (%s): %s", collection, e)
            # flush periodically to bound memory
            if len(batch) >= SCROLL_BATCH:
                if not dry_run:
                    docstore.add_documents(batch, allow_update=True)
                total_ok += len(batch)
                batch = []
                logger.info(
                    "  %s: %d/%d processed (%d total ok)",
                    collection,
                    seen,
                    count,
                    total_ok,
                )
        if batch:
            if not dry_run:
                docstore.add_documents(batch, allow_update=True)
            total_ok += len(batch)

    elapsed = time.time() - start
    logger.info(
        "Done scanning: %d points seen, %d nodes rebuilt, %d failed (%.1fs)",
        total_seen,
        total_ok,
        total_failed,
        elapsed,
    )
    logger.info(
        "Rebuilt docstore holds %d ref-docs (source files/documents)",
        len(docstore.get_all_ref_doc_info() or {}),
    )

    if dry_run:
        logger.info("Dry run -- nothing written. Re-run without --dry-run to persist.")
        return

    target_file = Path(local_data_path) / "docstore.json"
    if target_file.exists() and target_file.stat().st_size > 0:
        backup = target_file.with_suffix(f".json.bak.{int(time.time())}")
        shutil.copy2(target_file, backup)
        logger.info("Backed up existing docstore.json -> %s", backup.name)

    docstore.persist(persist_path=str(target_file))
    logger.info("Wrote %s (%d bytes)", target_file, target_file.stat().st_size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-c",
        "--collection",
        action="append",
        dest="collections",
        help="Collection to recover (repeatable). Default: all Qdrant collections.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Count/parse points but write nothing.",
    )
    args = ap.parse_args()
    rebuild(args.collections, args.dry_run)


if __name__ == "__main__":
    main()
