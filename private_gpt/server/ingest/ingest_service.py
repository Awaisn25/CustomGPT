import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, AnyStr, BinaryIO

from injector import inject, singleton
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.storage import StorageContext

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import get_ingestion_component
from private_gpt.components.ingest.pdf_converter import convert_to_pdf, needs_conversion
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.settings.settings import Settings, settings
from private_gpt.utils.collection_mapper import get_collection_for_path

if TYPE_CHECKING:
    from llama_index.core.storage.docstore.types import RefDocInfo

logger = logging.getLogger(__name__)


@singleton
class IngestService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
        settings: Settings,
    ) -> None:
        self.llm_service = llm_component
        self.vector_store_component = vector_store_component
        self.embedding_component = embedding_component
        self.node_store_component = node_store_component
        self.settings = settings
        # Cache for storage contexts and ingest components per collection
        self._storage_contexts: dict[str, StorageContext] = {}
        self._ingest_components: dict[str, Any] = {}

    def _get_storage_context(
        self, collection_name: str | None = None
    ) -> StorageContext:
        """Get or create storage context for the specified collection."""
        if collection_name is None:
            collection_name = self.settings.vectorstore.default_collection_name

        if collection_name not in self._storage_contexts:
            vector_store = self.vector_store_component.get_vector_store(
                collection_name
            )
            self._storage_contexts[collection_name] = StorageContext.from_defaults(
                vector_store=vector_store,
                docstore=self.node_store_component.doc_store,
                index_store=self.node_store_component.index_store,
            )
        return self._storage_contexts[collection_name]

    def _get_ingest_component(self, collection_name: str | None = None) -> Any:
        """Get or create ingest component for the specified collection."""
        if collection_name is None:
            collection_name = self.settings.vectorstore.default_collection_name

        if collection_name not in self._ingest_components:
            storage_context = self._get_storage_context(collection_name)
            node_parser = SentenceWindowNodeParser.from_defaults()

            self._ingest_components[collection_name] = get_ingestion_component(
                storage_context,
                embed_model=self.embedding_component.embedding_model,
                transformations=[
                    node_parser,
                    self.embedding_component.embedding_model,
                ],
                settings=self.settings,
            )
        return self._ingest_components[collection_name]

    def _ingest_data(
        self,
        file_name: str,
        file_data: AnyStr,
        collection_name: str | None = None,
        file_path: Path | None = None,
    ) -> list[IngestedDoc]:
        logger.debug("Got file data of size=%s to ingest", len(file_data))
        # llama-index mainly supports reading from files, so
        # we have to create a tmp file to read for it to work
        # delete=False to avoid a Windows 11 permission error.
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                path_to_tmp = Path(tmp.name)
                if isinstance(file_data, bytes):
                    path_to_tmp.write_bytes(file_data)
                else:
                    path_to_tmp.write_text(str(file_data))
                # Use file_path if provided, otherwise use tmp path
                actual_path = file_path if file_path else path_to_tmp
                return self.ingest_file(
                    file_name, path_to_tmp, collection_name=collection_name
                )
            finally:
                tmp.close()
                path_to_tmp.unlink()

    def ingest_file(
        self,
        file_name: str,
        file_data: Path,
        collection_name: str | None = None,
    ) -> list[IngestedDoc]:
        """Ingest a file into the specified collection.

        Args:
            file_name: Name of the file
            file_data: Path to the file
            collection_name: Optional collection name. If not provided, will be auto-detected from file path.

        Returns:
            List of ingested documents
        """
        # Auto-detect collection from file path if not provided
        if collection_name is None:
            collection_name = get_collection_for_path(file_data, self.settings)

        # Convert to PDF when the original format can't be rendered in browsers
        effective_path = file_data
        effective_name = file_name
        if needs_conversion(file_data):
            pdf_path = convert_to_pdf(file_data)
            if pdf_path is not None:
                effective_path = pdf_path
                effective_name = pdf_path.name

        logger.info(
            "Ingesting file_name=%s into collection=%s", effective_name, collection_name
        )
        ingest_component = self._get_ingest_component(collection_name)
        documents = ingest_component.ingest(effective_name, effective_path, collection_name=collection_name)
        logger.info(
            "Finished ingestion file_name=%s into collection=%s",
            effective_name,
            collection_name,
        )
        return [IngestedDoc.from_document(document) for document in documents]

    def ingest_text(
        self,
        file_name: str,
        text: str,
        collection_name: str | None = None,
    ) -> list[IngestedDoc]:
        """Ingest text into the specified collection.

        Args:
            file_name: Name of the file
            text: Text content to ingest
            collection_name: Optional collection name. If not provided, uses default collection.

        Returns:
            List of ingested documents
        """
        logger.debug(
            "Ingesting text data with file_name=%s into collection=%s",
            file_name,
            collection_name,
        )
        return self._ingest_data(file_name, text, collection_name=collection_name)

    def ingest_bin_data(
        self,
        file_name: str,
        raw_file_data: BinaryIO,
        
        collection_name: str | None = None,
    ) -> list[IngestedDoc]:
        """Ingest binary data into the specified collection.

        Args:
            file_name: Name of the file
            raw_file_data: Binary file data
            collection_name: Optional collection name. If not provided, uses default collection.

        Returns:
            List of ingested documents
        """
        logger.debug(
            "Ingesting binary data with file_name=%s into collection=%s",
            file_name,
            collection_name,
        )
        file_data = raw_file_data.read()
        return self._ingest_data(
            file_name, file_data, collection_name=collection_name
        )

    def bulk_ingest(
        self,
        files: list[tuple[str, Path]],
        collection_name: str | None = None,
    ) -> list[IngestedDoc]:
        """Bulk ingest files into the specified collection.

        Args:
            files: List of (file_name, file_path) tuples
            collection_name: Optional collection name. If not provided, will be auto-detected from file paths.

        Returns:
            List of ingested documents
        """
        logger.info("Ingesting file_names=%s", [f[0] for f in files])

        # Group files by collection if collection_name is not specified
        if collection_name is None:
            # Group files by their detected collection
            files_by_collection: dict[str, list[tuple[str, Path]]] = {}
            for file_name, file_path in files:
                detected_collection = get_collection_for_path(
                    file_path, self.settings
                )
                if detected_collection not in files_by_collection:
                    files_by_collection[detected_collection] = []
                files_by_collection[detected_collection].append(
                    (file_name, file_path)
                )

            # Ingest each group separately
            all_documents = []
            for coll_name, coll_files in files_by_collection.items():
                ingest_component = self._get_ingest_component(coll_name)
                documents = ingest_component.bulk_ingest(coll_files, collection_name=coll_name)
                all_documents.extend(documents)
            logger.info("Finished bulk ingestion")
            return [IngestedDoc.from_document(doc) for doc in all_documents]
        else:
            # All files go to the same collection
            ingest_component = self._get_ingest_component(collection_name)
            is_temporary = True if collection_name == self.settings.data.paths.temporary_collection_name else False
            documents = ingest_component.bulk_ingest(files, collection_name=collection_name, is_temporary=is_temporary)
            logger.info("Finished bulk ingestion into collection=%s", collection_name)
            return [IngestedDoc.from_document(document) for document in documents]

    def list_ingested(
        self, collection_name: str | None = None
    ) -> list[IngestedDoc]:
        """List ingested documents, optionally filtered by collection.

        Args:
            collection_name: Optional collection name to filter by. If None, lists all documents.

        Returns:
            List of ingested documents
        """
        ingested_docs: list[IngestedDoc] = []
        try:
            # The docstore is shared across all collections (only the vector store is
            # per-collection). Fetch everything from the docstore, then filter by the
            # collection_name stored in each document's metadata.
            storage_context = self._get_storage_context(
                collection_name or self.settings.vectorstore.default_collection_name
            )
            ref_docs: dict[str, RefDocInfo] | None = (
                storage_context.docstore.get_all_ref_doc_info()
            )

            if not ref_docs:
                return ingested_docs

            for doc_id, ref_doc_info in ref_docs.items():
                if ref_doc_info is None or ref_doc_info.metadata is None:
                    continue

                # Filter by collection_name stored in metadata (the docstore is shared,
                # so without this we would return documents from every collection).
                if collection_name is not None:
                    doc_collection = ref_doc_info.metadata.get("collection_name")
                    if doc_collection != collection_name:
                        continue

                doc_metadata = IngestedDoc.curate_metadata(ref_doc_info.metadata)
                ingested_docs.append(
                    IngestedDoc(
                        object="ingest.document",
                        doc_id=doc_id,
                        doc_metadata=doc_metadata,
                    )
                )
        except ValueError:
            logger.warning("Got an exception when getting list of docs", exc_info=True)
            pass
        logger.debug(
            "Found count=%s ingested documents (collection=%s)",
            len(ingested_docs),
            collection_name,
        )
        return ingested_docs

    def delete(
        self, doc_id: str, collection_name: str | None = None
    ) -> None:
        """Delete an ingested document.

        Args:
            doc_id: Document ID to delete
            collection_name: Optional collection name. If not provided, tries to find the document in all collections.

        :raises ValueError: if the document does not exist
        """
        logger.info(
            "Deleting the ingested document=%s from collection=%s",
            doc_id,
            collection_name,
        )
        if collection_name:
            ingest_component = self._get_ingest_component(collection_name)
            ingest_component.delete(doc_id)
        else:
            # Try to delete from default collection first
            # Note: This may need refinement to search across all collections
            ingest_component = self._get_ingest_component(
                self.settings.vectorstore.default_collection_name
            )
            ingest_component.delete(doc_id)

    def get_document_file_path(
        self, doc_id: str, collection_name: str | None = None
    ) -> Path:
        """Get the file path for a document by its ID.

        Args:
            doc_id: Document ID to look up
            collection_name: Optional collection name. If not provided, searches default collection.

        Returns:
            Path to the source file

        Raises:
            ValueError: If document not found or source_path not available
            FileNotFoundError: If the file no longer exists on disk
            PermissionError: If the file path is outside allowed directories
        """
        # Get the document metadata
        if collection_name:
            storage_context = self._get_storage_context(collection_name)
        else:
            storage_context = self._get_storage_context(
                self.settings.vectorstore.default_collection_name
            )

        docstore = storage_context.docstore
        ref_doc_info = docstore.get_ref_doc_info(doc_id)

        if ref_doc_info is None or ref_doc_info.metadata is None:
            raise ValueError(f"Document with ID {doc_id} not found")

        # Extract source_path from metadata
        source_path_str = ref_doc_info.metadata.get("source_path")
        if not source_path_str:
            raise ValueError(
                f"Document {doc_id} does not have a source_path in metadata"
            )

        source_path = Path(source_path_str).resolve()

        # Security validation: ensure path is within allowed directories
        persistent_path = Path(self.settings.data.paths.persistent_path).resolve()
        temporary_path = Path(self.settings.data.paths.temporary_path).resolve()

        is_in_persistent = False
        is_in_temporary = False

        try:
            source_path.relative_to(persistent_path)
            is_in_persistent = True
        except ValueError:
            pass

        try:
            source_path.relative_to(temporary_path)
            is_in_temporary = True
        except ValueError:
            pass

        if not (is_in_persistent or is_in_temporary):
            logger.warning(
                "Attempted to access file outside allowed paths: %s", source_path
            )
            raise PermissionError(
                f"File path is outside allowed directories: {source_path}"
            )

        # Verify file exists
        if not source_path.exists():
            raise FileNotFoundError(
                f"Source file no longer exists: {source_path}"
            )

        if not source_path.is_file():
            raise ValueError(f"Path is not a file: {source_path}")

        logger.debug("Retrieved file path for doc_id=%s: %s", doc_id, source_path)
        return source_path
