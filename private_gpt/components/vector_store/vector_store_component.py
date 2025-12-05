import logging
import typing
from collections.abc import Callable

from injector import inject, singleton
from llama_index.core.indices.vector_store import VectorIndexRetriever, VectorStoreIndex
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


def _doc_id_metadata_filter(
    context_filter: ContextFilter | None,
) -> MetadataFilters:
    filters = MetadataFilters(filters=[], condition=FilterCondition.OR)

    if context_filter is not None and context_filter.docs_ids is not None:
        for doc_id in context_filter.docs_ids:
            filters.filters.append(MetadataFilter(key="doc_id", value=doc_id))

    return filters


@singleton
class VectorStoreComponent:
    settings: Settings
    vector_store: BasePydanticVectorStore  # Default vector store for backward compatibility
    _vector_stores: dict[str, BasePydanticVectorStore]  # Cache of vector stores by collection name
    _vector_store_factory: Callable[[str], BasePydanticVectorStore]

    @inject
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._vector_stores = {}
        # Create factory function for creating vector stores
        self._vector_store_factory = self._create_vector_store_factory()
        # Create default vector store for backward compatibility
        self.vector_store = self.get_vector_store(
            settings.vectorstore.default_collection_name
        )

    def _create_vector_store_factory(
        self,
    ) -> Callable[[str], BasePydanticVectorStore]:
        """Create a factory function that returns a vector store for a given collection name."""
        settings = self.settings

        def create_vector_store(collection_name: str) -> BasePydanticVectorStore:
            match settings.vectorstore.database:
                case "postgres":
                    try:
                        from llama_index.vector_stores.postgres import (  # type: ignore
                            PGVectorStore,
                        )
                    except ImportError as e:
                        raise ImportError(
                            "Postgres dependencies not found, install with `poetry install --extras vector-stores-postgres`"
                        ) from e

                    if settings.postgres is None:
                        raise ValueError(
                            "Postgres settings not found. Please provide settings."
                        )

                    # Postgres uses table_name, we can use collection_name as table suffix
                    table_name = f"embeddings_{collection_name}"
                    return typing.cast(
                        BasePydanticVectorStore,
                        PGVectorStore.from_params(
                            **settings.postgres.model_dump(exclude_none=True),
                            table_name=table_name,
                            embed_dim=settings.embedding.embed_dim,
                        ),
                    )

                case "chroma":
                    try:
                        import chromadb  # type: ignore
                        from chromadb.config import (  # type: ignore
                            Settings as ChromaSettings,
                        )

                        from private_gpt.components.vector_store.batched_chroma import (
                            BatchedChromaVectorStore,
                        )
                    except ImportError as e:
                        raise ImportError(
                            "ChromaDB dependencies not found, install with `poetry install --extras vector-stores-chroma`"
                        ) from e

                    chroma_settings = ChromaSettings(anonymized_telemetry=False)
                    chroma_client = chromadb.PersistentClient(
                        path=str((local_data_path / "chroma_db").absolute()),
                        settings=chroma_settings,
                    )
                    chroma_collection = chroma_client.get_or_create_collection(
                        collection_name
                    )

                    return typing.cast(
                        BasePydanticVectorStore,
                        BatchedChromaVectorStore(
                            chroma_client=chroma_client,
                            chroma_collection=chroma_collection,
                        ),
                    )

                case "qdrant":
                    try:
                        from llama_index.vector_stores.qdrant import (  # type: ignore
                            QdrantVectorStore,
                        )
                        from qdrant_client import QdrantClient  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Qdrant dependencies not found, install with `poetry install --extras vector-stores-qdrant`"
                        ) from e

                    if settings.qdrant is None:
                        logger.info(
                            "Qdrant config not found. Using default settings."
                            "Trying to connect to Qdrant at localhost:6333."
                        )
                        client = QdrantClient()
                    else:
                        client = QdrantClient(
                            **settings.qdrant.model_dump(exclude_none=True)
                        )
                    return typing.cast(
                        BasePydanticVectorStore,
                        QdrantVectorStore(
                            client=client,
                            collection_name=collection_name,
                        ),
                    )

                case "milvus":
                    try:
                        from llama_index.vector_stores.milvus import (  # type: ignore
                            MilvusVectorStore,
                        )
                    except ImportError as e:
                        raise ImportError(
                            "Milvus dependencies not found, install with `poetry install --extras vector-stores-milvus`"
                        ) from e

                    if settings.milvus is None:
                        logger.info(
                            f"Milvus config not found. Using default settings.\n"
                            f"Trying to connect to Milvus at local_data/private_gpt/milvus/milvus_local.db "
                            f"with collection '{collection_name}'."
                        )

                        return typing.cast(
                            BasePydanticVectorStore,
                            MilvusVectorStore(
                                dim=settings.embedding.embed_dim,
                                collection_name=collection_name,
                                overwrite=True,
                            ),
                        )

                    else:
                        # Use provided settings but override collection_name
                        return typing.cast(
                            BasePydanticVectorStore,
                            MilvusVectorStore(
                                dim=settings.embedding.embed_dim,
                                uri=settings.milvus.uri,
                                token=settings.milvus.token,
                                collection_name=collection_name,
                                overwrite=settings.milvus.overwrite,
                            ),
                        )

                case "clickhouse":
                    try:
                        from clickhouse_connect import (  # type: ignore
                            get_client,
                        )
                        from llama_index.vector_stores.clickhouse import (  # type: ignore
                            ClickHouseVectorStore,
                        )
                    except ImportError as e:
                        raise ImportError(
                            "ClickHouse dependencies not found, install with `poetry install --extras vector-stores-clickhouse`"
                        ) from e

                    if settings.clickhouse is None:
                        raise ValueError(
                            "ClickHouse settings not found. Please provide settings."
                        )

                    clickhouse_client = get_client(
                        host=settings.clickhouse.host,
                        port=settings.clickhouse.port,
                        username=settings.clickhouse.username,
                        password=settings.clickhouse.password,
                    )
                    # ClickHouse doesn't support collection names directly
                    # We'll use the same client for all collections
                    # Note: This may need adjustment based on ClickHouse implementation
                    return ClickHouseVectorStore(clickhouse_client=clickhouse_client)
                case _:
                    # Should be unreachable
                    # The settings validator should have caught this
                    raise ValueError(
                        f"Vectorstore database {settings.vectorstore.database} not supported"
                    )

        return create_vector_store

    def get_vector_store(
        self, collection_name: str | None = None
    ) -> BasePydanticVectorStore:
        """Get a vector store for the specified collection name.

        Args:
            collection_name: Name of the collection. If None, returns the default vector store.

        Returns:
            Vector store instance for the specified collection.
        """
        if collection_name is None:
            return self.vector_store

        # Return cached vector store if available
        if collection_name in self._vector_stores:
            return self._vector_stores[collection_name]

        # Create new vector store for this collection
        vector_store = self._vector_store_factory(collection_name)
        self._vector_stores[collection_name] = vector_store
        logger.debug(f"Created vector store for collection: {collection_name}")
        return vector_store

    def get_retriever(
        self,
        index: VectorStoreIndex,
        context_filter: ContextFilter | None = None,
        similarity_top_k: int = 2,
    ) -> VectorIndexRetriever:
        # This way we support qdrant (using doc_ids) and the rest (using filters)
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            doc_ids=context_filter.docs_ids if context_filter else None,
            filters=(
                _doc_id_metadata_filter(context_filter)
                if self.settings.vectorstore.database != "qdrant"
                else None
            ),
        )

    def close(self) -> None:
        """Close all vector store connections."""
        # Close default vector store
        if hasattr(self.vector_store, "client") and hasattr(
            self.vector_store.client, "close"
        ):
            try:
                self.vector_store.client.close()
            except Exception:
                pass

        # Close all cached vector stores
        for collection_name, vector_store in self._vector_stores.items():
            if hasattr(vector_store, "client") and hasattr(
                vector_store.client, "close"
            ):
                try:
                    vector_store.client.close()
                except Exception:
                    logger.warning(
                        f"Error closing vector store for collection {collection_name}"
                    )
