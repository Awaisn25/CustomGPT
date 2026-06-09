import logging
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import chain
from typing import Any

from injector import inject, singleton
from llama_index.core import (
    Document,
    StorageContext,
    SummaryIndex,
)
from llama_index.core.base.response.schema import Response, StreamingResponse
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.types import TokenGen

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    filename: str
    doc_id: str
    summary: str

DEFAULT_SUMMARIZE_PROMPT = (
    "Provide a comprehensive summary of the provided context information. "
    "The summary should cover all the key points and main ideas presented in "
    "the original text, while also condensing the information into a concise "
    "and easy-to-understand format. Please ensure that the summary includes "
    "relevant details and examples that support the main ideas, while avoiding "
    "any unnecessary information or repetition."
)


@singleton
class SummarizeService:
    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        node_store_component: NodeStoreComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.node_store_component = node_store_component
        self.vector_store_component = vector_store_component
        self.embedding_component = embedding_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )

    @staticmethod
    def _filter_ref_docs(
        ref_docs: dict[str, RefDocInfo], context_filter: ContextFilter | None
    ) -> list[RefDocInfo]:
        if context_filter is None or not context_filter.docs_ids:
            return list(ref_docs.values())

        return [
            ref_doc
            for doc_id, ref_doc in ref_docs.items()
            if doc_id in context_filter.docs_ids
        ]

    def _get_llm(self) -> LLM:
        """Return an LLM instance, optionally with the summarize-specific timeout applied."""
        timeout = self.settings.summarize.request_timeout
        if timeout is None:
            return self.llm_component.llm

        llm = self.llm_component.llm
        llm_mode = self.settings.llm.mode
        if llm_mode == "ollama":
            try:
                from llama_index.llms.ollama import Ollama  # type: ignore

                if isinstance(llm, Ollama):
                    # Build a copy with the overridden timeout; Ollama stores it as
                    # request_timeout on the instance.
                    kwargs: dict[str, Any] = dict(
                        model=llm.model,
                        base_url=llm.base_url,
                        temperature=llm.temperature,
                        context_window=llm.context_window,
                        additional_kwargs=llm.additional_kwargs,
                        request_timeout=timeout,
                    )
                    return Ollama(**kwargs)
            except Exception:
                pass
        elif llm_mode in ("openai", "openailike"):
            try:
                llm_copy = llm.copy()  # type: ignore[attr-defined]
                llm_copy.timeout = timeout
                return llm_copy
            except Exception:
                pass

        logger.warning(
            "summarize.request_timeout is set but could not be applied to LLM mode '%s'; "
            "using provider default timeout.",
            llm_mode,
        )
        return llm

    def _run_tree_summarize(
        self,
        nodes: list[BaseNode],
        stream: bool,
        summarize_query: str,
    ) -> str | TokenGen:
        """Run TREE_SUMMARIZE over a list of nodes and return a string or token generator."""
        summary_index = SummaryIndex(
            nodes=nodes,
            storage_context=StorageContext.from_defaults(),
            show_progress=True,
        )
        query_engine = summary_index.as_query_engine(
            llm=self._get_llm(),
            response_mode=ResponseMode.TREE_SUMMARIZE,
            streaming=stream,
            use_async=self.settings.summarize.use_async,
        )
        response = query_engine.query(summarize_query)
        if isinstance(response, Response):
            return response.response or ""
        elif isinstance(response, StreamingResponse):
            return response.response_gen
        else:
            raise TypeError(f"The result is not of a supported type: {type(response)}")

    def _summarize(
        self,
        use_context: bool = False,
        stream: bool = False,
        text: str | None = None,
        instructions: str | None = None,
        context_filter: ContextFilter | None = None,
        prompt: str | None = None,
    ) -> str | TokenGen:

        nodes_to_summarize: list[BaseNode] = []

        # Add text to summarize
        if text:
            text_documents = [Document(text=text)]
            nodes_to_summarize += (
                SentenceSplitter.from_defaults().get_nodes_from_documents(
                    text_documents
                )
            )

        # Add context documents to summarize
        if use_context:
            ref_docs: dict[str, RefDocInfo] | None = (
                self.storage_context.docstore.get_all_ref_doc_info()
            )
            if ref_docs is None:
                raise ValueError("No documents have been ingested yet.")

            filtered_ref_docs = self._filter_ref_docs(ref_docs, context_filter)

            filtered_node_ids = chain.from_iterable(
                [ref_doc.node_ids for ref_doc in filtered_ref_docs]
            )
            filtered_nodes = self.storage_context.docstore.get_nodes(
                node_ids=list(filtered_node_ids),
            )

            nodes_to_summarize += filtered_nodes

        summarize_query = (prompt or DEFAULT_SUMMARIZE_PROMPT) + "\n" + (instructions or "")
        chunk_size = self.settings.summarize.max_nodes_per_chunk

        # Map-reduce path: chunk large node sets so each LLM call stays manageable.
        # Streaming is only applied to the final reduction step.
        if len(nodes_to_summarize) > chunk_size:
            logger.info(
                "Large document: %d nodes exceed max_nodes_per_chunk=%d. "
                "Using chunked map-reduce summarization.",
                len(nodes_to_summarize),
                chunk_size,
            )
            chunks = [
                nodes_to_summarize[i : i + chunk_size]
                for i in range(0, len(nodes_to_summarize), chunk_size)
            ]
            chunk_summaries: list[str] = []
            for idx, chunk in enumerate(chunks):
                logger.info("Summarizing chunk %d/%d (%d nodes)…", idx + 1, len(chunks), len(chunk))
                result = self._run_tree_summarize(chunk, stream=False, summarize_query=summarize_query)
                chunk_summaries.append(result if isinstance(result, str) else "")

            # Combine chunk summaries and do a final pass
            combined_docs = [Document(text=s) for s in chunk_summaries if s]
            final_nodes: list[BaseNode] = SentenceSplitter.from_defaults().get_nodes_from_documents(
                combined_docs
            )
            return self._run_tree_summarize(final_nodes, stream=stream, summarize_query=summarize_query)

        return self._run_tree_summarize(nodes_to_summarize, stream=stream, summarize_query=summarize_query)

    def summarize(
        self,
        use_context: bool = False,
        text: str | None = None,
        instructions: str | None = None,
        context_filter: ContextFilter | None = None,
        prompt: str | None = None,
    ) -> str:
        return self._summarize(
            use_context=use_context,
            stream=False,
            text=text,
            instructions=instructions,
            context_filter=context_filter,
            prompt=prompt,
        )  # type: ignore

    def stream_summarize(
        self,
        use_context: bool = False,
        text: str | None = None,
        instructions: str | None = None,
        context_filter: ContextFilter | None = None,
        prompt: str | None = None,
    ) -> TokenGen:
        return self._summarize(
            use_context=use_context,
            stream=True,
            text=text,
            instructions=instructions,
            context_filter=context_filter,
            prompt=prompt,
        )  # type: ignore

    def summarize_batch(
        self,
        doc_items: list[tuple[str, list[str]]],
        instructions: str | None = None,
        max_workers: int | None = None,
    ) -> Generator[SummaryResult, None, None]:
        """Summarize multiple documents in parallel, yielding results as each worker finishes.

        Args:
            doc_items: List of (filename, doc_ids) pairs. All doc_ids for a filename
                       are used together so multi-page documents are summarized as a whole.
            instructions: Optional custom instruction appended to the default summarize prompt.
            max_workers: Number of parallel workers. Defaults to settings.summarize.max_workers.
        """
        workers = max_workers if max_workers is not None else self.settings.summarize.max_workers

        def _summarize_one(filename: str, doc_ids: list[str]) -> SummaryResult:
            context_filter = ContextFilter(docs_ids=doc_ids)
            summary = self.summarize(
                use_context=True,
                instructions=instructions,
                context_filter=context_filter,
            )
            return SummaryResult(filename=filename, doc_id=doc_ids[0], summary=summary)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_summarize_one, filename, doc_ids): filename
                for filename, doc_ids in doc_items
            }
            for future in as_completed(futures):
                yield future.result()
