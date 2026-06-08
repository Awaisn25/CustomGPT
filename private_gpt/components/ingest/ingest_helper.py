import logging
from pathlib import Path
from typing import Optional, Dict, List

from llama_index.core.readers import StringIterableReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.json import JSONReader
from llama_index.core.schema import Document

from private_gpt.components.readers.custom_readers import CustomImageReader, ScannedPDFReader

logger = logging.getLogger(__name__)


def _try_loading_included_file_formats() -> dict[str, type[BaseReader]]:
    try:
        from llama_index.readers.file.docs import (  # type: ignore
            DocxReader,
            HWPReader,
            PDFReader,
        )
        from llama_index.readers.file.epub import EpubReader  # type: ignore
        from llama_index.readers.file.image import ImageReader  # type: ignore
        from llama_index.readers.file.ipynb import IPYNBReader  # type: ignore
        from llama_index.readers.file.markdown import MarkdownReader  # type: ignore
        from llama_index.readers.file.mbox import MboxReader  # type: ignore
        from llama_index.readers.file.slides import PptxReader  # type: ignore
        from llama_index.readers.file.tabular import PandasCSVReader  # type: ignore
        from llama_index.readers.file.video_audio import (  # type: ignore
            VideoAudioReader,
        )
    except ImportError as e:
        raise ImportError("`llama-index-readers-file` package not found") from e

    default_file_reader_cls: dict[str, type[BaseReader]] = {
        ".hwp": HWPReader,
        ".pdf": ScannedPDFReader,
        ".docx": DocxReader,
        ".pptx": PptxReader,
        ".ppt": PptxReader,
        ".pptm": PptxReader,
        ".jpg": CustomImageReader,
        ".png": CustomImageReader,
        ".jpeg": CustomImageReader,
        ".mp3": VideoAudioReader,
        ".mp4": VideoAudioReader,
        ".csv": PandasCSVReader,
        ".epub": EpubReader,
        ".md": MarkdownReader,
        ".mbox": MboxReader,
        ".ipynb": IPYNBReader,
    }
    return default_file_reader_cls


# Patching the default file reader to support other file types
FILE_READER_CLS = _try_loading_included_file_formats()
FILE_READER_CLS.update(
    {
        ".json": JSONReader,
    }
)


class IngestionHelper:
    """Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """

    @staticmethod
    def transform_file_into_documents(
        file_name: str,
        file_data: Path,
        collection_name: str | None = None,
        is_temporary: bool | None = None,
    ) -> list[Document]:
        """Transform a file into a list of documents with enhanced metadata.

        Args:
            file_name: Name of the file
            file_data: Path to the file
            collection_name: Optional collection name for metadata
            is_temporary: Optional flag indicating if this is a temporary document

        Returns:
            List of documents with metadata
        """
        documents = IngestionHelper._load_file_to_documents(file_name, file_data)
        for document in documents:
            document.metadata["file_name"] = file_name
            # Add enhanced metadata
            document.metadata["source_path"] = str(file_data.resolve())
            if collection_name is not None:
                document.metadata["collection_name"] = collection_name
            if is_temporary is not None:
                document.metadata["is_temporary"] = is_temporary
            # Remove surrogates and replace with the unicode replacement character to avoid encoding errors
            document.text = document.text.encode("utf-8", errors="replace").decode("utf-8")
        IngestionHelper._exclude_metadata(documents)
        return documents

    @staticmethod
    def _load_file_to_documents(file_name: str, file_data: Path) -> list[Document]:
        logger.debug("Transforming file_name=%s into documents", file_name)
        extension = Path(file_name).suffix
        reader_cls = FILE_READER_CLS.get(extension)
        if reader_cls is None:
            logger.debug(
                "No reader found for extension=%s, using default string reader",
                extension,
            )
            # Read as a plain text
            string_reader = StringIterableReader()
            return string_reader.load_data([file_data.read_text()])

        logger.debug("Specific reader found for extension=%s", extension)
        documents = reader_cls().load_data(file_data)

        # Sanitize NUL bytes in text which can't be stored in Postgres
        for i in range(len(documents)):
            documents[i].text = documents[i].text.replace("\u0000", "")

        return documents

    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> None:
        logger.debug("Excluding metadata from count=%s documents", len(documents))
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = [
                "doc_id",
                "source_path",
                "collection_name",
                "is_temporary",
            ]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = [
                "file_name",
                "doc_id",
                "page_label",
                "source_path",
                "collection_name",
                "is_temporary",
            ]
