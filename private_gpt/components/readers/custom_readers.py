"""Image parser.

Contains parsers for image files.

"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from fsspec import AbstractFileSystem
from PIL import Image

from llama_index.core.img_utils import img_2_b64
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, ImageDocument
import ollama

logger = logging.getLogger(__name__)


class CustomImageReader(BaseReader):
    """Image parser.

    Extract text from images using an Ollama vision model.

    """

    def __init__(
        self,
        keep_image: bool = False,
        prompt: Optional[str] = None,
        model: str = "qwen2.5vl:7b",
    ):
        """Parse by sending a request to an Ollama vision model."""
        self._model = model
        self._keep_image = keep_image
        self._prompt = prompt or (
            "First determine whether there is text in the image, if the image is completely a text "
            "then return it as it is in raw without any additional commentary. If its an image describe "
            "it briefly. If its a combination of text and image, return the text as it is in raw and "
            "describe the image briefly."
        )

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file by sending a request to an Ollama vision model."""

        image = Image.open(file)
        image_str: Optional[str] = None
        if self._keep_image:
            image_str = img_2_b64(image)

        response = ollama.chat(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": self._prompt,
                    "images": [file],
                }
            ],
            think=False,
        )
        text_str = response.message.content

        return [
            ImageDocument(
                text=text_str,
                image=image_str,
                image_path=str(file),
                metadata=extra_info or {},
            )
        ]


class ScannedPDFReader(BaseReader):
    """PDF reader that handles both text-based and scanned (image-only) PDFs.

    For pages with embedded text, extracts it directly.
    For scanned pages, renders each page to an image and OCRs via an Ollama vision model.
    Page labels are preserved in metadata for source attribution.
    """

    def __init__(
        self,
        vision_model: str = "qwen2.5vl:7b",
        dpi: int = 150,
    ):
        self._image_reader = CustomImageReader(model=vision_model)
        self._dpi = dpi

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Load PDF, handling both text and scanned pages."""
        import fitz  # PyMuPDF

        pdf = fitz.open(str(file))
        total_pages = len(pdf)
        logger.info("Ingesting PDF file=%s total_pages=%d", file.name, total_pages)

        documents = []
        scanned_pages: List[str] = []

        # Build page label map if the PDF has a custom label tree
        page_labels = self._get_page_labels(pdf)

        for page_num in range(total_pages):
            page = pdf[page_num]
            page_label = page_labels.get(page_num, str(page_num + 1))

            # Try embedded text extraction first (fast path for text PDFs)
            text = page.get_text().strip()
            if text:
                logger.debug("file=%s page=%s has embedded text, skipping OCR", file.name, page_label)
                documents.append(
                    Document(
                        text=text,
                        metadata={**(extra_info or {}), "page_label": page_label},
                    )
                )
                continue

            # Scanned page — render to image and OCR via vision model
            logger.debug("file=%s page=%s is scanned, sending to vision model", file.name, page_label)
            scanned_pages.append(page_label)

            mat = fitz.Matrix(self._dpi / 72, self._dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    pix.save(tmp_path)

                img_docs = self._image_reader.load_data(Path(tmp_path))
                ocr_text = img_docs[0].text if img_docs else ""
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            documents.append(
                Document(
                    text=ocr_text,
                    metadata={**(extra_info or {}), "page_label": page_label},
                )
            )

        pdf.close()

        if scanned_pages:
            logger.info(
                "file=%s contained %d scanned page(s) (OCR applied): %s",
                file.name,
                len(scanned_pages),
                ", ".join(scanned_pages),
            )
        else:
            logger.info("file=%s all %d page(s) had embedded text, no OCR needed", file.name, total_pages)

        return documents

    @staticmethod
    def _get_page_labels(pdf: "fitz.Document") -> dict:  # type: ignore[name-defined]
        """Extract PDF page labels (e.g. roman numerals, custom numbering) if present."""
        labels: dict = {}
        try:
            for page_num in range(len(pdf)):
                label = pdf[page_num].get_label()
                if label:
                    labels[page_num] = label
        except Exception:
            pass
        return labels
