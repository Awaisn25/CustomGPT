"""Image parser.

Contains parsers for image files.

"""

import re
from pathlib import Path
from typing import Dict, List, Optional, cast, Any
from fsspec import AbstractFileSystem
from io import BytesIO
from PIL import Image

from llama_index.core.img_utils import img_2_b64
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, ImageDocument
import ollama


class CustomImageReader(BaseReader):
    """Image parser.

    Extract text from images using DONUT or pytesseract.

    """

    def __init__(
        self,
        keep_image: bool = False,
        prompt: Optional[str] = None,
    ):
        """Parse by sending a request to Ollama Qwen3-VL model"""
        self._model = "qwen3-vl:2b"
        self._keep_image = keep_image
        self._prompt = prompt or "First determine whether there is text in the image, if the image is completely a text then return it as it is in raw without any additional commentary. If its an image describe it briefly. If its a combination of text and image, return the text as it is in raw and describe the image briefly."
    
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file by sending a request to Ollama Qwen3-VL model"""

        image = Image.open(file)
        image_str: Optional[str] = None
        if self._keep_image:
            image_str = img_2_b64(image)

        # Parse image into text
        response = ollama.chat(
            model=self._model,
            messages=[
                {
                    'role': 'user',
                    'content': self._prompt,
                    'images': [file]
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
