import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

PDF_CONVERTIBLE_EXTENSIONS = {
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".pptm",
    ".xlsx",
    ".xls",
    ".odt",
    ".odp",
    ".ods",
    ".hwp",
    ".epub",
    ".rtf",
}


def needs_conversion(file_path: Path) -> bool:
    return file_path.suffix.lower() in PDF_CONVERTIBLE_EXTENSIONS


def convert_to_pdf(source_path: Path) -> Path | None:
    """Convert source_path to PDF using LibreOffice CLI.

    Saves the PDF as a sibling file in the same directory with the same stem.
    If the sibling PDF already exists, returns it immediately without re-converting.

    Returns the converted PDF path, or None if conversion was not possible.
    """
    pdf_path = source_path.with_suffix(".pdf")
    if pdf_path.exists():
        logger.debug("Converted PDF already exists, reusing: %s", pdf_path)
        return pdf_path

    logger.info("Converting %s to PDF via LibreOffice", source_path.name)
    try:
        result = subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(source_path.parent),
                str(source_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and pdf_path.exists():
            logger.info("Converted %s → %s", source_path.name, pdf_path.name)
            return pdf_path
        logger.warning(
            "LibreOffice conversion failed for %s (rc=%d): %s",
            source_path.name,
            result.returncode,
            result.stderr.strip(),
        )
        return None
    except FileNotFoundError:
        logger.warning(
            "LibreOffice (soffice) not found on PATH — ingesting %s in original format",
            source_path.name,
        )
        return None
    except subprocess.TimeoutExpired:
        logger.error("LibreOffice conversion timed out for %s", source_path)
        return None
