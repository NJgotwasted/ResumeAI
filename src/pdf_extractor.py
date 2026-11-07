"""
pdf_extractor.py - High-fidelity PDF Text Extraction using PyMuPDF.

Extracts text, metadata, and structural layout from uploaded resume PDFs.
Handles corrupted files, password-protected files, and empty scanned documents.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import pymupdf  # PyMuPDF (fitz)


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails due to format, corruption, or encryption."""
    pass


class PDFExtractor:
    """Extracts raw and normalized text from PDF documents via PyMuPDF."""

    def __init__(self, clean_whitespace: bool = True):
        self.clean_whitespace = clean_whitespace

    def extract_from_bytes(
        self,
        pdf_bytes: bytes,
        filename: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Extracts text and page metadata from in-memory PDF bytes.
        
        Args:
            pdf_bytes: Raw bytes of the PDF file.
            filename: Optional source file name for logging and tracking.

        Returns:
            Dictionary containing:
                - text: Consolidated clean text string
                - pages: List of individual page texts
                - page_count: Total number of pages
                - char_count: Total extracted characters
                - word_count: Total extracted words
                - is_scanned_likely: True if text yield is unusually low
                - filename: Source filename
        """
        if not pdf_bytes:
            raise PDFExtractionError("Provided PDF byte buffer is empty (0 bytes).")

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise PDFExtractionError(f"Failed to open PDF document: {e}") from e

        return self._process_doc(doc, filename=filename or "uploaded.pdf")

    def extract_from_path(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """Extracts text and metadata from a local PDF file path."""
        path = Path(file_path)
        if not path.exists():
            raise PDFExtractionError(f"File not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise PDFExtractionError(f"File '{path.name}' is not a valid PDF file extension.")

        try:
            doc = pymupdf.open(str(path))
        except Exception as e:
            raise PDFExtractionError(f"Failed to parse PDF file '{path.name}': {e}") from e

        return self._process_doc(doc, filename=path.name)

    def _process_doc(self, doc: pymupdf.Document, filename: str) -> Dict[str, any]:
        """Processes an opened PyMuPDF Document object."""
        try:
            if doc.is_encrypted:
                raise PDFExtractionError(
                    f"PDF '{filename}' is password-protected or encrypted. Please decrypt before uploading."
                )

            page_count = len(doc)
            if page_count == 0:
                raise PDFExtractionError(f"PDF '{filename}' contains 0 pages.")

            pages_text: List[str] = []
            for page_index in range(page_count):
                page = doc.load_page(page_index)
                page_text = page.get_text("text") or ""
                if self.clean_whitespace:
                    page_text = self._clean_text(page_text)
                pages_text.append(page_text)

            consolidated_text = "\n\n--- Page Break ---\n\n".join(
                p for p in pages_text if p.strip()
            ) if page_count > 1 else (pages_text[0] if pages_text else "")

            char_count = len(consolidated_text.strip())
            word_count = len(consolidated_text.strip().split())

            # Detect if file might be a scanned image with no OCR
            is_scanned_likely = (char_count < 50 and page_count > 0)

            return {
                "text": consolidated_text,
                "pages": pages_text,
                "page_count": page_count,
                "char_count": char_count,
                "word_count": word_count,
                "is_scanned_likely": is_scanned_likely,
                "filename": filename,
            }
        finally:
            doc.close()

    def _clean_text(self, text: str) -> str:
        """Normalizes extracted text while preserving structural paragraphs and bullets."""
        # Replace non-breaking spaces and irregular whitespace characters
        text = text.replace('\xa0', ' ').replace('\r\n', '\n').replace('\r', '\n')
        # Replace multiple consecutive spaces on the same line
        text = re.sub(r'[ \t]+', ' ', text)
        # Collapse 3 or more consecutive newlines to 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


def extract_pdf_text(
    file_source: Union[str, Path, bytes, io.BytesIO],
    filename: Optional[str] = None
) -> Dict[str, any]:
    """
    Convenience function to extract text from a file path, raw bytes, or BytesIO object.
    """
    extractor = PDFExtractor()
    if isinstance(file_source, (str, Path)):
        return extractor.extract_from_path(file_source)
    elif isinstance(file_source, bytes):
        return extractor.extract_from_bytes(file_source, filename=filename)
    elif hasattr(file_source, "read"):
        content = file_source.read()
        return extractor.extract_from_bytes(content, filename=filename)
    else:
        raise ValueError(f"Unsupported file source type: {type(file_source)}")

