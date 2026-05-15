"""Text extraction from PDFs using pdfplumber.

Handles well-formatted PDFs where text is embedded (not scanned).
For scanned/image PDFs, the ingestion layer routes to OCR instead.
"""

import logging
from pathlib import Path

from code.pipeline.ingestion import PageContent

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extracts text from PDFs using pdfplumber."""

    def extract_from_pdf(self, pdf_path: Path) -> list[PageContent]:
        """Extract text from all pages of a PDF.

        Returns list of PageContent with high confidence (text-based extraction).
        Pages with very little text will be flagged for OCR by the ingester.
        """
        import pdfplumber

        pages = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    pages.append(PageContent(
                        page_number=i,
                        text=text.strip(),
                        confidence=0.95 if text.strip() else 0.0,
                        ocr_used=False,
                    ))
        except Exception as e:
            logger.error("PDF text extraction failed for %s: %s", pdf_path, e)
            pages = [PageContent(
                page_number=1, text="", confidence=0.0,
                ocr_error=f"PDF extraction failed: {e}"
            )]

        return pages

    def extract_metadata(self, pdf_path: Path) -> dict:
        """Extract PDF metadata (title, author, creation date, etc.)."""
        import pdfplumber

        try:
            with pdfplumber.open(pdf_path) as pdf:
                meta = pdf.metadata or {}
                return {
                    "title": meta.get("Title", ""),
                    "author": meta.get("Author", ""),
                    "creation_date": meta.get("CreationDate", ""),
                    "page_count": len(pdf.pages),
                    "producer": meta.get("Producer", ""),
                }
        except Exception as e:
            logger.error("Metadata extraction failed: %s", e)
            return {}
