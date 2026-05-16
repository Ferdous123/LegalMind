"""Document ingestion — orchestrates upload, text extraction, OCR routing, structuring.

This is the main entry point for processing a new document. It detects file type,
routes to the appropriate extraction method, and produces a ProcessedDocument.
"""

import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json

from config.paths import UPLOADS_DIR, PROCESSED_DIR, PAGES_DIR

logger = logging.getLogger(__name__)

SPARSE_TEXT_THRESHOLD = 100  # chars per page — below this, route to OCR


@dataclass
class PageContent:
    """Content extracted from a single page."""
    page_number: int
    text: str
    confidence: float = 1.0
    ocr_used: bool = False
    ocr_error: str = ""


@dataclass
class TextChunk:
    """A text chunk ready for retrieval indexing."""
    id: str = ""
    text: str = ""
    document_id: str = ""
    page_number: int = 0
    char_start: int = 0
    char_end: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = f"chunk_{uuid.uuid4().hex[:8]}"


@dataclass
class ProcessedDocument:
    """Complete processed document output."""
    id: str = ""
    filename: str = ""
    upload_timestamp: str = ""
    pages: list = field(default_factory=list)
    structured_fields: dict = field(default_factory=dict)
    chunks: list = field(default_factory=list)
    full_text: str = ""
    ocr_used: bool = False
    confidence: float = 1.0
    page_count: int = 0
    processing_errors: list = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = f"doc_{uuid.uuid4().hex[:12]}"
        if not self.upload_timestamp:
            self.upload_timestamp = datetime.now(timezone.utc).isoformat()

    def save(self) -> Path:
        """Save processed document to data/processed/ as JSON."""
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        filepath = PROCESSED_DIR / f"{self.id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        return filepath

    @classmethod
    def load(cls, doc_id: str) -> Optional["ProcessedDocument"]:
        """Load a processed document from storage."""
        filepath = PROCESSED_DIR / f"{doc_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class DocumentIngester:
    """Orchestrates the full document processing pipeline.

    Usage:
        ingester = DocumentIngester()
        doc = ingester.process("path/to/file.pdf", draft_type="case_fact_summary")
    """

    def __init__(self):
        from code.pipeline.text_extractor import TextExtractor
        from code.pipeline.ocr_engine import OCREngine
        from code.pipeline.structurer import DocumentStructurer
        from code.pipeline.chunker import TextChunker

        self._text_extractor = TextExtractor()
        self._ocr_engine = OCREngine()
        self._structurer = DocumentStructurer()
        self._chunker = TextChunker()

    def process(self, file_path: str, draft_type: str = "case_fact_summary") -> ProcessedDocument:
        """Process a document through the full pipeline.

        Args:
            file_path: Path to the uploaded file (PDF, PNG, JPG, TIFF).
            draft_type: Type of draft to generate (affects structuring).

        Returns:
            ProcessedDocument with all extracted content.
        """
        path = Path(file_path)
        doc = ProcessedDocument(filename=path.name)

        logger.info("Processing document: %s (type: %s)", path.name, draft_type)

        # Step 1: Extract text based on file type
        pages = self._extract_pages(path, doc.id)
        doc.pages = [asdict(p) for p in pages]
        doc.page_count = len(pages)
        doc.ocr_used = any(p.ocr_used for p in pages)

        # Build full_text AND record where each page lands inside it, so the
        # chunker can stamp chunks with their true source page (needed for
        # the citation -> correct page-image link on multi-page PDFs). The
        # join separator below MUST match the one used to build full_text.
        _SEP = "\n\n"
        page_spans: list[tuple[int, int, int]] = []
        parts: list[str] = []
        offset = 0
        for p in pages:
            if not p.text:
                continue
            start = offset
            end = start + len(p.text)
            page_spans.append((p.page_number, start, end))
            parts.append(p.text)
            offset = end + len(_SEP)
        doc.full_text = _SEP.join(parts)

        # Step 2: Calculate overall confidence
        if pages:
            doc.confidence = sum(p.confidence for p in pages) / len(pages)

        # Step 3: Structure extraction
        try:
            doc.structured_fields = self._structurer.extract(doc.full_text, draft_type)
        except Exception as e:
            logger.error("Structuring failed: %s", e)
            doc.processing_errors.append(f"Structuring failed: {e}")
            doc.structured_fields = {}

        # Step 4: Chunk for retrieval (page-aware so citations map to the
        # real source page, not a char-offset estimate)
        chunks = self._chunker.chunk(doc.full_text, doc.id, page_spans=page_spans)
        doc.chunks = [asdict(c) for c in chunks]

        # Step 5: Save
        doc.save()
        logger.info("Document processed: %s (%d pages, confidence: %.2f)",
                    doc.id, doc.page_count, doc.confidence)

        return doc

    def _extract_pages(self, path: Path, doc_id: str = "") -> list[PageContent]:
        """Route document to appropriate extraction method."""
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._process_pdf(path, doc_id)
        elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
            return self._process_image(path)
        elif suffix in (".txt", ".md", ".text"):
            return self._process_text_file(path)
        else:
            logger.warning("Unsupported file type: %s", suffix)
            return [PageContent(page_number=1, text="", confidence=0.0,
                               ocr_error=f"Unsupported file type: {suffix}")]

    def _process_text_file(self, path: Path) -> list[PageContent]:
        """Process plain text file — split into ~3000-char pages."""
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error("Text file read failed for %s: %s", path, e)
            return [PageContent(page_number=1, text="", confidence=0.0,
                               ocr_error=f"Text read failed: {e}")]

        chunk_size = 3000
        pages = []
        for i in range(0, max(1, len(raw)), chunk_size):
            chunk = raw[i : i + chunk_size].strip()
            pages.append(PageContent(
                page_number=len(pages) + 1,
                text=chunk,
                confidence=1.0,
                ocr_used=False,
            ))
        logger.info("Text file %s split into %d page(s)", path.name, len(pages))
        return pages or [PageContent(page_number=1, text="", confidence=1.0)]

    def _process_pdf(self, path: Path, doc_id: str = "") -> list[PageContent]:
        """Process PDF: try text extraction first, fall back to OCR if sparse.

        Also renders each page to JPEG at ~150 DPI for human audit.
        Images saved to data/pages/{doc_id}/page_{n:03d}.jpg.
        """
        pages = self._text_extractor.extract_from_pdf(path)

        # OCR fallback for sparse pages
        for i, page in enumerate(pages):
            if len(page.text.strip()) < SPARSE_TEXT_THRESHOLD:
                logger.info("Page %d sparse text (%d chars), routing to OCR",
                           page.page_number, len(page.text))
                ocr_result = self._ocr_engine.process_pdf_page(path, page.page_number)
                if ocr_result and len(ocr_result.text) > len(page.text):
                    pages[i] = ocr_result

        # Render page images for audit panel
        if doc_id:
            self._save_page_images(path, doc_id, len(pages))

        return pages

    @staticmethod
    def _save_page_images(path: Path, doc_id: str, page_count: int) -> None:
        """Render PDF pages to JPEG for the audit/evidence view page feature."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not installed — page images unavailable")
            return

        out_dir = PAGES_DIR / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            with fitz.open(str(path)) as pdf:
                n_pages = len(pdf)
                mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
                for page_idx in range(n_pages):
                    img_path = out_dir / f"page_{page_idx + 1:03d}.jpg"
                    if img_path.exists():
                        continue
                    pix = pdf[page_idx].get_pixmap(matrix=mat, alpha=False)
                    pix.save(str(img_path), output="jpeg", jpg_quality=85)
            logger.info("Saved %d page images for %s", n_pages, doc_id)
        except Exception as e:
            logger.warning("Page image rendering failed for %s: %s", doc_id, e)

    def _process_image(self, path: Path) -> list[PageContent]:
        """Process image file: always use OCR."""
        result = self._ocr_engine.process_image(str(path))
        return [result] if result else [
            PageContent(page_number=1, text="", confidence=0.0, ocr_used=True,
                       ocr_error="OCR returned empty result")
        ]
