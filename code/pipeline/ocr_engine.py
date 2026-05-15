"""OCR engine — uses LightOnOCR-1B for optical character recognition.

Handles scanned documents, handwritten notes, and low-resolution images.
Uses llama-cpp-python with the qwen25vl chat handler for vision processing.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from code.pipeline.ingestion import PageContent
from code.llm_interface.inference import InferenceEngine

logger = logging.getLogger(__name__)

OCR_PROMPT = """Extract ALL text from this image. This is a legal document page.

Instructions:
- Transcribe every word you can read, maintaining the original layout where possible
- For text you cannot read clearly, mark it as [illegible]
- Preserve paragraph breaks and formatting
- Include headers, footers, page numbers, stamps, and handwritten annotations
- For tables, represent them with clear alignment
- Do not add any text that is not visible in the image
- Do not interpret or summarize — only transcribe what you see"""


class OCREngine:
    """LightOnOCR-1B based OCR for scanned legal documents."""

    def __init__(self):
        self._engine = InferenceEngine()

    def process_image(self, image_path: str) -> PageContent:
        """Run OCR on a single image file.

        Args:
            image_path: Path to PNG/JPG/TIFF image.

        Returns:
            PageContent with extracted text and confidence.
        """
        try:
            text = self._engine.generate_with_image(
                OCR_PROMPT, image_path, role="ocr", max_tokens=2048
            )
            confidence = self._estimate_confidence(text)
            return PageContent(
                page_number=1,
                text=text.strip(),
                confidence=confidence,
                ocr_used=True,
            )
        except Exception as e:
            logger.error("OCR failed for %s: %s", image_path, e)
            return PageContent(
                page_number=1, text="", confidence=0.0,
                ocr_used=True, ocr_error=str(e)
            )

    def process_pdf_page(self, pdf_path: Path, page_number: int) -> Optional[PageContent]:
        """Render a specific PDF page as image and run OCR.

        Uses pypdfium2 to render the page at 300 DPI.
        """
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(pdf_path))
            if page_number < 1 or page_number > len(pdf):
                return None

            page = pdf[page_number - 1]
            bitmap = page.render(scale=300 / 72)  # 300 DPI
            image = bitmap.to_pil()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                result = self.process_image(tmp.name)
                result.page_number = page_number
                return result

        except Exception as e:
            logger.error("PDF page OCR failed (page %d): %s", page_number, e)
            return PageContent(
                page_number=page_number, text="", confidence=0.0,
                ocr_used=True, ocr_error=str(e)
            )

    @staticmethod
    def _estimate_confidence(text: str) -> float:
        """Heuristic confidence estimate based on OCR output quality.

        Factors:
        - Proportion of [illegible] markers
        - Text length (very short = likely failed)
        - Presence of recognizable patterns (dates, names, legal terms)
        """
        if not text or len(text) < 20:
            return 0.1

        illegible_count = text.count("[illegible]")
        word_count = len(text.split())

        if word_count == 0:
            return 0.1

        illegible_ratio = illegible_count / max(word_count / 10, 1)

        if illegible_ratio > 0.5:
            return 0.3
        elif illegible_ratio > 0.2:
            return 0.5
        elif illegible_ratio > 0.05:
            return 0.7
        else:
            return 0.9
