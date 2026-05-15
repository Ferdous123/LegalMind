"""OCR engine — uses LightOnOCR-1B for optical character recognition.

Per TRACE D-000057: LightOnOCR is an image-only OCR model — no text prompt.
Uses base64 data URI (not file:// URL). Model called directly via ModelManager
with TRACE-aligned message format.
"""

import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional

from code.pipeline.ingestion import PageContent
from code.llm_interface.model_manager import ModelManager

logger = logging.getLogger(__name__)


def _image_to_data_uri(image_path: str) -> str:
    """Convert image file to base64 data URI for multimodal input."""
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "tiff": "image/tiff", "tif": "image/tiff", "bmp": "image/bmp"}.get(suffix, "image/png")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class OCREngine:
    """LightOnOCR-1B based OCR for scanned legal documents.

    Follows TRACE approach: image-only messages, base64 data URI,
    no text prompt (LightOnOCR is trained without text instructions).
    """

    def __init__(self):
        self._mgr = ModelManager.instance()

    def _run_ocr(self, image_path: str) -> str:
        """Load OCR model and run inference on an image.

        Sends image as base64 data URI with empty system message.
        No text prompt — LightOnOCR is an image-only OCR model.
        Per TRACE pattern: load() returns the handle, call it directly.
        """
        handle = self._mgr.load("ocr")

        data_uri = _image_to_data_uri(image_path)
        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ]

        cfg = self._mgr._config.get("local_models", {}).get("ocr", {})
        stop_tokens = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
        response = handle.create_chat_completion(
            messages=messages,
            max_tokens=cfg.get("max_output_tokens", 2048),
            temperature=cfg.get("temperature", 0.2),
            top_p=0.9,
            top_k=0,
            repeat_penalty=cfg.get("repeat_penalty", 1.0),
            stop=stop_tokens,
        )
        return response["choices"][0]["message"]["content"] or ""

    def process_image(self, image_path: str) -> PageContent:
        """Run OCR on a single image file."""
        try:
            text = self._run_ocr(image_path)
            confidence = self._estimate_confidence(text)
            logger.info("OCR complete for %s: %d chars, confidence=%.2f",
                        Path(image_path).name, len(text), confidence)
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
        """Heuristic confidence from OCR output quality."""
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
