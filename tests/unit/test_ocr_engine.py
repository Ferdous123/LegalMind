"""Unit tests for code.pipeline.ocr_engine.OCREngine.

All LLM/GPU calls are mocked via ModelManager so these tests run fast (< 1 s each)
and require no hardware.
"""

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# test_ocr_engine_imports
# ---------------------------------------------------------------------------

def test_ocr_engine_imports():
    """Verify that OCREngine can be imported cleanly."""
    from code.pipeline.ocr_engine import OCREngine  # noqa: F401
    assert OCREngine is not None


# ---------------------------------------------------------------------------
# test_process_image_returns_page_content
# ---------------------------------------------------------------------------

def test_process_image_returns_page_content(tmp_path):
    """Mock ModelManager so process_image returns a PageContent with known text.

    ocr_engine.py calls ModelManager.instance() and then calls
    _current_model.create_chat_completion(). We mock that chain.
    """
    from code.pipeline.ocr_engine import OCREngine
    from code.pipeline.ingestion import PageContent

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header bytes

    mock_mgr = MagicMock()
    mock_mgr._config = {
        "local_models": {
            "ocr": {"max_output_tokens": 512, "temperature": 0.2, "repeat_penalty": 1.0}
        }
    }
    mock_mgr.load.return_value.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Extracted text here."}}]
    }

    with patch("code.pipeline.ocr_engine.ModelManager") as mock_mgr_cls:
        mock_mgr_cls.instance.return_value = mock_mgr
        engine = OCREngine()
        result = engine.process_image(str(image_path))

    assert isinstance(result, PageContent), "Expected a PageContent instance"
    assert result.text == "Extracted text here.", f"Unexpected text: {result.text!r}"
    assert result.ocr_used is True, "ocr_used should be True for image processing"
    assert result.confidence > 0.0, "Confidence must be positive for non-empty text"


# ---------------------------------------------------------------------------
# test_process_image_error_graceful
# ---------------------------------------------------------------------------

def test_process_image_error_graceful(tmp_path):
    """When ModelManager.load raises, process_image returns PageContent with
    ocr_error set and confidence 0.0 rather than propagating the exception.
    """
    from code.pipeline.ocr_engine import OCREngine
    from code.pipeline.ingestion import PageContent

    image_path = tmp_path / "bad_page.png"
    image_path.write_bytes(b"not a real image")

    mock_mgr = MagicMock()
    mock_mgr.load.side_effect = RuntimeError("GPU out of memory")

    with patch("code.pipeline.ocr_engine.ModelManager") as mock_mgr_cls:
        mock_mgr_cls.instance.return_value = mock_mgr
        engine = OCREngine()
        result = engine.process_image(str(image_path))

    assert isinstance(result, PageContent), "Must return PageContent even on error"
    assert result.confidence == 0.0, "Confidence must be 0.0 on error"
    assert result.ocr_error, "ocr_error must be set on error"
    assert "GPU out of memory" in result.ocr_error, (
        f"Error message not propagated: {result.ocr_error!r}"
    )
    assert result.text == "", "text should be empty on OCR failure"


# ---------------------------------------------------------------------------
# Additional: _estimate_confidence static method
# ---------------------------------------------------------------------------

def test_estimate_confidence_clean_text():
    """Verify _estimate_confidence returns a high value for clean text."""
    from code.pipeline.ocr_engine import OCREngine

    clean = "The defendant filed a motion with the court on January 15 2024."
    score = OCREngine._estimate_confidence(clean)
    assert 0.8 <= score <= 1.0, f"Expected high confidence for clean text, got {score}"


def test_estimate_confidence_empty_text():
    """Verify _estimate_confidence returns a low value for very short text."""
    from code.pipeline.ocr_engine import OCREngine

    score = OCREngine._estimate_confidence("")
    assert score <= 0.2, f"Expected low confidence for empty text, got {score}"


def test_estimate_confidence_illegible_heavy():
    """Verify _estimate_confidence returns a low value when many illegibles appear."""
    from code.pipeline.ocr_engine import OCREngine

    text = " ".join(["[illegible]"] * 12 + ["word"] * 4)
    score = OCREngine._estimate_confidence(text)
    assert score <= 0.5, f"Expected low confidence for illegible-heavy text, got {score}"
