"""Unit tests for code.pipeline.ocr_engine.OCREngine.

All LLM/GPU calls are mocked so these tests run fast (< 1 s each)
and require no hardware.
"""

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# test_ocr_engine_imports
# ---------------------------------------------------------------------------

def test_ocr_engine_imports():
    """Verify that OCREngine and its dependencies can be imported cleanly."""
    from code.pipeline.ocr_engine import OCREngine, OCR_PROMPT  # noqa: F401
    assert OCREngine is not None
    assert isinstance(OCR_PROMPT, str)


# ---------------------------------------------------------------------------
# test_process_image_returns_page_content
# ---------------------------------------------------------------------------

def test_process_image_returns_page_content(tmp_path):
    """Mock InferenceEngine.generate_with_image to return a known string.

    Verifies that process_image returns a PageContent whose .text equals
    the stripped mock string and whose .confidence is > 0.
    """
    from code.pipeline.ocr_engine import OCREngine
    from code.pipeline.ingestion import PageContent

    # Create a dummy image file so the path exists (OCREngine passes it to
    # the engine without actually opening it in tests).
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header bytes

    mock_engine = MagicMock()
    mock_engine.generate_with_image.return_value = "Extracted text here."

    with patch(
        "code.pipeline.ocr_engine.InferenceEngine",
        return_value=mock_engine,
    ):
        engine = OCREngine()
        result = engine.process_image(str(image_path))

    assert isinstance(result, PageContent), "Expected a PageContent instance"
    assert result.text == "Extracted text here.", (
        f"Unexpected text: {result.text!r}"
    )
    assert result.ocr_used is True, "ocr_used should be True for image processing"
    assert result.confidence > 0.0, "Confidence must be positive for non-empty text"
    assert result.ocr_error == "", "No error expected on success"


# ---------------------------------------------------------------------------
# test_process_image_error_graceful
# ---------------------------------------------------------------------------

def test_process_image_error_graceful(tmp_path):
    """When InferenceEngine.generate_with_image raises, process_image must
    return a PageContent with ocr_error set and confidence 0.0 rather than
    propagating the exception.
    """
    from code.pipeline.ocr_engine import OCREngine
    from code.pipeline.ingestion import PageContent

    image_path = tmp_path / "bad_page.png"
    image_path.write_bytes(b"not a real image")

    mock_engine = MagicMock()
    mock_engine.generate_with_image.side_effect = RuntimeError("GPU out of memory")

    with patch(
        "code.pipeline.ocr_engine.InferenceEngine",
        return_value=mock_engine,
    ):
        engine = OCREngine()
        result = engine.process_image(str(image_path))

    assert isinstance(result, PageContent), "Must return PageContent even on error"
    assert result.confidence == 0.0, "Confidence must be 0.0 on error"
    assert result.ocr_error != "", "ocr_error must be set on error"
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

    # Construct text where > 50 % of 'words' are [illegible]
    text = " ".join(["[illegible]"] * 12 + ["word"] * 4)
    score = OCREngine._estimate_confidence(text)
    assert score <= 0.5, f"Expected low confidence for illegible-heavy text, got {score}"
