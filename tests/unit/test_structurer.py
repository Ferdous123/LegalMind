"""Unit tests for code.pipeline.structurer.DocumentStructurer.

All LLM calls are mocked. Tests run fast (< 1 s each).
"""

import json
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# test_structurer_imports
# ---------------------------------------------------------------------------

def test_structurer_imports():
    """Verify DocumentStructurer, FIELD_SCHEMAS, and EXTRACTION_PROMPT import cleanly."""
    from code.pipeline.structurer import (  # noqa: F401
        DocumentStructurer,
        FIELD_SCHEMAS,
        EXTRACTION_PROMPT,
    )
    assert DocumentStructurer is not None
    assert "case_fact_summary" in FIELD_SCHEMAS
    assert len(FIELD_SCHEMAS) >= 4, "Expected at least 4 draft-type schemas"


# ---------------------------------------------------------------------------
# test_extract_returns_dict
# ---------------------------------------------------------------------------

def test_extract_returns_dict():
    """When InferenceEngine.generate_structured returns valid JSON-compatible dict,
    DocumentStructurer.extract should return that dict.
    """
    from code.pipeline.structurer import DocumentStructurer

    expected_result = {"parties": {"defendant": "John Doe"}}

    mock_engine = MagicMock()
    # generate_structured returns a dict (already parsed by InferenceEngine)
    mock_engine.generate_structured.return_value = expected_result

    with patch(
        "code.pipeline.structurer.InferenceEngine",
        return_value=mock_engine,
    ):
        structurer = DocumentStructurer()
        result = structurer.extract(
            text="The defendant John Doe filed a motion.",
            draft_type="case_fact_summary",
            use_cascade=False,
        )

    assert isinstance(result, dict), "extract() must return a dict"
    assert result == expected_result, (
        f"Returned dict does not match mock output.\nGot: {result}"
    )


# ---------------------------------------------------------------------------
# test_extract_invalid_json_returns_empty_or_schema
# ---------------------------------------------------------------------------

def test_extract_invalid_json_returns_empty():
    """When InferenceEngine returns a _parse_error dict (i.e. bad JSON from LLM),
    DocumentStructurer.extract must not raise and must return a dict (the schema
    fallback, not a bare exception).
    """
    from code.pipeline.structurer import DocumentStructurer, FIELD_SCHEMAS

    # Simulate what InferenceEngine._parse_json_response returns for bad JSON
    parse_error_result = {"_parse_error": True, "_raw_response": "not json at all"}

    mock_engine = MagicMock()
    mock_engine.generate_structured.return_value = parse_error_result

    with patch(
        "code.pipeline.structurer.InferenceEngine",
        return_value=mock_engine,
    ):
        structurer = DocumentStructurer()
        result = structurer.extract(
            text="Garbled text with no structure.",
            draft_type="case_fact_summary",
            use_cascade=False,
        )

    # Must not raise; must return a dict (the schema fallback)
    assert isinstance(result, dict), (
        "extract() must return a dict even when JSON parsing fails"
    )
    # The fallback is the schema template — it must not be None or raise
    assert result is not None


# ---------------------------------------------------------------------------
# test_extract_text_truncated_at_max_chars
# ---------------------------------------------------------------------------

def test_extract_text_truncated_at_max_chars():
    """Documents longer than 8000 characters should be silently truncated;
    the structurer must still call generate_structured and return a dict.
    """
    from code.pipeline.structurer import DocumentStructurer

    long_text = "Legal text. " * 1000  # ~12 000 chars

    mock_engine = MagicMock()
    mock_engine.generate_structured.return_value = {"parties": {}}

    with patch(
        "code.pipeline.structurer.InferenceEngine",
        return_value=mock_engine,
    ):
        structurer = DocumentStructurer()
        result = structurer.extract(long_text, draft_type="notice_summary", use_cascade=False)

    assert isinstance(result, dict)
    # Verify generate_structured was indeed called (with the truncated prompt)
    mock_engine.generate_structured.assert_called_once()
    call_args = mock_engine.generate_structured.call_args
    prompt_arg = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
    assert "[Document truncated for processing]" in prompt_arg, (
        "Truncation marker should appear in the prompt for long documents"
    )


# ---------------------------------------------------------------------------
# test_get_schema_returns_dict_for_known_type
# ---------------------------------------------------------------------------

def test_get_schema_returns_dict_for_known_type():
    """get_schema returns the correct schema dict for known draft types."""
    from code.pipeline.structurer import DocumentStructurer, FIELD_SCHEMAS

    mock_engine = MagicMock()
    with patch("code.pipeline.structurer.InferenceEngine", return_value=mock_engine):
        structurer = DocumentStructurer()

    for draft_type in FIELD_SCHEMAS:
        schema = structurer.get_schema(draft_type)
        assert isinstance(schema, dict), f"Schema for {draft_type!r} is not a dict"
        assert schema == FIELD_SCHEMAS[draft_type]


# ---------------------------------------------------------------------------
# test_supported_draft_types
# ---------------------------------------------------------------------------

def test_supported_draft_types():
    """supported_draft_types() returns a list of at least 4 strings."""
    from code.pipeline.structurer import DocumentStructurer

    types = DocumentStructurer.supported_draft_types()
    assert isinstance(types, list)
    assert len(types) >= 4
    assert "case_fact_summary" in types
    assert "title_review_summary" in types
