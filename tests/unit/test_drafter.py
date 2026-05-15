"""Unit tests for code.generation.drafter.DraftGenerator and
code.generation.templates helper functions.

All LLM, ChromaDB, and retrieval calls are mocked. Tests run fast (< 1 s each).
"""

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# test_drafter_imports
# ---------------------------------------------------------------------------

def test_drafter_imports():
    """Verify DraftGenerator, DraftOutput, and related symbols import cleanly."""
    from code.generation.drafter import DraftGenerator, DraftOutput  # noqa: F401
    from code.generation import get_draft_label, get_supported_types  # noqa: F401
    assert DraftGenerator is not None
    assert DraftOutput is not None


# ---------------------------------------------------------------------------
# test_generate_draft_returns_output
# ---------------------------------------------------------------------------

def test_generate_draft_returns_output(tmp_path):
    """generate_draft should return a DraftOutput with draft_type set correctly.

    Mocks out: InferenceEngine, EvidenceSearcher (returns []),
    ExemplarRetriever (returns []), and GroundingVerifier.
    """
    from code.generation.drafter import DraftGenerator, DraftOutput

    mock_engine = MagicMock()
    mock_engine.generate_text.return_value = "## Case Fact Summary\nDefendant: Marcus Bell."

    mock_searcher = MagicMock()
    mock_searcher.search.return_value = []  # no evidence chunks

    mock_exemplar_retriever = MagicMock()
    mock_exemplar_retriever.get_relevant_exemplars.return_value = []
    mock_exemplar_retriever.format_exemplars_for_prompt.return_value = ""

    mock_grounding = MagicMock()
    mock_grounding.verify_draft_citations.return_value = {}

    with (
        patch("code.generation.drafter.InferenceEngine", return_value=mock_engine),
        patch("code.generation.drafter.EvidenceSearcher", return_value=mock_searcher),
        patch("code.generation.drafter.ExemplarRetriever", return_value=mock_exemplar_retriever),
        patch("code.generation.drafter.GroundingVerifier", return_value=mock_grounding),
        # Redirect PROMPTS_DIR and LEARNED_RULES_YAML to tmp so file loads are safe
        patch("code.generation.drafter.PROMPTS_DIR", tmp_path / "prompts"),
        patch("code.generation.drafter.LEARNED_RULES_YAML", tmp_path / "rules.yaml"),
    ):
        generator = DraftGenerator()
        output = generator.generate_draft(
            document_id="doc_abc",
            draft_type="case_fact_summary",
            full_text="The defendant Marcus Bell filed a motion.",
        )

    assert isinstance(output, DraftOutput), "generate_draft must return a DraftOutput"
    assert output.draft_type == "case_fact_summary", (
        f"Wrong draft_type: {output.draft_type!r}"
    )
    assert output.document_id == "doc_abc"
    assert output.content_markdown != "", "content_markdown should be non-empty"


# ---------------------------------------------------------------------------
# test_generate_draft_populates_evidence_count
# ---------------------------------------------------------------------------

def test_generate_draft_populates_evidence_count(tmp_path):
    """When the searcher returns N evidence chunks, output.evidence_count == N."""
    from code.generation.drafter import DraftGenerator
    from code.retrieval.searcher import EvidenceChunk

    chunks = [
        EvidenceChunk(
            chunk_id=f"c{i}", text=f"evidence text {i}",
            document_id="doc_abc", page_number=1,
            char_start=0, char_end=50, similarity_score=0.8,
        )
        for i in range(4)
    ]

    mock_engine = MagicMock()
    mock_engine.generate_text.return_value = "Draft text."

    mock_searcher = MagicMock()
    mock_searcher.search.return_value = chunks

    mock_exemplar_retriever = MagicMock()
    mock_exemplar_retriever.get_relevant_exemplars.return_value = []
    mock_exemplar_retriever.format_exemplars_for_prompt.return_value = ""

    mock_grounding = MagicMock()
    mock_grounding.verify_draft_citations.return_value = {}

    with (
        patch("code.generation.drafter.InferenceEngine", return_value=mock_engine),
        patch("code.generation.drafter.EvidenceSearcher", return_value=mock_searcher),
        patch("code.generation.drafter.ExemplarRetriever", return_value=mock_exemplar_retriever),
        patch("code.generation.drafter.GroundingVerifier", return_value=mock_grounding),
        patch("code.generation.drafter.PROMPTS_DIR", tmp_path / "prompts"),
        patch("code.generation.drafter.LEARNED_RULES_YAML", tmp_path / "rules.yaml"),
    ):
        generator = DraftGenerator()
        output = generator.generate_draft("doc_abc", "notice_summary")

    assert output.evidence_count == 4, (
        f"Expected evidence_count=4, got {output.evidence_count}"
    )


# ---------------------------------------------------------------------------
# test_get_supported_types
# ---------------------------------------------------------------------------

def test_get_supported_types():
    """get_supported_types() returns a list with at least 4 items, each having
    'id' and 'label' keys.
    """
    from code.generation.templates import get_supported_types

    types = get_supported_types()
    assert isinstance(types, list), "get_supported_types must return a list"
    assert len(types) >= 4, f"Expected >= 4 types, got {len(types)}"
    for item in types:
        assert "id" in item, f"Missing 'id' key in item: {item}"
        assert "label" in item, f"Missing 'label' key in item: {item}"


# ---------------------------------------------------------------------------
# test_get_draft_label
# ---------------------------------------------------------------------------

def test_get_draft_label():
    """get_draft_label('case_fact_summary') should return 'Case Fact Summary'."""
    from code.generation.templates import get_draft_label

    assert get_draft_label("case_fact_summary") == "Case Fact Summary"
    assert get_draft_label("title_review_summary") == "Title Review Summary"
    assert get_draft_label("notice_summary") == "Notice-Related Summary"
    assert get_draft_label("document_checklist") == "Document Checklist"


# ---------------------------------------------------------------------------
# test_get_draft_label_unknown_type
# ---------------------------------------------------------------------------

def test_get_draft_label_unknown_type():
    """get_draft_label for an unknown draft type must not raise and must return
    a non-empty string (the titlecased fallback).
    """
    from code.generation.templates import get_draft_label

    label = get_draft_label("custom_summary_type")
    assert isinstance(label, str) and label, (
        "Unknown draft type must return a non-empty string label"
    )


# ---------------------------------------------------------------------------
# test_draft_output_auto_timestamp
# ---------------------------------------------------------------------------

def test_draft_output_auto_timestamp():
    """DraftOutput.__post_init__ should set generation_timestamp automatically."""
    from code.generation.drafter import DraftOutput

    output = DraftOutput(draft_type="document_checklist", document_id="doc_ts")
    assert output.generation_timestamp, "generation_timestamp must be auto-populated"
    # Should be a valid ISO 8601 string
    from datetime import datetime
    datetime.fromisoformat(output.generation_timestamp)  # raises if invalid
