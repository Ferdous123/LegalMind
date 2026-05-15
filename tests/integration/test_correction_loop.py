"""Integration tests for the correction → pattern extraction feedback loop.

These tests exercise the CorrectionStore + PatternExtractor should_trigger()
logic and (optionally) the ExemplarRetriever with sentence_transformers.

All LLM/GPU calls inside PatternExtractor (InferenceEngine) are mocked.
File I/O is isolated via the isolated_dirs fixture that patches CORRECTIONS_DIR
and LEARNED_RULES_YAML in every affected module.

Mark: pytest.mark.integration
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_dirs(tmp_path, monkeypatch):
    """Patch CORRECTIONS_DIR and LEARNED_RULES_YAML in all modules that bind
    these names at import time, giving each test a clean filesystem state.

    Returns a dict with 'corrections_dir' and 'rules_yaml' paths.
    """
    corrections_dir = tmp_path / "corrections"
    corrections_dir.mkdir(parents=True, exist_ok=True)
    rules_yaml = tmp_path / "learned_rules.yaml"

    import config.paths as paths
    import code.learning.correction_store as cs
    import code.learning.pattern_extractor as pe

    monkeypatch.setattr(paths, "CORRECTIONS_DIR", corrections_dir)
    monkeypatch.setattr(paths, "LEARNED_RULES_YAML", rules_yaml)
    monkeypatch.setattr(cs, "CORRECTIONS_DIR", corrections_dir)
    monkeypatch.setattr(pe, "LEARNED_RULES_YAML", rules_yaml)

    return {"corrections_dir": corrections_dir, "rules_yaml": rules_yaml}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_correction(idx: int, draft_type: str = "case_fact_summary"):
    """Create a Correction with a unique index."""
    from code.learning.correction_store import Correction

    return Correction(
        document_id=f"doc_{idx:03d}",
        draft_type=draft_type,
        field_path="parties.defendant",
        source_ocr_chunk=f"The defendant filed a motion on date {idx}",
        generated_text=f"Defendant: [unknown {idx}]",
        edited_text=f"Defendant: Named Party {idx}",
        correction_type="omission",
    )


def _make_pattern_extractor():
    """Create a PatternExtractor with InferenceEngine mocked out."""
    mock_engine = MagicMock()
    mock_engine.generate_text.return_value = "Always identify defendants by full name."
    mock_engine.embed_batch.return_value = [[0.1] * 128]

    with patch("code.learning.pattern_extractor.InferenceEngine", return_value=mock_engine):
        from code.learning.pattern_extractor import PatternExtractor
        extractor = PatternExtractor()

    return extractor


# ---------------------------------------------------------------------------
# test_correction_saved_triggers_extraction_when_count_met
# ---------------------------------------------------------------------------

def test_correction_saved_triggers_extraction_when_count_met(isolated_dirs):
    """Saving >= 20 corrections must make PatternExtractor.should_trigger() return True.

    PatternExtractor.should_trigger() checks whether current_count >=
    last_trigger_count + TRIGGER_INTERVAL. With no prior extraction,
    last_trigger_count == 0, so >= 20 corrections triggers it.
    """
    from code.learning.correction_store import CorrectionStore
    from code.learning.pattern_extractor import TRIGGER_INTERVAL

    store = CorrectionStore()
    for i in range(TRIGGER_INTERVAL):
        store.save_correction(_make_correction(idx=i))

    assert store.get_count() == TRIGGER_INTERVAL, (
        f"Expected {TRIGGER_INTERVAL} corrections, got {store.get_count()}"
    )

    extractor = _make_pattern_extractor()
    assert extractor.should_trigger() is True, (
        f"should_trigger() must return True after {TRIGGER_INTERVAL} corrections"
    )


# ---------------------------------------------------------------------------
# test_should_trigger_returns_false_below_threshold
# ---------------------------------------------------------------------------

def test_should_trigger_returns_false_below_threshold(isolated_dirs):
    """PatternExtractor.should_trigger() must return False when fewer than
    TRIGGER_INTERVAL corrections exist.
    """
    from code.learning.correction_store import CorrectionStore
    from code.learning.pattern_extractor import TRIGGER_INTERVAL

    store = CorrectionStore()
    for i in range(TRIGGER_INTERVAL - 1):
        store.save_correction(_make_correction(idx=i))

    extractor = _make_pattern_extractor()
    assert extractor.should_trigger() is False, (
        "should_trigger() must be False when count < TRIGGER_INTERVAL"
    )


# ---------------------------------------------------------------------------
# test_multiple_draft_types_counted_separately
# ---------------------------------------------------------------------------

def test_multiple_draft_types_counted_separately(isolated_dirs):
    """Corrections across different draft_types are stored in separate JSONL files.

    get_count() with no argument sums all files; get_count(draft_type) counts
    only that type.
    """
    from code.learning.correction_store import CorrectionStore

    store = CorrectionStore()
    for i in range(5):
        store.save_correction(_make_correction(idx=i, draft_type="case_fact_summary"))
    for i in range(3):
        store.save_correction(_make_correction(idx=i, draft_type="notice_summary"))

    assert store.get_count("case_fact_summary") == 5, (
        f"Expected 5, got {store.get_count('case_fact_summary')}"
    )
    assert store.get_count("notice_summary") == 3, (
        f"Expected 3, got {store.get_count('notice_summary')}"
    )
    assert store.get_count() == 8, (
        f"Expected total 8, got {store.get_count()}"
    )


# ---------------------------------------------------------------------------
# test_exemplar_retrieval_after_save
# ---------------------------------------------------------------------------

def test_exemplar_retrieval_after_save(isolated_dirs):
    """Save a correction, index it via a mocked ChromaDB/ModelManager, then
    retrieve it by similar source text.

    The test uses a deterministic mock ModelManager (no real embeddings model
    needed) and a mock ChromaDB collection that returns the saved correction's id.
    """
    from code.learning.correction_store import CorrectionStore, Correction
    from code.learning.exemplar_retriever import ExemplarRetriever

    # Also patch the correction_store module-level reference used by ExemplarRetriever
    import code.learning.exemplar_retriever as er_mod
    import code.learning.correction_store as cs_mod

    store = CorrectionStore()
    corr = Correction(
        document_id="doc_exemplar_test",
        draft_type="case_fact_summary",
        field_path="parties.defendant",
        source_ocr_chunk="the defendant Marcus Bell filed a motion",
        generated_text="Defendant: [unknown]",
        edited_text="Defendant: Marcus Bell",
        correction_type="omission",
    )
    store.save_correction(corr)

    # Deterministic mock embeddings
    mock_mgr = MagicMock()
    unit_vec = [1.0] + [0.0] * 127
    mock_mgr.embed.return_value = [unit_vec]

    # Mock ChromaDB so we control the query result
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_collection.upsert.return_value = None
    mock_collection.query.return_value = {
        "ids": [[corr.id]],
        "metadatas": [[{
            "draft_type": "case_fact_summary",
            "field_path": "parties.defendant",
            "correction_type": "omission",
            "active": True,
        }]],
        "distances": [[0.05]],
        "documents": [["the defendant Marcus Bell filed a motion"]],
    }

    with (
        patch("code.learning.exemplar_retriever.chromadb.PersistentClient",
              return_value=mock_client),
        patch("code.learning.exemplar_retriever.ModelManager") as mock_cls,
    ):
        mock_cls.instance.return_value = mock_mgr
        retriever = ExemplarRetriever()
        retriever.index_correction(corr)

        exemplars = retriever.get_relevant_exemplars(
            source_text="the defendant Marcus Bell filed a motion",
            draft_type="case_fact_summary",
            k=3,
        )

    assert len(exemplars) >= 1, (
        "At least one exemplar should be returned for matching source text"
    )
    assert exemplars[0].id == corr.id, (
        f"Expected exemplar id={corr.id!r}, got {exemplars[0].id!r}"
    )
    assert exemplars[0].edited_text == "Defendant: Marcus Bell"


# ---------------------------------------------------------------------------
# test_format_exemplars_for_prompt
# ---------------------------------------------------------------------------

def test_format_exemplars_for_prompt(isolated_dirs):
    """format_exemplars_for_prompt should produce a non-empty string that
    includes the source_ocr_chunk and edited_text of each exemplar.
    """
    from code.learning.correction_store import Correction
    from code.learning.exemplar_retriever import ExemplarRetriever

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    with (
        patch("code.learning.exemplar_retriever.chromadb.PersistentClient",
              return_value=mock_client),
        patch("code.learning.exemplar_retriever.ModelManager"),
    ):
        retriever = ExemplarRetriever()

    exemplar = Correction(
        document_id="doc_fmt",
        draft_type="case_fact_summary",
        field_path="parties.plaintiff",
        source_ocr_chunk="the plaintiff Alice Corp seeks damages",
        generated_text="Plaintiff: [unknown]",
        edited_text="Plaintiff: Alice Corp",
        correction_type="omission",
    )

    formatted = retriever.format_exemplars_for_prompt([exemplar])

    assert isinstance(formatted, str) and formatted, (
        "format_exemplars_for_prompt must return a non-empty string"
    )
    assert "Alice Corp" in formatted, (
        "Formatted exemplar must contain part of the correction content"
    )
    assert "Example 1" in formatted, (
        "Formatted exemplar must include 'Example 1' header"
    )


# ---------------------------------------------------------------------------
# test_format_exemplars_empty_list
# ---------------------------------------------------------------------------

def test_format_exemplars_empty_list(isolated_dirs):
    """format_exemplars_for_prompt with an empty list must return an empty string."""
    from code.learning.exemplar_retriever import ExemplarRetriever

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    with (
        patch("code.learning.exemplar_retriever.chromadb.PersistentClient",
              return_value=mock_client),
        patch("code.learning.exemplar_retriever.ModelManager"),
    ):
        retriever = ExemplarRetriever()

    result = retriever.format_exemplars_for_prompt([])
    assert result == "", (
        f"Expected empty string for empty exemplar list, got {result!r}"
    )


# ---------------------------------------------------------------------------
# test_trigger_not_repeated_before_new_corrections
# ---------------------------------------------------------------------------

def test_trigger_not_repeated_before_new_corrections(isolated_dirs):
    """After the trigger count is recorded, should_trigger() must return False
    until TRIGGER_INTERVAL more corrections are added.

    Simulates a completed extraction cycle by directly calling
    _update_last_trigger_count() after saving 20 corrections.
    """
    from code.learning.correction_store import CorrectionStore
    from code.learning.pattern_extractor import PatternExtractor, TRIGGER_INTERVAL

    store = CorrectionStore()
    for i in range(TRIGGER_INTERVAL):
        store.save_correction(_make_correction(idx=i))

    extractor = _make_pattern_extractor()
    assert extractor.should_trigger() is True

    # Simulate the extractor recording that it has run
    extractor._update_last_trigger_count()
    # Re-create extractor to reload the persisted trigger count
    extractor2 = _make_pattern_extractor()

    assert extractor2.should_trigger() is False, (
        "should_trigger() must be False immediately after updating trigger count"
    )

    # Add TRIGGER_INTERVAL more corrections — now it should trigger again
    for i in range(TRIGGER_INTERVAL, TRIGGER_INTERVAL * 2):
        store.save_correction(_make_correction(idx=i))

    extractor3 = _make_pattern_extractor()
    assert extractor3.should_trigger() is True, (
        "should_trigger() must be True again after adding another batch of corrections"
    )
