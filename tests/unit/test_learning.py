"""Unit tests for code.learning.correction_store.CorrectionStore.

Uses the isolated_store fixture which patches CORRECTIONS_DIR in both
config.paths AND code.learning.correction_store so each test gets a
completely clean slate with no cross-test leakage. No LLM or GPU calls
are made. Tests run fast (< 1 s each).
"""

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    """Patch CORRECTIONS_DIR inside both config.paths and the correction_store
    module so CorrectionStore always reads/writes to a fresh tmp directory.

    Returns the temp path so tests can inspect the filesystem if needed.
    """
    corrections_dir = tmp_path / "corrections"
    corrections_dir.mkdir(parents=True, exist_ok=True)

    import config.paths as paths
    import code.learning.correction_store as cs

    monkeypatch.setattr(paths, "CORRECTIONS_DIR", corrections_dir)
    monkeypatch.setattr(cs, "CORRECTIONS_DIR", corrections_dir)

    return corrections_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_correction(idx: int = 0, draft_type: str = "case_fact_summary"):
    """Return a Correction instance with unique id & timestamp."""
    from code.learning.correction_store import Correction

    return Correction(
        document_id=f"doc_{idx:03d}",
        draft_type=draft_type,
        field_path="parties.defendant",
        source_ocr_chunk=f"defendant text segment {idx}",
        generated_text=f"Generated: defendant unknown {idx}",
        edited_text=f"Edited: Defendant Name {idx}",
        correction_type="omission",
    )


# ---------------------------------------------------------------------------
# test_correction_store_save_and_retrieve
# ---------------------------------------------------------------------------

def test_correction_store_save_and_retrieve(isolated_store):
    """Saving a correction and calling get_all_active must return that correction.

    Verifies basic persistence round-trip: save → read back → data intact.
    """
    from code.learning.correction_store import CorrectionStore

    store = CorrectionStore()
    corr = _make_correction(idx=1)
    saved_id = store.save_correction(corr)

    active = store.get_all_active("case_fact_summary")

    assert len(active) == 1, f"Expected 1 active correction, got {len(active)}"
    assert active[0].id == saved_id, "Saved ID must match retrieved correction ID"
    assert active[0].field_path == "parties.defendant"
    assert active[0].active is True


# ---------------------------------------------------------------------------
# test_correction_count
# ---------------------------------------------------------------------------

def test_correction_count(isolated_store):
    """Saving 3 corrections must result in get_count() == 3."""
    from code.learning.correction_store import CorrectionStore

    store = CorrectionStore()
    for i in range(3):
        store.save_correction(_make_correction(idx=i))

    count = store.get_count()
    assert count == 3, f"Expected count 3, got {count}"


# ---------------------------------------------------------------------------
# test_archive_correction
# ---------------------------------------------------------------------------

def test_archive_correction(isolated_store):
    """Archiving a correction must make get_all_active return 0 items for
    that draft type (the correction remains on disk but is flagged inactive).
    """
    from code.learning.correction_store import CorrectionStore

    store = CorrectionStore()
    corr = _make_correction(idx=10)
    corr_id = store.save_correction(corr)

    # Confirm it appears as active
    active_before = store.get_all_active("case_fact_summary")
    assert len(active_before) == 1, (
        f"Expected 1 active correction before archive, got {len(active_before)}"
    )

    # Archive it
    store.archive_corrections([corr_id])

    active_after = store.get_all_active("case_fact_summary")
    assert len(active_after) == 0, (
        f"Expected 0 active corrections after archive, got {len(active_after)}"
    )


# ---------------------------------------------------------------------------
# test_get_recent
# ---------------------------------------------------------------------------

def test_get_recent(isolated_store):
    """Saving 5 corrections and calling get_recent(3) must return exactly 3."""
    from code.learning.correction_store import CorrectionStore

    store = CorrectionStore()
    for i in range(5):
        store.save_correction(_make_correction(idx=i))

    recent = store.get_recent(3)
    assert len(recent) == 3, f"Expected 3 recent corrections, got {len(recent)}"


# ---------------------------------------------------------------------------
# test_get_recent_returns_all_when_fewer_than_n
# ---------------------------------------------------------------------------

def test_get_recent_returns_all_when_fewer_than_n(isolated_store):
    """If the store has fewer than n corrections, get_recent(n) returns all of them."""
    from code.learning.correction_store import CorrectionStore

    store = CorrectionStore()
    for i in range(2):
        store.save_correction(_make_correction(idx=i))

    recent = store.get_recent(10)
    assert len(recent) == 2, (
        f"Expected 2 corrections (all of them), got {len(recent)}"
    )


# ---------------------------------------------------------------------------
# test_get_count_per_draft_type
# ---------------------------------------------------------------------------

def test_get_count_per_draft_type(isolated_store):
    """get_count(draft_type) must count only active corrections of that type."""
    from code.learning.correction_store import CorrectionStore

    store = CorrectionStore()
    for i in range(3):
        store.save_correction(_make_correction(idx=i, draft_type="case_fact_summary"))
    for i in range(2):
        store.save_correction(_make_correction(idx=i, draft_type="notice_summary"))

    assert store.get_count("case_fact_summary") == 3, (
        f"Expected 3 case_fact_summary corrections, got {store.get_count('case_fact_summary')}"
    )
    assert store.get_count("notice_summary") == 2, (
        f"Expected 2 notice_summary corrections, got {store.get_count('notice_summary')}"
    )
    assert store.get_count() == 5, (
        f"Expected total 5 corrections, got {store.get_count()}"
    )


# ---------------------------------------------------------------------------
# test_correction_fields_preserved_after_round_trip
# ---------------------------------------------------------------------------

def test_correction_fields_preserved_after_round_trip(isolated_store):
    """All Correction fields must survive JSON serialisation and deserialisation."""
    from code.learning.correction_store import CorrectionStore, Correction

    store = CorrectionStore()
    original = Correction(
        document_id="doc_fields",
        draft_type="title_review_summary",
        field_path="chain_of_title[0].grantor",
        source_ocr_chunk="Grantor: Jane Smith executed the deed",
        generated_text="Grantor: [unknown]",
        edited_text="Grantor: Jane Smith",
        correction_type="omission",
    )
    store.save_correction(original)

    retrieved = store.get_all_active("title_review_summary")
    assert len(retrieved) == 1
    r = retrieved[0]
    assert r.document_id == "doc_fields"
    assert r.field_path == "chain_of_title[0].grantor"
    assert r.source_ocr_chunk == "Grantor: Jane Smith executed the deed"
    assert r.generated_text == "Grantor: [unknown]"
    assert r.edited_text == "Grantor: Jane Smith"
    assert r.correction_type == "omission"
    assert r.active is True


# ---------------------------------------------------------------------------
# test_get_corrections_since
# ---------------------------------------------------------------------------

def test_get_corrections_since(isolated_store):
    """get_corrections_since returns only corrections newer than the given timestamp."""
    from datetime import datetime, timezone
    from code.learning.correction_store import CorrectionStore, Correction

    store = CorrectionStore()

    # Save a first correction with an explicit early timestamp
    early_ts = "2024-01-01T00:00:00+00:00"
    c_early = Correction(
        document_id="doc_early",
        draft_type="case_fact_summary",
        field_path="parties.defendant",
        source_ocr_chunk="early text",
        generated_text="Generated early",
        edited_text="Edited early",
        correction_type="omission",
        timestamp=early_ts,
    )
    store.save_correction(c_early)

    # Use the early timestamp as the cutoff, then save a correction with a later timestamp
    cutoff = early_ts

    late_ts = "2025-06-01T12:00:00+00:00"
    c_late = Correction(
        document_id="doc_late",
        draft_type="case_fact_summary",
        field_path="parties.plaintiff",
        source_ocr_chunk="late text",
        generated_text="Generated late",
        edited_text="Edited late",
        correction_type="omission",
        timestamp=late_ts,
    )
    store.save_correction(c_late)

    since = store.get_corrections_since(cutoff)
    # Only c_late has a strictly greater timestamp
    ids = [c.id for c in since]
    assert c_late.id in ids, "The later correction must appear in get_corrections_since"
    assert c_early.id not in ids, "The earlier (cutoff) correction must NOT appear"
