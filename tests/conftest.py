# Fixtures used across all tests
import pytest
from pathlib import Path
import json, tempfile, shutil

@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect all path constants to a temp dir so tests are isolated."""
    # patch config.paths constants
    import config.paths as paths
    monkeypatch.setattr(paths, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(paths, "CORRECTIONS_DIR", tmp_path / "corrections")
    monkeypatch.setattr(paths, "CHROMA_DIR", tmp_path / "chromadb")
    monkeypatch.setattr(paths, "EXEMPLARS_DIR", tmp_path / "exemplars")
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path / "logs")
    paths.ensure_dirs()
    return tmp_path

@pytest.fixture
def processed_document(tmp_data_dir):
    from code.pipeline.ingestion import ProcessedDocument, TextChunk
    doc = ProcessedDocument(
        id="doc_testfixture",
        filename="test.pdf",
        full_text="The defendant, Marcus Bell, filed a motion on 2024-01-15. The plaintiff, Alice Corp, seeks damages of $50,000.",
        page_count=1,
        confidence=0.92,
    )
    doc.chunks = [{"id": "chunk_001", "text": doc.full_text, "document_id": doc.id, "page_number": 1, "char_start": 0, "char_end": len(doc.full_text)}]
    doc.structured_fields = {"parties": {"defendant": "Marcus Bell", "plaintiff": "Alice Corp"}, "date": "2024-01-15", "damages": "$50,000"}
    doc.save()
    return doc

@pytest.fixture
def sample_correction():
    from code.learning.correction_store import Correction
    return Correction(
        id="corr_testfixture",
        document_id="doc_testfixture",
        draft_type="case_fact_summary",
        field_path="parties.defendant",
        source_ocr_chunk="the defendant, Marcus Bell, filed",
        generated_text="Defendant: [Not clearly identified]",
        edited_text="Defendant: Marcus Bell",
        correction_type="omission",
    )

@pytest.fixture
def app_client(tmp_data_dir):
    import httpx
    from webapp.main import app
    # Use httpx sync client for simplicity (the API endpoints accept dicts)
    with httpx.Client(app=app, base_url="http://test") as client:
        yield client
