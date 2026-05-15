"""Integration test — full document ingestion pipeline.

Mocks the LLM (OCREngine and DocumentStructurer) but uses real filesystem I/O
and real TextChunker logic.

Mark: pytest.mark.integration
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pipeline_dirs(tmp_path, monkeypatch):
    """Patch PROCESSED_DIR, UPLOADS_DIR in both config.paths and the ingestion
    module so DocumentIngester.save() writes to the temp directory.

    Returns tmp_path for path construction in tests.
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    import config.paths as paths
    import code.pipeline.ingestion as ing

    monkeypatch.setattr(paths, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(paths, "UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr(ing, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(ing, "UPLOADS_DIR", uploads_dir)

    return tmp_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_text_pdf(path: Path, text: str) -> Path:
    """Create a minimal but valid text PDF at *path* using only stdlib."""
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from tests.fixtures.create_sample_pdf import create_minimal_pdf

    return Path(create_minimal_pdf(str(path), text))


# ---------------------------------------------------------------------------
# test_document_ingestion_text_pdf
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    pytest.importorskip("pdfplumber", reason="pdfplumber not installed") is None,
    reason="pdfplumber not available",
)
def test_document_ingestion_text_pdf(pipeline_dirs):
    """End-to-end document ingestion with mocked OCR and structurer.

    Steps:
    1. Create a minimal text PDF.
    2. Mock OCREngine so no GPU is required.
    3. Mock DocumentStructurer to return a canned structured dict.
    4. Run DocumentIngester.process().
    5. Assert that a ProcessedDocument is saved to disk.
    6. Assert that chunks is non-empty.
    """
    from code.pipeline.ingestion import DocumentIngester, PageContent

    # Create the test PDF in the patched uploads dir
    pdf_path = pipeline_dirs / "uploads" / "test_case.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    legal_text = (
        "CASE NO. 2024-CV-5678\n\n"
        "PLAINTIFF: Jane Roe, individually.\n"
        "DEFENDANT: BigCo LLC, a Delaware limited liability company.\n\n"
        "COMPLAINT FOR BREACH OF CONTRACT\n\n"
        "Filed: February 10, 2024.\n\n"
        "FACTS:\n"
        "1. On March 1, 2023, the parties entered into an agreement.\n"
        "2. The total fee agreed was $75,000 payable in two installments.\n"
        "3. Plaintiff completed all deliverables by August 31, 2023.\n"
        "4. Defendant failed to pay the second installment of $37,500.\n\n"
        "RELIEF SOUGHT:\n"
        "Plaintiff seeks $37,500 plus interest and attorney fees."
    )
    _create_text_pdf(pdf_path, legal_text)

    # Mock OCR engine — returns PageContent with the legal text directly
    mock_ocr_page_content = PageContent(
        page_number=1,
        text=legal_text,
        confidence=0.95,
        ocr_used=False,
    )
    mock_ocr_engine = MagicMock()
    mock_ocr_engine.process_pdf_page.return_value = mock_ocr_page_content

    # Mock structurer — returns a canned dict without calling the LLM
    mock_structurer = MagicMock()
    mock_structurer.extract.return_value = {
        "parties": {
            "plaintiffs": [{"name": "Jane Roe", "source_span": "PLAINTIFF: Jane Roe"}],
            "defendants": [{"name": "BigCo LLC", "source_span": "DEFENDANT: BigCo LLC"}],
        },
        "key_dates": [
            {"event": "Filing", "date": "2024-02-10",
             "source_span": "Filed: February 10, 2024"},
        ],
    }

    # DocumentIngester.__init__ does `from code.pipeline.ocr_engine import OCREngine`
    # and `from code.pipeline.structurer import DocumentStructurer` — patching those
    # classes at their definition module makes the local import pick up the mock.
    with (
        patch("code.pipeline.ocr_engine.OCREngine", return_value=mock_ocr_engine),
        patch("code.pipeline.structurer.DocumentStructurer", return_value=mock_structurer),
    ):
        ingester = DocumentIngester()
        doc = ingester.process(str(pdf_path), draft_type="case_fact_summary")

    # --- Assertions ---
    assert doc is not None, "ingester.process() must return a ProcessedDocument"
    assert doc.id, "ProcessedDocument must have a non-empty id"
    assert doc.filename == "test_case.pdf"
    assert doc.page_count >= 1, "At least one page must be processed"
    assert len(doc.chunks) > 0, (
        "Non-empty legal text must produce at least one chunk"
    )

    # Verify the document is saved to the patched PROCESSED_DIR
    saved_path = pipeline_dirs / "processed" / f"{doc.id}.json"
    assert saved_path.exists(), (
        f"ProcessedDocument must be saved at {saved_path}"
    )
    with open(saved_path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["id"] == doc.id


# ---------------------------------------------------------------------------
# test_document_ingestion_unsupported_type
# ---------------------------------------------------------------------------

def test_document_ingestion_unsupported_type(pipeline_dirs):
    """Processing an unsupported file extension must not raise; instead the
    document should be saved to disk (possibly with empty text and no chunks).
    """
    from code.pipeline.ingestion import DocumentIngester

    bad_file = pipeline_dirs / "uploads" / "doc.xyz"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text("not a pdf or image")

    mock_ocr_engine = MagicMock()
    mock_structurer = MagicMock()
    mock_structurer.extract.return_value = {}

    with (
        patch("code.pipeline.ocr_engine.OCREngine", return_value=mock_ocr_engine),
        patch("code.pipeline.structurer.DocumentStructurer", return_value=mock_structurer),
    ):
        ingester = DocumentIngester()
        doc = ingester.process(str(bad_file), draft_type="case_fact_summary")

    assert doc is not None
    saved_path = pipeline_dirs / "processed" / f"{doc.id}.json"
    assert saved_path.exists(), (
        "Document must be saved even for unsupported file type"
    )


# ---------------------------------------------------------------------------
# test_chunker_produces_correct_structure
# ---------------------------------------------------------------------------

def test_chunker_produces_correct_structure():
    """TextChunker.chunk() must return TextChunk objects with correct fields.

    This is a pure-logic test — no I/O, no mocking needed.
    """
    from code.pipeline.chunker import TextChunker

    text = (
        "FACTS:\n"
        "1. The parties entered into an agreement on March 1, 2023.\n"
        "2. Plaintiff completed all deliverables by August 31, 2023.\n\n"
        "RELIEF SOUGHT:\n"
        "Plaintiff seeks $37,500 plus interest."
    )

    chunker = TextChunker()
    chunks = chunker.chunk(text, document_id="doc_chunk_test")

    assert len(chunks) >= 1, "Non-empty text must produce at least one chunk"
    for chunk in chunks:
        assert chunk.id, "Each chunk must have a non-empty id"
        assert chunk.document_id == "doc_chunk_test"
        assert chunk.text, "Each chunk must have non-empty text"
        assert chunk.char_start >= 0
        assert chunk.char_end > chunk.char_start


# ---------------------------------------------------------------------------
# test_chunker_empty_text
# ---------------------------------------------------------------------------

def test_chunker_empty_text():
    """TextChunker.chunk() on empty/whitespace-only text must return []."""
    from code.pipeline.chunker import TextChunker

    chunker = TextChunker()
    assert chunker.chunk("", "doc_empty") == []
    assert chunker.chunk("   \n\n  ", "doc_ws") == []


# ---------------------------------------------------------------------------
# test_processed_document_save_and_load_round_trip
# ---------------------------------------------------------------------------

def test_processed_document_save_and_load_round_trip(pipeline_dirs):
    """ProcessedDocument.save() writes JSON; ProcessedDocument.load() reads it back
    with all fields intact.
    """
    from code.pipeline.ingestion import ProcessedDocument
    import code.pipeline.ingestion as ing_mod

    # Already patched by pipeline_dirs fixture
    doc = ProcessedDocument(
        id="doc_roundtrip",
        filename="round.pdf",
        full_text="Some legal text for round-trip test.",
        page_count=2,
        confidence=0.88,
    )
    doc.structured_fields = {"parties": {"defendant": "Test Corp"}}
    doc.chunks = [
        {"id": "c1", "text": "Some legal text", "document_id": "doc_roundtrip",
         "page_number": 1, "char_start": 0, "char_end": 15}
    ]
    saved_path = doc.save()

    assert saved_path.exists(), "save() must create the JSON file"

    loaded = ProcessedDocument.load("doc_roundtrip")
    assert loaded is not None, "load() must return a ProcessedDocument"
    assert loaded.id == "doc_roundtrip"
    assert loaded.filename == "round.pdf"
    assert loaded.full_text == "Some legal text for round-trip test."
    assert loaded.page_count == 2
    assert abs(loaded.confidence - 0.88) < 1e-9
    assert loaded.structured_fields["parties"]["defendant"] == "Test Corp"
    assert len(loaded.chunks) == 1
