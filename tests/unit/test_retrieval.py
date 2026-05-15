"""Unit tests for code.retrieval — BM25-based indexer, searcher, and EvidencePackager.

No ChromaDB, no embedding model. All retrieval is keyword-based BM25 over
processed document JSON files. Tests run fast (< 1 s each).
"""

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# test_indexer_imports
# ---------------------------------------------------------------------------

def test_indexer_imports():
    """Verify DocumentIndexer can be imported."""
    from code.retrieval.indexer import DocumentIndexer  # noqa: F401
    assert DocumentIndexer is not None


# ---------------------------------------------------------------------------
# test_index_document_empty_chunks
# ---------------------------------------------------------------------------

def test_index_document_empty_chunks(tmp_path, monkeypatch):
    """index_document with an empty chunk list must return 0."""
    import code.retrieval.indexer as indexer_module
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    monkeypatch.setattr(indexer_module, "PROCESSED_DIR", processed_dir)

    from code.retrieval.indexer import DocumentIndexer
    indexer = DocumentIndexer()
    count = indexer.index_document("doc_empty", [])

    assert count == 0, "Empty chunk list must return 0"


# ---------------------------------------------------------------------------
# test_index_document_returns_count
# ---------------------------------------------------------------------------

def test_index_document_returns_count(tmp_path, monkeypatch):
    """index_document with N chunks must return N when the doc JSON exists."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)

    import code.retrieval.indexer as indexer_module
    monkeypatch.setattr(indexer_module, "PROCESSED_DIR", processed_dir)

    # Create the expected document JSON so the indexer can find it
    doc_id = "doc_abc"
    chunks = [
        {"id": f"c{i}", "text": f"chunk text {i}", "page_number": 1,
         "char_start": i * 100, "char_end": i * 100 + 99}
        for i in range(3)
    ]
    (processed_dir / f"{doc_id}.json").write_text(
        json.dumps({"id": doc_id, "chunks": chunks}), encoding="utf-8"
    )

    from code.retrieval.indexer import DocumentIndexer
    indexer = DocumentIndexer()
    count = indexer.index_document(doc_id, chunks)

    assert count == 3, f"Expected 3, got {count}"


# ---------------------------------------------------------------------------
# test_searcher_imports
# ---------------------------------------------------------------------------

def test_searcher_imports():
    """Verify EvidenceSearcher and EvidenceChunk can be imported."""
    from code.retrieval.searcher import EvidenceSearcher, EvidenceChunk  # noqa: F401
    assert EvidenceSearcher is not None
    assert EvidenceChunk is not None


# ---------------------------------------------------------------------------
# test_searcher_returns_empty_when_no_docs
# ---------------------------------------------------------------------------

def test_searcher_returns_empty_when_no_docs(tmp_path, monkeypatch):
    """EvidenceSearcher.search() returns [] when PROCESSED_DIR has no documents."""
    import code.retrieval.searcher as searcher_module
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    monkeypatch.setattr(searcher_module, "PROCESSED_DIR", processed_dir)

    from code.retrieval.searcher import EvidenceSearcher
    searcher = EvidenceSearcher()
    results = searcher.search("defendant names", top_k=3)

    assert results == [], f"Expected [] with no docs, got {results}"


# ---------------------------------------------------------------------------
# test_searcher_bm25_ranks_relevant_chunk_first
# ---------------------------------------------------------------------------

def test_searcher_bm25_ranks_relevant_chunk_first(tmp_path, monkeypatch):
    """The chunk most relevant to the query should rank first via BM25."""
    import code.retrieval.searcher as searcher_module
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    monkeypatch.setattr(searcher_module, "PROCESSED_DIR", processed_dir)

    doc_id = "doc_bm25"
    chunks = [
        {"id": "c_irrelevant", "text": "The weather in London is mild today.",
         "document_id": doc_id, "page_number": 1, "char_start": 0, "char_end": 36},
        {"id": "c_relevant",
         "text": "The defendant Marcus Bell filed a motion seeking dismissal.",
         "document_id": doc_id, "page_number": 2, "char_start": 37, "char_end": 95},
    ]
    (processed_dir / f"{doc_id}.json").write_text(
        json.dumps({"id": doc_id, "chunks": chunks}), encoding="utf-8"
    )

    from code.retrieval.searcher import EvidenceSearcher
    searcher = EvidenceSearcher()
    results = searcher.search("defendant filed motion", doc_ids=[doc_id], top_k=2)

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0].chunk_id == "c_relevant", (
        f"Relevant chunk should rank first; got {results[0].chunk_id!r}"
    )


# ---------------------------------------------------------------------------
# test_evidence_packager_format
# ---------------------------------------------------------------------------

def test_evidence_packager_format():
    """EvidencePackager.package() with 2 chunks should produce [E1] and [E2] markers."""
    from code.retrieval.evidence import EvidencePackager
    from code.retrieval.searcher import EvidenceChunk

    chunk1 = EvidenceChunk(
        chunk_id="c1",
        text="The defendant filed a motion on January 15, 2024.",
        document_id="doc_001",
        page_number=1,
        char_start=0,
        char_end=50,
        similarity_score=0.92,
    )
    chunk2 = EvidenceChunk(
        chunk_id="c2",
        text="Plaintiff seeks damages of fifty thousand dollars.",
        document_id="doc_001",
        page_number=2,
        char_start=51,
        char_end=100,
        similarity_score=0.85,
    )

    packager = EvidencePackager()
    package = packager.package([chunk1, chunk2])

    assert "[E1]" in package.formatted_text, "[E1] citation must appear in formatted_text"
    assert "[E2]" in package.formatted_text, "[E2] citation must appear in formatted_text"
    assert package.chunk_count == 2, f"Expected chunk_count=2, got {package.chunk_count}"
    assert "E1" in package.citation_map, "E1 must be a key in citation_map"
    assert "E2" in package.citation_map, "E2 must be a key in citation_map"


# ---------------------------------------------------------------------------
# test_evidence_packager_empty_chunks
# ---------------------------------------------------------------------------

def test_evidence_packager_empty_chunks():
    """EvidencePackager.package() with no chunks returns chunk_count == 0."""
    from code.retrieval.evidence import EvidencePackager

    packager = EvidencePackager()
    package = packager.package([])

    assert package.chunk_count == 0
    assert package.formatted_text  # should not be empty string
    assert package.citation_map == {}


# ---------------------------------------------------------------------------
# test_evidence_packager_truncates_long_chunks
# ---------------------------------------------------------------------------

def test_evidence_packager_truncates_long_chunks():
    """Chunks with text longer than 500 chars must be truncated with '...' in output."""
    from code.retrieval.evidence import EvidencePackager
    from code.retrieval.searcher import EvidenceChunk

    long_text = "A" * 600
    chunk = EvidenceChunk(
        chunk_id="c_long",
        text=long_text,
        document_id="doc_long",
        page_number=1,
        char_start=0,
        char_end=600,
        similarity_score=0.7,
    )

    packager = EvidencePackager()
    package = packager.package([chunk])

    assert "..." in package.formatted_text, (
        "Long chunk text should be truncated with '...'"
    )
    # The raw text in the citation_map must remain intact (not truncated)
    assert package.citation_map["E1"]["text"] == long_text
