"""Unit tests for code.retrieval.indexer.DocumentIndexer,
code.retrieval.searcher.EvidenceSearcher, and code.retrieval.evidence.EvidencePackager.

ChromaDB and ModelManager are mocked so no GPU or persistent state is needed.
Tests run fast (< 1 s each).
"""

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# test_indexer_imports
# ---------------------------------------------------------------------------

def test_indexer_imports():
    """Verify DocumentIndexer and COLLECTION_NAME can be imported."""
    from code.retrieval.indexer import DocumentIndexer, COLLECTION_NAME  # noqa: F401
    assert DocumentIndexer is not None
    assert isinstance(COLLECTION_NAME, str)


# ---------------------------------------------------------------------------
# test_index_document_empty_chunks
# ---------------------------------------------------------------------------

def test_index_document_empty_chunks(tmp_path):
    """index_document with an empty chunk list must return 0 without hitting
    ChromaDB or ModelManager.
    """
    from code.retrieval.indexer import DocumentIndexer

    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection

    with (
        patch("code.retrieval.indexer.chromadb.PersistentClient", return_value=mock_chroma_client),
        patch("config.paths.CHROMA_DIR", tmp_path / "chroma"),
    ):
        indexer = DocumentIndexer()
        count = indexer.index_document("doc_empty", [])

    assert count == 0, "Empty chunk list must return 0"
    # ModelManager.embed should NOT have been called
    mock_collection.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# test_index_document_returns_count
# ---------------------------------------------------------------------------

def test_index_document_returns_count(tmp_path):
    """index_document with N chunks must return N after upserting into ChromaDB."""
    from code.retrieval.indexer import DocumentIndexer

    chunks = [
        {"id": f"c{i}", "text": f"chunk text {i}", "page_number": 1,
         "char_start": i * 100, "char_end": i * 100 + 100}
        for i in range(3)
    ]

    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    # _remove_document: simulate no existing ids
    mock_collection.get.return_value = {"ids": []}

    mock_mgr = MagicMock()
    # embed returns one vector per chunk
    mock_mgr.embed.return_value = [[0.1] * 128] * len(chunks)

    with (
        patch("code.retrieval.indexer.chromadb.PersistentClient", return_value=mock_chroma_client),
        patch("code.retrieval.indexer.ModelManager") as mock_mgr_cls,
        patch("config.paths.CHROMA_DIR", tmp_path / "chroma"),
    ):
        mock_mgr_cls.instance.return_value = mock_mgr
        indexer = DocumentIndexer()
        count = indexer.index_document("doc_abc", chunks)

    assert count == 3, f"Expected 3, got {count}"
    mock_collection.upsert.assert_called_once()


# ---------------------------------------------------------------------------
# test_searcher_imports
# ---------------------------------------------------------------------------

def test_searcher_imports():
    """Verify EvidenceSearcher and EvidenceChunk can be imported."""
    from code.retrieval.searcher import EvidenceSearcher, EvidenceChunk  # noqa: F401
    assert EvidenceSearcher is not None
    assert EvidenceChunk is not None


# ---------------------------------------------------------------------------
# test_evidence_packager_format
# ---------------------------------------------------------------------------

def test_evidence_packager_format():
    """EvidencePackager.package() with 2 EvidenceChunks should produce
    formatted_text that contains [E1] and [E2] citation markers.
    """
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
    """EvidencePackager.package() with no chunks returns a package with
    chunk_count == 0 and a non-empty formatted_text placeholder.
    """
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


# ---------------------------------------------------------------------------
# test_searcher_search_returns_empty_on_exception(tmp_path)
# ---------------------------------------------------------------------------

def test_searcher_search_returns_empty_on_exception(tmp_path):
    """If the ChromaDB query raises an exception, search() must return []
    rather than propagating the exception.
    """
    from code.retrieval.searcher import EvidenceSearcher

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.query.side_effect = RuntimeError("collection is empty")
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_mgr = MagicMock()
    mock_mgr.embed.return_value = [[0.0] * 128]

    with (
        patch("code.retrieval.searcher.chromadb.PersistentClient", return_value=mock_client),
        patch("code.retrieval.searcher.ModelManager") as mock_cls,
        patch("config.paths.CHROMA_DIR", tmp_path / "chroma"),
    ):
        mock_cls.instance.return_value = mock_mgr
        searcher = EvidenceSearcher()
        results = searcher.search("defendant names", top_k=3)

    assert results == [], f"Expected [] on exception, got {results}"
