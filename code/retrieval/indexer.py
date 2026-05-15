"""Document indexer — chunks are stored in the processed document JSON.

No external vector store or embedding model is needed. The indexer is a
thin shim that validates chunk data and confirms the document has been
saved. Actual retrieval uses BM25 over the stored chunks (see searcher.py).
"""

import logging

from config.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Confirms document chunks are ready for BM25 retrieval.

    Chunks are persisted as part of the ProcessedDocument JSON in
    data/processed/<doc_id>.json — no separate index needed.
    """

    def index_document(self, document_id: str, chunks: list[dict]) -> int:
        """Validate and confirm chunks for the given document.

        Args:
            document_id: Unique document identifier.
            chunks: List of chunk dicts (id, text, page_number, etc.).

        Returns:
            Number of chunks confirmed.
        """
        if not chunks:
            logger.warning("No chunks to index for document %s", document_id)
            return 0

        doc_path = PROCESSED_DIR / f"{document_id}.json"
        if not doc_path.exists():
            logger.warning("Document file not found for %s — save before indexing", document_id)
            return 0

        valid = sum(1 for c in chunks if c.get("text", "").strip())
        logger.info("Indexed %d/%d chunks for %s (stored in doc JSON)", valid, len(chunks), document_id)
        return valid

    def get_document_count(self) -> int:
        """Return number of processed documents available for retrieval."""
        try:
            return len([p for p in PROCESSED_DIR.glob("*.json") if not p.stem.startswith("draft_")])
        except Exception:
            return 0

    def get_indexed_documents(self) -> list[str]:
        """Return list of document IDs available for retrieval."""
        try:
            return sorted(
                p.stem for p in PROCESSED_DIR.glob("*.json")
                if not p.stem.startswith("draft_")
            )
        except Exception:
            return []
