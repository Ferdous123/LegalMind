"""Document indexer — stores document chunks in ChromaDB for retrieval."""

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings

from config.paths import CHROMA_DIR
from code.llm_interface.model_manager import ModelManager

logger = logging.getLogger(__name__)

COLLECTION_NAME = "legalmind_documents"


class DocumentIndexer:
    """Indexes document chunks into ChromaDB for semantic retrieval.

    One collection holds all document chunks. Each chunk is stored with:
    - embedding (BGE-M3)
    - metadata (document_id, page_number, char offsets)
    - raw text
    """

    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def index_document(self, document_id: str, chunks: list[dict]) -> int:
        """Index all chunks from a processed document.

        Args:
            document_id: Unique document identifier.
            chunks: List of chunk dicts with keys: id, text, page_number, char_start, char_end.

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            return 0

        # Remove existing entries for this document (idempotent re-indexing)
        self._remove_document(document_id)

        # Generate embeddings
        mgr = ModelManager.instance()
        texts = [c["text"] for c in chunks]
        embeddings = mgr.embed(texts)

        # Prepare batch data
        ids = [c.get("id", f"chunk_{i}") for i, c in enumerate(chunks)]
        metadatas = [{
            "document_id": document_id,
            "page_number": c.get("page_number", 0),
            "char_start": c.get("char_start", 0),
            "char_end": c.get("char_end", 0),
        } for c in chunks]

        # Upsert in batches (ChromaDB has batch size limits)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self._collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                documents=texts[i:end],
            )

        logger.info("Indexed %d chunks for document %s", len(ids), document_id)
        return len(ids)

    def _remove_document(self, document_id: str) -> None:
        """Remove all indexed chunks for a document."""
        try:
            results = self._collection.get(
                where={"document_id": document_id}
            )
            if results["ids"]:
                self._collection.delete(ids=results["ids"])
        except Exception as e:
            logger.debug("No existing entries to remove for %s: %s", document_id, e)

    def get_document_count(self) -> int:
        """Return total number of indexed chunks."""
        return self._collection.count()

    def get_indexed_documents(self) -> list[str]:
        """Return list of unique document IDs that have been indexed."""
        results = self._collection.get(include=["metadatas"])
        doc_ids = set()
        for meta in results.get("metadatas", []):
            if meta and "document_id" in meta:
                doc_ids.add(meta["document_id"])
        return sorted(doc_ids)
