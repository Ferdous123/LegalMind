"""Document indexer — stores chunks in ChromaDB for semantic retrieval.

Chunks are embedded using sentence-transformers (BAAI/bge-small-en-v1.5 for speed,
upgradeable to bge-m3) and stored in a persistent ChromaDB collection. Falls back
to JSON-only storage if ChromaDB is unavailable.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from config.paths import PROCESSED_DIR, CHROMA_DIR

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "legalmind_chunks"
_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_chroma_client = None
_embedding_fn = None


def _get_embedding_fn():
    """Lazy-load the embedding function (runs on CPU to preserve GPU for LLMs)."""
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=_EMBEDDING_MODEL,
            device="cpu",
        )
        logger.info("Loaded embedding model: %s (CPU)", _EMBEDDING_MODEL)
        return _embedding_fn
    except Exception as exc:
        logger.warning("Could not load embedding model: %s", exc)
        return None


def _get_collection():
    """Get or create the ChromaDB collection with persistence."""
    global _chroma_client
    try:
        import chromadb
        if _chroma_client is None:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        ef = _get_embedding_fn()
        if ef:
            return _chroma_client.get_or_create_collection(
                name=_COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            return _chroma_client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
    except Exception as exc:
        logger.warning("ChromaDB unavailable: %s", exc)
        return None


class DocumentIndexer:
    """Indexes document chunks in ChromaDB for semantic retrieval.

    Chunks are also persisted in the ProcessedDocument JSON for BM25 fallback.
    """

    def index_document(self, document_id: str, chunks: list[dict]) -> int:
        """Index chunks for the given document into ChromaDB.

        Args:
            document_id: Unique document identifier.
            chunks: List of chunk dicts (id, text, page_number, etc.).

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            logger.warning("No chunks to index for document %s", document_id)
            return 0

        doc_path = PROCESSED_DIR / f"{document_id}.json"
        if not doc_path.exists():
            logger.warning("Document file not found for %s — save before indexing", document_id)
            return 0

        valid_chunks = [c for c in chunks if c.get("text", "").strip()]
        if not valid_chunks:
            return 0

        collection = _get_collection()
        if collection is None:
            logger.info("ChromaDB unavailable — using JSON-only storage for %s", document_id)
            return len(valid_chunks)

        try:
            self._delete_document_chunks(collection, document_id)

            ids = [c.get("id", f"chunk_{i}") for i, c in enumerate(valid_chunks)]
            documents = [c["text"] for c in valid_chunks]
            metadatas = [
                {
                    "document_id": document_id,
                    "page_number": c.get("page_number", 0),
                    "char_start": c.get("char_start", 0),
                    "char_end": c.get("char_end", 0),
                }
                for c in valid_chunks
            ]

            batch_size = 100
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i:i + batch_size],
                    documents=documents[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                )

            logger.info("Indexed %d chunks for %s in ChromaDB", len(valid_chunks), document_id)
        except Exception as exc:
            logger.warning("ChromaDB indexing failed for %s: %s — falling back to JSON", document_id, exc)

        return len(valid_chunks)

    def _delete_document_chunks(self, collection, document_id: str) -> None:
        """Remove existing chunks for a document before re-indexing."""
        try:
            collection.delete(where={"document_id": document_id})
        except Exception:
            pass

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
