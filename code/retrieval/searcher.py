"""Evidence searcher — retrieves relevant chunks for draft generation."""

import logging
from dataclasses import dataclass
from typing import Optional

import chromadb

from config.paths import CHROMA_DIR
from code.llm_interface.model_manager import ModelManager

logger = logging.getLogger(__name__)

COLLECTION_NAME = "legalmind_documents"


@dataclass
class EvidenceChunk:
    """A retrieved evidence chunk with relevance scoring."""
    chunk_id: str
    text: str
    document_id: str
    page_number: int
    char_start: int
    char_end: int
    similarity_score: float


class EvidenceSearcher:
    """Retrieves relevant document chunks for a given query.

    Uses semantic search (BGE-M3 embeddings + cosine similarity) over
    indexed chunks in ChromaDB.
    """

    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def search(self, query: str, doc_ids: Optional[list[str]] = None,
               top_k: int = 5) -> list[EvidenceChunk]:
        """Search for relevant evidence chunks.

        Args:
            query: Natural language query or field description.
            doc_ids: Optional filter to specific documents.
            top_k: Number of results to return.

        Returns:
            List of EvidenceChunk sorted by relevance (highest first).
        """
        try:
            mgr = ModelManager.instance()
            query_embedding = mgr.embed([query])[0]
        except Exception as e:
            logger.warning("Embedding unavailable for search (model loading?): %s", e)
            return self._fallback_keyword_search(query, doc_ids, top_k)

        where_filter = None
        if doc_ids and len(doc_ids) == 1:
            where_filter = {"document_id": doc_ids[0]}
        elif doc_ids and len(doc_ids) > 1:
            where_filter = {"document_id": {"$in": doc_ids}}

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = 1.0 - distance  # ChromaDB returns distance, not similarity

                chunks.append(EvidenceChunk(
                    chunk_id=chunk_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    document_id=meta.get("document_id", ""),
                    page_number=meta.get("page_number", 0),
                    char_start=meta.get("char_start", 0),
                    char_end=meta.get("char_end", 0),
                    similarity_score=similarity,
                ))

        return chunks

    def _fallback_keyword_search(self, query: str, doc_ids: Optional[list[str]] = None,
                                 top_k: int = 5) -> list[EvidenceChunk]:
        """Return chunks for the requested documents without embedding search.

        Falls back to fetching stored chunks directly from ChromaDB by document
        ID when the embedding model is unavailable.  No relevance ranking is
        applied — chunks are returned in storage order.
        """
        try:
            where_filter = None
            if doc_ids and len(doc_ids) == 1:
                where_filter = {"document_id": doc_ids[0]}
            elif doc_ids and len(doc_ids) > 1:
                where_filter = {"document_id": {"$in": doc_ids}}

            get_kwargs: dict = {"include": ["documents", "metadatas"]}
            if where_filter:
                get_kwargs["where"] = where_filter

            results = self._collection.get(**get_kwargs)

            chunks = []
            ids = results.get("ids", [])
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []

            for i in range(min(top_k, len(ids))):
                meta = metas[i] if i < len(metas) else {}
                text = docs[i] if i < len(docs) else ""
                chunks.append(EvidenceChunk(
                    chunk_id=ids[i],
                    text=text,
                    document_id=meta.get("document_id", ""),
                    page_number=meta.get("page_number", 0),
                    char_start=meta.get("char_start", 0),
                    char_end=meta.get("char_end", 0),
                    similarity_score=0.0,
                ))
            return chunks
        except Exception as e:
            logger.error("Fallback keyword search failed: %s", e)
            return []

    def search_by_field(self, field_name: str, doc_ids: Optional[list[str]] = None,
                        top_k: int = 3) -> list[EvidenceChunk]:
        """Search for evidence relevant to a specific field.

        Constructs a query from the field name for targeted retrieval.
        """
        field_queries = {
            "parties": "names of parties plaintiffs defendants individuals organizations",
            "key_dates": "dates filing deadline hearing trial incident",
            "claims": "claims allegations charges causes of action legal basis",
            "procedural_history": "procedural history motions filings court orders",
            "evidence_items": "evidence exhibits documents testimony witnesses",
            "relief_sought": "relief damages remedy injunction requested",
            "property_description": "property legal description address parcel lot",
            "chain_of_title": "deed transfer grantor grantee ownership conveyance",
            "encumbrances": "lien mortgage easement encumbrance restriction",
            "deadlines": "deadline due date response time limit cure period",
            "requirements": "requirements demands obligations compliance",
        }

        query = field_queries.get(field_name, field_name.replace("_", " "))
        return self.search(query, doc_ids=doc_ids, top_k=top_k)
