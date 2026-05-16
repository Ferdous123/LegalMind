"""Evidence searcher — hybrid semantic + BM25 retrieval over document chunks.

Primary: ChromaDB semantic search using sentence-transformer embeddings.
Fallback: BM25 keyword retrieval from processed document JSON files.
Results from both methods are fused using Reciprocal Rank Fusion (RRF).
"""

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from config.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)


@dataclass
class EvidenceChunk:
    """A retrieved evidence chunk with relevance score."""
    chunk_id: str
    text: str
    document_id: str
    page_number: int
    char_start: int
    char_end: int
    similarity_score: float


# ---------------------------------------------------------------------------
# BM25 implementation (fallback when ChromaDB unavailable)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if t]


class _BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(1, self.N)
        self.df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        doc = self.corpus[doc_index]
        doc_len = len(doc)
        tf_map = Counter(doc)
        score = 0.0
        for term in query_tokens:
            if term not in tf_map:
                continue
            tf = tf_map[term]
            df = self.df.get(term, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            score += idf * norm
        return score

    def rank(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        scores = [(i, self.score(query_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# ---------------------------------------------------------------------------
# Field-to-query expansions (legal vocabulary for field-specific retrieval)
# ---------------------------------------------------------------------------

_FIELD_QUERIES: dict[str, str] = {
    "parties": "names parties plaintiffs defendants individuals organizations counsel",
    "key_dates": "dates filing deadline hearing trial incident signed effective",
    "claims": "claims allegations charges causes action legal basis relief",
    "procedural_history": "procedural history motions filings court orders judgment",
    "evidence_items": "evidence exhibits documents testimony witnesses statement",
    "relief_sought": "relief damages remedy injunction requested amount award",
    "property_description": "property legal description address parcel lot block survey",
    "chain_of_title": "deed transfer grantor grantee ownership conveyance title",
    "encumbrances": "lien mortgage easement encumbrance restriction covenant",
    "deadlines": "deadline due date response time limit cure period notice",
    "requirements": "requirements demands obligations compliance steps must shall",
    "documents_present": "documents present included attached exhibit filed received",
    "documents_missing": "documents missing absent required outstanding needed",
}


class EvidenceSearcher:
    """Hybrid retrieval: semantic search (ChromaDB) + BM25 keyword matching.

    Uses Reciprocal Rank Fusion to merge results from both methods.
    Falls back to BM25-only if ChromaDB/embeddings are unavailable.
    """

    def search(self, query: str, doc_ids: Optional[list[str]] = None,
               top_k: int = 5) -> list[EvidenceChunk]:
        """Search for relevant evidence chunks using hybrid retrieval.

        Args:
            query: Natural language query string.
            doc_ids: Optional filter to specific document IDs.
            top_k: Maximum number of chunks to return.

        Returns:
            List of EvidenceChunk sorted by relevance (highest first).
        """
        semantic_results = self._semantic_search(query, doc_ids, top_k * 2)
        bm25_results = self._bm25_search(query, doc_ids, top_k * 2)

        if semantic_results and bm25_results:
            fused = self._reciprocal_rank_fusion(semantic_results, bm25_results, top_k)
            logger.info("Hybrid search '%s': %d semantic + %d BM25 → %d fused",
                        query[:40], len(semantic_results), len(bm25_results), len(fused))
            return fused
        elif semantic_results:
            return semantic_results[:top_k]
        elif bm25_results:
            return bm25_results[:top_k]
        return []

    def search_by_field(self, field_name: str, doc_ids: Optional[list[str]] = None,
                        top_k: int = 3) -> list[EvidenceChunk]:
        """Search for evidence relevant to a specific legal field."""
        query = _FIELD_QUERIES.get(field_name, field_name.replace("_", " "))
        return self.search(query, doc_ids=doc_ids, top_k=top_k)

    # ------------------------------------------------------------------
    # Semantic search via ChromaDB
    # ------------------------------------------------------------------

    def _semantic_search(self, query: str, doc_ids: Optional[list[str]],
                         top_k: int) -> list[EvidenceChunk]:
        """Query ChromaDB for semantically similar chunks."""
        try:
            from code.retrieval.indexer import _get_collection
            collection = _get_collection()
            if collection is None:
                return []

            where_filter = None
            if doc_ids and len(doc_ids) == 1:
                where_filter = {"document_id": doc_ids[0]}
            elif doc_ids and len(doc_ids) > 1:
                where_filter = {"document_id": {"$in": doc_ids}}

            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            chunks = []
            if results and results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    text = results["documents"][0][i] if results["documents"] else ""
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    similarity = max(0.0, 1.0 - distance)

                    chunks.append(EvidenceChunk(
                        chunk_id=chunk_id,
                        text=text,
                        document_id=meta.get("document_id", ""),
                        page_number=meta.get("page_number", 0),
                        char_start=meta.get("char_start", 0),
                        char_end=meta.get("char_end", 0),
                        similarity_score=similarity,
                    ))
            return chunks
        except Exception as exc:
            logger.warning("Semantic search failed: %s — falling back to BM25", exc)
            return []

    # ------------------------------------------------------------------
    # BM25 keyword search (fallback)
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, doc_ids: Optional[list[str]],
                     top_k: int) -> list[EvidenceChunk]:
        """BM25 keyword search over stored chunks."""
        chunks = self._load_chunks(doc_ids)
        if not chunks:
            return []

        corpus = [_tokenise(c["text"]) for c in chunks]
        bm25 = _BM25(corpus)
        query_tokens = _tokenise(query)

        if not query_tokens:
            return [self._to_evidence(chunks[i], 0.0) for i in range(min(top_k, len(chunks)))]

        ranked = bm25.rank(query_tokens)
        max_score = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0

        results = []
        for idx, raw_score in ranked[:top_k]:
            norm_score = raw_score / max_score if max_score > 0 else 0.0
            results.append(self._to_evidence(chunks[idx], norm_score))
        return results

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(self, semantic: list[EvidenceChunk],
                                bm25: list[EvidenceChunk],
                                top_k: int, k: int = 60) -> list[EvidenceChunk]:
        """Fuse results from semantic and BM25 using RRF scoring."""
        scores: dict[str, float] = {}
        chunk_map: dict[str, EvidenceChunk] = {}

        for rank, chunk in enumerate(semantic):
            key = chunk.chunk_id or chunk.text[:80]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[key] = chunk

        for rank, chunk in enumerate(bm25):
            key = chunk.chunk_id or chunk.text[:80]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = chunk

        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results = []
        for key in sorted_keys[:top_k]:
            chunk = chunk_map[key]
            chunk.similarity_score = scores[key]
            results.append(chunk)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_chunks(self, doc_ids: Optional[list[str]]) -> list[dict]:
        """Load chunks from processed document JSON files."""
        chunks: list[dict] = []
        try:
            if doc_ids:
                paths = [PROCESSED_DIR / f"{did}.json" for did in doc_ids]
            else:
                paths = [p for p in PROCESSED_DIR.glob("*.json") if not p.stem.startswith("draft_")]

            for path in paths:
                if not path.exists():
                    continue
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                    for chunk in doc.get("chunks", []):
                        if chunk.get("text", "").strip():
                            chunks.append(chunk)
                except Exception as exc:
                    logger.warning("Could not load chunks from %s: %s", path, exc)
        except Exception as exc:
            logger.error("Chunk loading failed: %s", exc)
        return chunks

    @staticmethod
    def _to_evidence(chunk: dict, score: float) -> EvidenceChunk:
        return EvidenceChunk(
            chunk_id=chunk.get("id", ""),
            text=chunk.get("text", ""),
            document_id=chunk.get("document_id", ""),
            page_number=chunk.get("page_number", 0),
            char_start=chunk.get("char_start", 0),
            char_end=chunk.get("char_end", 0),
            similarity_score=score,
        )
