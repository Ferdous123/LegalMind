"""Evidence searcher — BM25 keyword retrieval over processed document chunks.

No embedding model or vector store. Chunks are loaded directly from the
processed document JSON (data/processed/<doc_id>.json) and ranked with
BM25. This follows the same model-free retrieval pattern as TRACE's
query_by_field: pure keyword matching, foolproof, no GPU required.
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
    """A retrieved evidence chunk with BM25 relevance score."""
    chunk_id: str
    text: str
    document_id: str
    page_number: int
    char_start: int
    char_end: int
    similarity_score: float  # BM25 score normalised to [0, 1]


# ---------------------------------------------------------------------------
# Minimal BM25 implementation (no external library required)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop empty tokens."""
    return [t for t in re.split(r"\W+", text.lower()) if t]


class _BM25:
    """BM25 Okapi over a corpus of token lists.

    Parameters follow the standard defaults: k1=1.5, b=0.75.
    """

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
        """Return (index, score) pairs sorted descending by score."""
        scores = [(i, self.score(query_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# ---------------------------------------------------------------------------
# Field-to-query expansions (same as TRACE's query_by_field vocabulary)
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
    """Retrieves relevant document chunks using BM25 keyword ranking.

    Chunks are loaded from data/processed/<doc_id>.json on each search
    (lightweight JSON read). No embedding model or vector store required.
    """

    def search(self, query: str, doc_ids: Optional[list[str]] = None,
               top_k: int = 5) -> list[EvidenceChunk]:
        """Search for relevant evidence chunks via BM25.

        Args:
            query: Natural language query string.
            doc_ids: Optional filter to specific document IDs.
            top_k: Maximum number of chunks to return.

        Returns:
            List of EvidenceChunk sorted by BM25 score (highest first).
        """
        chunks = self._load_chunks(doc_ids)
        if not chunks:
            logger.warning("No chunks found for doc_ids=%s", doc_ids)
            return []

        corpus = [_tokenise(c["text"]) for c in chunks]
        bm25 = _BM25(corpus)
        query_tokens = _tokenise(query)

        if not query_tokens:
            # No meaningful query terms — return first top_k chunks
            return [self._to_evidence(chunks[i], 0.0) for i in range(min(top_k, len(chunks)))]

        ranked = bm25.rank(query_tokens)
        # Normalise scores to [0, 1]
        max_score = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0

        results = []
        for idx, raw_score in ranked[:top_k]:
            norm_score = raw_score / max_score if max_score > 0 else 0.0
            results.append(self._to_evidence(chunks[idx], norm_score))

        logger.info("BM25 search '%s': %d chunks searched, %d returned",
                    query[:60], len(chunks), len(results))
        return results

    def search_by_field(self, field_name: str, doc_ids: Optional[list[str]] = None,
                        top_k: int = 3) -> list[EvidenceChunk]:
        """Search for evidence relevant to a specific legal field.

        Expands the field name to a keyword query using the legal vocabulary
        table, then delegates to search().
        """
        query = _FIELD_QUERIES.get(field_name, field_name.replace("_", " "))
        return self.search(query, doc_ids=doc_ids, top_k=top_k)

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
