"""Exemplar retriever — finds relevant past corrections for prompt injection.

Uses BM25 keyword search over the active correction JSONL files to find
semantically similar past corrections for few-shot prompt injection.
"""

import logging
import math
import re
from collections import Counter
from typing import Optional

from code.learning.correction_store import Correction, CorrectionStore

logger = logging.getLogger(__name__)

_STOP = {"the", "a", "an", "of", "in", "to", "and", "or", "is", "was",
         "are", "were", "be", "been", "has", "have", "had", "that", "this",
         "it", "for", "on", "at", "by", "with", "from", "as", "not"}


def _tokenise(text: str) -> list[str]:
    words = re.split(r"\W+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOP]


def _bm25_score(query_tokens: list[str], doc_tokens: list[str],
                df: dict[str, int], N: int, avgdl: float,
                k1: float = 1.5, b: float = 0.75) -> float:
    tf_map = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in tf_map:
            continue
        tf = tf_map[term]
        n_docs = df.get(term, 0)
        idf = math.log((N - n_docs + 0.5) / (n_docs + 0.5) + 1)
        norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / max(1, avgdl)))
        score += idf * norm
    return score


class ExemplarRetriever:
    """Retrieves similar past corrections for few-shot prompt injection."""

    def __init__(self):
        self._store = CorrectionStore()

    def index_correction(self, correction: Correction) -> None:
        """No-op — corrections are already stored in JSONL by CorrectionStore."""
        pass

    def get_relevant_exemplars(self, source_text: str,
                                field_type: Optional[str] = None,
                                draft_type: Optional[str] = None,
                                k: int = 3) -> list[Correction]:
        """Retrieve top-K most similar corrections using BM25.

        Args:
            source_text: The source text being processed (query).
            field_type: Optional filter by field path prefix.
            draft_type: Optional filter by draft type.
            k: Number of exemplars to return.

        Returns:
            List of Correction objects ranked by BM25 relevance.
        """
        # Collect candidate corrections
        draft_types = ([draft_type] if draft_type
                       else ["case_fact_summary", "title_review_summary",
                             "notice_summary", "document_checklist"])
        candidates: list[Correction] = []
        for dt in draft_types:
            try:
                candidates.extend(self._store.get_all_active(dt))
            except Exception as exc:
                logger.warning("Could not load corrections for %s: %s", dt, exc)

        if field_type:
            candidates = [c for c in candidates if c.field_path.startswith(field_type)]

        if not candidates:
            return []

        # Build BM25 corpus from source_ocr_chunk of each correction
        corpus = [_tokenise(c.source_ocr_chunk) for c in candidates]
        N = len(corpus)
        avgdl = sum(len(d) for d in corpus) / max(1, N)
        df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        query_tokens = _tokenise(source_text)
        if not query_tokens:
            return candidates[:k]

        scored = [
            (i, _bm25_score(query_tokens, corpus[i], df, N, avgdl))
            for i in range(N)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [candidates[i] for i, _ in scored[:k]]

    def format_exemplars_for_prompt(self, exemplars: list[Correction]) -> str:
        """Format exemplars as few-shot examples for prompt injection."""
        if not exemplars:
            return ""

        lines = ["PAST CORRECTIONS (use these to avoid similar errors):", ""]
        for i, ex in enumerate(exemplars, 1):
            lines.append(f"Example {i}:")
            lines.append(f"- Source text: \"{ex.source_ocr_chunk[:200]}\"")
            lines.append(f"- System generated: \"{ex.generated_text[:200]}\"")
            lines.append(f"- Operator corrected to: \"{ex.edited_text[:200]}\"")
            lines.append(f"- Field: {ex.field_path}")
            lines.append("")

        return "\n".join(lines)

    def rebuild_index(self) -> int:
        """No-op — no persistent index to rebuild."""
        return 0
