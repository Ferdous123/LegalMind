"""Grounding verifier — checks that generated claims cite actual evidence.

Scans draft output for citation markers [E1], [E2], etc. and verifies
that each cited passage actually supports the claim being made.
"""

import logging
import re
from typing import Optional

from code.llm_interface.model_manager import ModelManager

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[E(\d+)\]")


class GroundingVerifier:
    """Verifies that generated text is grounded in cited evidence."""

    def __init__(self, similarity_threshold_high: float = 0.75,
                 similarity_threshold_low: float = 0.5):
        self._high_threshold = similarity_threshold_high
        self._low_threshold = similarity_threshold_low

    def verify_draft_citations(self, draft_text: str,
                                citation_map: dict) -> dict:
        """Verify all citations in a draft against the evidence.

        Returns dict mapping each claim segment to verification status.
        """
        segments = self._extract_cited_segments(draft_text)
        results = {}

        for segment in segments:
            citation_ids = segment["citations"]
            claim_text = segment["text"]

            if not citation_ids:
                results[claim_text[:50]] = "unsupported"
                continue

            best_status = "unsupported"
            for cid in citation_ids:
                evidence = citation_map.get(cid, {})
                evidence_text = evidence.get("text", "")
                if not evidence_text:
                    continue

                status = self._verify_single_citation(claim_text, evidence_text)
                if status == "verified":
                    best_status = "verified"
                    break
                elif status == "uncertain" and best_status != "verified":
                    best_status = "uncertain"

            results[claim_text[:50]] = best_status

        return results

    def verify_single_claim(self, claim: str, evidence_text: str) -> str:
        """Verify a single claim against evidence text.

        Returns: "verified", "uncertain", or "unsupported"
        """
        return self._verify_single_citation(claim, evidence_text)

    def _verify_single_citation(self, claim: str, evidence: str) -> str:
        """Check if evidence supports the claim using embedding similarity."""
        try:
            mgr = ModelManager.instance()
            embeddings = mgr.embed([claim, evidence])
            similarity = self._cosine_similarity(embeddings[0], embeddings[1])

            if similarity >= self._high_threshold:
                return "verified"
            elif similarity >= self._low_threshold:
                return "uncertain"
            else:
                return "unsupported"
        except Exception as e:
            logger.warning("Grounding verification failed: %s", e)
            return "uncertain"

    def _extract_cited_segments(self, text: str) -> list[dict]:
        """Extract text segments with their citation references."""
        segments = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            if not sentence.strip():
                continue
            citations = CITATION_PATTERN.findall(sentence)
            citation_ids = [f"E{num}" for num in citations]
            clean_text = CITATION_PATTERN.sub("", sentence).strip()
            if clean_text:
                segments.append({
                    "text": clean_text,
                    "citations": citation_ids,
                })

        return segments

    def get_uncited_claims(self, draft_text: str) -> list[str]:
        """Return list of sentences in the draft that have no citations."""
        segments = self._extract_cited_segments(draft_text)
        return [s["text"] for s in segments if not s["citations"]
                and len(s["text"]) > 20]  # skip short fragments

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
