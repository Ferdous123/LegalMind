"""Grounding verifier — checks that generated claims cite actual evidence.

Uses TRACE-style source anchoring: a claim is "verified" if it contains a
5-word n-gram that appears verbatim in the cited evidence, OR if token
containment exceeds threshold.  Segments without citations are skipped
(they are either structural or deliberate absence statements).
"""

import logging
import re

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[E(\d+)\]")

_STOP = {"the", "a", "an", "of", "in", "to", "and", "or", "is", "was",
         "are", "were", "be", "been", "has", "have", "had", "that", "this",
         "it", "for", "on", "at", "by", "with", "from", "as", "not"}

NGRAM_SIZE = 4


def _tokens(text: str) -> set[str]:
    words = re.split(r"\W+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOP}


def _word_list(text: str) -> list[str]:
    return [w for w in re.split(r"\W+", text.lower()) if w]


def _has_ngram_overlap(claim: str, evidence: str, n: int = NGRAM_SIZE) -> bool:
    """Return True if any n-gram from the claim appears verbatim in the evidence."""
    claim_words = _word_list(claim)
    if len(claim_words) < n:
        return claim.lower().strip() in evidence.lower()
    evidence_lower = evidence.lower()
    for i in range(len(claim_words) - n + 1):
        ngram = " ".join(claim_words[i:i + n])
        if ngram in evidence_lower:
            return True
    return False


def _containment(claim_tokens: set, evidence_tokens: set) -> float:
    """What fraction of claim tokens appear in the evidence (asymmetric)."""
    if not claim_tokens or not evidence_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


class GroundingVerifier:
    """Verifies that generated text is grounded in cited evidence."""

    def __init__(self, containment_threshold: float = 0.40):
        self._containment_threshold = containment_threshold

    def verify_draft_citations(self, draft_text: str, citation_map: dict) -> dict:
        """Verify all citations in a draft against the evidence.

        Only scores segments that carry citation markers [E1], [E2], etc.
        Segments without citations are skipped.
        """
        segments = self._extract_cited_segments(draft_text)
        results = {}

        for segment in segments:
            citation_ids = segment["citations"]
            claim_text = segment["text"]

            if not citation_ids:
                continue

            if self._is_structural(claim_text):
                continue

            best_status = "unsupported"
            for cid in citation_ids:
                evidence = citation_map.get(cid, {})
                evidence_text = evidence.get("text", "")
                if not evidence_text:
                    continue

                status = self._verify_single(claim_text, evidence_text)
                if status == "verified":
                    best_status = "verified"
                    break
                elif status == "uncertain" and best_status != "verified":
                    best_status = "uncertain"

            results[claim_text[:50]] = best_status

        return results

    def verify_single_claim(self, claim: str, evidence_text: str) -> str:
        """Verify a single claim against evidence text."""
        return self._verify_single(claim, evidence_text)

    def _verify_single(self, claim: str, evidence: str) -> str:
        """TRACE-style L0 anchoring: n-gram verbatim match OR high containment."""
        if _has_ngram_overlap(claim, evidence):
            return "verified"
        score = _containment(_tokens(claim), _tokens(evidence))
        if score >= self._containment_threshold:
            return "verified"
        elif score >= 0.20:
            return "uncertain"
        return "unsupported"

    @staticmethod
    def _is_structural(text: str) -> bool:
        """Return True if text is a markdown header, label, or table divider."""
        stripped = text.strip().strip("-").strip()
        if stripped.startswith("#"):
            return True
        if stripped.startswith("|") and stripped.endswith("|"):
            return True
        if len(_tokens(stripped)) < 3:
            return True
        if re.match(r"^(Full legal name|Role description|Represented by|Event|Date|"
                    r"Confidence|Citation|Plaintiffs|Defendants|Other Parties|"
                    r"Not stated)", stripped):
            return True
        return False

    def _extract_cited_segments(self, text: str) -> list[dict]:
        segments = []
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            sentences = re.split(r"(?<=[.!?])\s+", line)
            for sentence in sentences:
                if not sentence.strip():
                    continue
                citations = CITATION_PATTERN.findall(sentence)
                citation_ids = [f"E{num}" for num in citations]
                clean_text = CITATION_PATTERN.sub("", sentence).strip()
                if clean_text:
                    segments.append({"text": clean_text, "citations": citation_ids})
        return segments

    def get_uncited_claims(self, draft_text: str) -> list[str]:
        segments = self._extract_cited_segments(draft_text)
        return [s["text"] for s in segments if not s["citations"] and len(s["text"]) > 20]
