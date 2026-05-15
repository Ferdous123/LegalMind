"""Source anchor verification — TRACE-style L0 n-gram verbatim match.

A claim is "verified" if it contains a 4-word n-gram present verbatim in
the source evidence text.  Fallback to token containment for shorter claims.

Thresholds (TRACE-aligned):
- n-gram hit OR containment >= 0.40: verified
- containment >= 0.20: uncertain
- below: unsupported
"""

import logging
import re

logger = logging.getLogger(__name__)

NGRAM_SIZE = 4


def _word_list(text: str) -> list[str]:
    return [w for w in re.split(r"\W+", text.lower()) if w]


def _tokens(text: str) -> set[str]:
    _STOP = {"the", "a", "an", "of", "in", "to", "and", "or", "is", "was",
              "are", "were", "be", "been", "has", "have", "had", "that", "this",
              "it", "for", "on", "at", "by", "with", "from", "as", "not"}
    words = re.split(r"\W+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOP}


def _has_ngram_overlap(claim: str, evidence: str, n: int = NGRAM_SIZE) -> bool:
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
    if not claim_tokens or not evidence_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


class SourceAnchorVerifier:
    """Verifies generated claims against source evidence via n-gram anchoring."""

    def __init__(self, containment_high: float = 0.40, containment_low: float = 0.20):
        self._high = containment_high
        self._low = containment_low

    def verify(self, claim: str, evidence_text: str) -> dict:
        """Verify a single claim against evidence.

        Returns dict with: status, similarity, claim, evidence_snippet.
        """
        if not claim.strip() or not evidence_text.strip():
            return {
                "status": "unsupported",
                "similarity": 0.0,
                "claim": claim,
                "evidence_snippet": "",
            }

        if _has_ngram_overlap(claim, evidence_text):
            return {
                "status": "verified",
                "similarity": 1.0,
                "claim": claim,
                "evidence_snippet": evidence_text[:200],
            }

        score = _containment(_tokens(claim), _tokens(evidence_text))

        if score >= self._high:
            status = "verified"
        elif score >= self._low:
            status = "uncertain"
        else:
            status = "unsupported"

        return {
            "status": status,
            "similarity": round(score, 4),
            "claim": claim,
            "evidence_snippet": evidence_text[:200],
        }
