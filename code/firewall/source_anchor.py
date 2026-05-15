"""Source anchor verification — ensures claims are supported by cited evidence."""

import logging

from code.llm_interface.model_manager import ModelManager

logger = logging.getLogger(__name__)


class SourceAnchorVerifier:
    """Verifies generated claims against source evidence using embedding similarity.

    Thresholds:
    - >= 0.75: verified (evidence clearly supports claim)
    - 0.50 - 0.75: uncertain (some support but not conclusive)
    - < 0.50: unsupported (evidence does not support claim)
    """

    def __init__(self, threshold_high: float = 0.75, threshold_low: float = 0.50):
        self._high = threshold_high
        self._low = threshold_low

    def verify(self, claim: str, evidence_text: str) -> dict:
        """Verify a single claim against evidence.

        Returns:
            dict with keys: status, similarity, claim, evidence_snippet
        """
        if not claim.strip() or not evidence_text.strip():
            return {
                "status": "unsupported",
                "similarity": 0.0,
                "claim": claim,
                "evidence_snippet": "",
            }

        try:
            mgr = ModelManager.instance()
            embeddings = mgr.embed([claim, evidence_text])
            similarity = self._cosine_sim(embeddings[0], embeddings[1])
        except Exception as e:
            logger.error("Source anchor verification failed: %s", e)
            return {
                "status": "uncertain",
                "similarity": 0.0,
                "claim": claim,
                "evidence_snippet": evidence_text[:100],
            }

        if similarity >= self._high:
            status = "verified"
        elif similarity >= self._low:
            status = "uncertain"
        else:
            status = "unsupported"

        return {
            "status": status,
            "similarity": round(similarity, 4),
            "claim": claim,
            "evidence_snippet": evidence_text[:200],
        }

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
