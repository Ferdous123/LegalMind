"""Firewall runner — orchestrates verification layers."""

import logging
from dataclasses import dataclass
from typing import Optional

from code.firewall.source_anchor import SourceAnchorVerifier
from code.firewall.confidence import ConfidenceScorer

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verification for a single field or claim."""
    field_path: str
    status: str  # verified, uncertain, unsupported, manual_review
    confidence_score: float
    confidence_level: str  # HIGH, MEDIUM, LOW, UNSUPPORTED
    similarity: float
    evidence_snippet: str
    reasons: list


class FirewallRunner:
    """Orchestrates multi-layer verification of generated drafts.

    Layer 1: Source anchor — verify claims cite real evidence
    Layer 2: Confidence scoring — score based on OCR quality, source spans, etc.
    Combined: produce final status per field
    """

    def __init__(self):
        self._anchor = SourceAnchorVerifier()
        self._confidence = ConfidenceScorer()

    def verify_draft(self, draft_content: str, structured_fields: dict,
                     citation_map: dict, ocr_confidence: float = 1.0) -> list[VerificationResult]:
        """Run full verification on a generated draft.

        Args:
            draft_content: The generated markdown draft text.
            structured_fields: Extracted structured fields dict.
            citation_map: Mapping from citation IDs to evidence metadata.
            ocr_confidence: Overall OCR confidence for the source document.

        Returns:
            List of VerificationResult for each verified claim/field.
        """
        results = []

        # Layer 1: Confidence scoring on structured fields
        confidence_results = self._confidence.score_document(
            structured_fields, ocr_confidence
        )

        # Layer 2: Source anchor verification for each field with evidence
        for field_name, conf in confidence_results.get("fields", {}).items():
            # Find relevant evidence for this field
            evidence_text = self._find_evidence_for_field(field_name, citation_map)

            anchor_result = {"status": "uncertain", "similarity": 0.0, "evidence_snippet": ""}
            if evidence_text:
                anchor_result = self._anchor.verify(
                    claim=field_name,
                    evidence_text=evidence_text,
                )

            # Combine signals
            final_status = self._combine_signals(
                anchor_status=anchor_result["status"],
                confidence_level=conf["level"],
            )

            results.append(VerificationResult(
                field_path=field_name,
                status=final_status,
                confidence_score=conf["score"],
                confidence_level=conf["level"],
                similarity=anchor_result["similarity"],
                evidence_snippet=anchor_result.get("evidence_snippet", ""),
                reasons=conf.get("reasons", []),
            ))

        return results

    def _find_evidence_for_field(self, field_name: str, citation_map: dict) -> str:
        """Find the most relevant evidence text for a field."""
        if not citation_map:
            return ""

        # Simple heuristic: return first evidence chunk text
        # In production, would match field content against evidence
        for cid, meta in citation_map.items():
            text = meta.get("text", "")
            if text:
                return text
        return ""

    @staticmethod
    def _combine_signals(anchor_status: str, confidence_level: str) -> str:
        """Combine anchor verification and confidence into final status.

        Rules:
        - Both verified/HIGH → verified
        - One uncertain → uncertain
        - Either unsupported → unsupported
        - LOW confidence alone → manual_review
        """
        if anchor_status == "verified" and confidence_level == "HIGH":
            return "verified"
        elif anchor_status == "unsupported" or confidence_level == "UNSUPPORTED":
            return "unsupported"
        elif confidence_level == "LOW":
            return "manual_review"
        elif anchor_status == "uncertain" or confidence_level == "MEDIUM":
            return "uncertain"
        else:
            return "verified"

    def get_summary(self, results: list[VerificationResult]) -> dict:
        """Produce a summary of verification results."""
        if not results:
            return {"total": 0, "verified": 0, "uncertain": 0, "unsupported": 0}

        return {
            "total": len(results),
            "verified": sum(1 for r in results if r.status == "verified"),
            "uncertain": sum(1 for r in results if r.status == "uncertain"),
            "unsupported": sum(1 for r in results if r.status == "unsupported"),
            "manual_review": sum(1 for r in results if r.status == "manual_review"),
            "overall_confidence": sum(r.confidence_score for r in results) / len(results),
        }
