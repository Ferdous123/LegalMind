"""Firewall runner — orchestrates verification layers."""

import logging
import re
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

    # Citations like [E1], [E2] inside generated markdown
    _CITATION_RE = re.compile(r"\[E(\d+)\]")

    def verify_draft(self, draft_content: str, structured_fields: dict,
                     citation_map: dict, ocr_confidence: float = 1.0) -> list[VerificationResult]:
        """Run full verification on a generated draft.

        Verifies two things:
        1. Cited claim segments from the draft markdown (the user-facing
           output). Each sentence carrying a `[E1]` marker is checked against
           that specific citation's evidence text.
        2. Operator-meaningful structured fields (skipping `_cascade_meta`
           and similar internal diagnostics).

        Args:
            draft_content: The generated markdown draft text.
            structured_fields: Extracted structured fields dict.
            citation_map: Mapping from citation IDs to evidence metadata.
            ocr_confidence: Overall OCR confidence for the source document.

        Returns:
            List of VerificationResult — one per cited claim and per field.
        """
        results: list[VerificationResult] = []
        all_evidence = self._concatenate_evidence(citation_map)

        # --- Layer A: cited markdown claims (the visible draft) ---
        for idx, segment in enumerate(self._extract_cited_segments(draft_content)):
            claim_text = segment["text"]
            cited_ids = segment["citations"]

            # Evidence for THIS claim = union of its cited chunks' text. Falls
            # back to all evidence if the cited IDs aren't in the map.
            scoped_evidence_parts: list[str] = []
            for cid in cited_ids:
                meta = citation_map.get(cid) or citation_map.get(f"E{cid}") or {}
                if isinstance(meta, dict):
                    text = meta.get("text", "")
                    if text:
                        scoped_evidence_parts.append(text)
            scoped_evidence = " ".join(scoped_evidence_parts) if scoped_evidence_parts else all_evidence

            anchor_result = self._anchor.verify(claim=claim_text, evidence_text=scoped_evidence)
            confidence_level = (
                "HIGH" if anchor_result["status"] == "verified"
                else "MEDIUM" if anchor_result["status"] == "uncertain"
                else "LOW"
            )
            confidence_score = anchor_result.get("similarity", 0.0)
            if anchor_result["status"] == "verified":
                confidence_score = max(confidence_score, 0.9)

            final_status = self._combine_signals(
                anchor_status=anchor_result["status"],
                confidence_level=confidence_level,
            )
            results.append(VerificationResult(
                field_path=f"draft.claim[{idx}]",
                status=final_status,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                similarity=anchor_result["similarity"],
                evidence_snippet=anchor_result.get("evidence_snippet", ""),
                reasons=[f"cited: {', '.join(cited_ids)}"],
            ))

        # --- Layer B: operator-meaningful structured fields ---
        confidence_results = self._confidence.score_document(
            structured_fields, ocr_confidence
        )

        for field_name, conf in confidence_results.get("fields", {}).items():
            field_value = conf.get("_value", field_name)

            anchor_result = {"status": "uncertain", "similarity": 0.0, "evidence_snippet": ""}
            if all_evidence and field_value and len(str(field_value)) > 3:
                anchor_result = self._anchor.verify(
                    claim=str(field_value),
                    evidence_text=all_evidence,
                )

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

    def _extract_cited_segments(self, draft_text: str) -> list[dict]:
        """Split draft into sentence-level segments carrying [E_] citations.

        Only segments with at least one citation are returned. Structural
        lines (headers, table separators) are skipped.
        """
        if not draft_text:
            return []
        segments: list[dict] = []
        for line in draft_text.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("|---"):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", stripped):
                if not sentence.strip():
                    continue
                cites = self._CITATION_RE.findall(sentence)
                if not cites:
                    continue
                clean = self._CITATION_RE.sub("", sentence).strip()
                if len(clean) < 8:
                    continue
                segments.append({"text": clean, "citations": [f"E{n}" for n in cites]})
        return segments

    @staticmethod
    def _concatenate_evidence(citation_map: dict) -> str:
        """Concatenate all evidence texts for broad field verification."""
        parts = []
        for meta in citation_map.values():
            text = meta.get("text", "") if isinstance(meta, dict) else ""
            if text:
                parts.append(text)
        return " ".join(parts)

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
