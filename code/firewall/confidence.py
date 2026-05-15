"""Confidence scoring — assigns confidence levels to extracted fields."""

import logging

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Scores confidence of extracted fields based on multiple signals.

    Signals used:
    - OCR confidence (from extraction phase)
    - Source span length (longer = more context = higher confidence)
    - Evidence similarity (from retrieval)
    - Field completeness (null/empty = low confidence)
    """

    def score_field(self, field_value: str, source_span: str = "",
                    ocr_confidence: float = 1.0,
                    retrieval_similarity: float = 0.0) -> dict:
        """Score confidence for a single extracted field.

        Returns:
            dict with: level (HIGH/MEDIUM/LOW/UNSUPPORTED), score (0.0-1.0), reasons
        """
        if not field_value or field_value.strip() in ("", "null", "None", "N/A"):
            return {
                "level": "UNSUPPORTED",
                "score": 0.0,
                "reasons": ["Field is empty or null"],
            }

        scores = []
        reasons = []

        # Signal 1: OCR confidence
        scores.append(ocr_confidence)
        if ocr_confidence < 0.5:
            reasons.append("Low OCR confidence on source page")

        # Signal 2: Source span availability
        if source_span and len(source_span) > 10:
            scores.append(0.9)
        elif source_span:
            scores.append(0.6)
            reasons.append("Short source span")
        else:
            scores.append(0.3)
            reasons.append("No source span provided")

        # Signal 3: Retrieval similarity (if available)
        if retrieval_similarity > 0:
            scores.append(retrieval_similarity)
            if retrieval_similarity < 0.5:
                reasons.append("Low retrieval similarity")

        # Signal 4: Value quality heuristics
        if "[illegible]" in field_value:
            scores.append(0.3)
            reasons.append("Contains illegible markers")
        elif "unclear" in field_value.lower() or "not clearly" in field_value.lower():
            scores.append(0.4)
            reasons.append("Value indicates uncertainty")
        else:
            scores.append(0.9)

        # Aggregate
        avg_score = sum(scores) / len(scores) if scores else 0.0

        if avg_score >= 0.8:
            level = "HIGH"
        elif avg_score >= 0.5:
            level = "MEDIUM"
        elif avg_score > 0.0:
            level = "LOW"
        else:
            level = "UNSUPPORTED"

        return {
            "level": level,
            "score": round(avg_score, 3),
            "reasons": reasons if reasons else ["All signals indicate high confidence"],
            "_value": field_value,
        }

    def score_document(self, fields: dict, ocr_confidence: float = 1.0) -> dict:
        """Score confidence across all fields in a document.

        Returns summary with per-field scores and overall assessment.
        """
        results = {}
        total_score = 0.0
        count = 0

        for field_name, field_data in self._flatten_fields(fields):
            value = ""
            source_span = ""

            if isinstance(field_data, dict):
                value = str(field_data.get("value", field_data.get("name", "")))
                source_span = field_data.get("source_span", "")
            elif isinstance(field_data, str):
                value = field_data

            score = self.score_field(value, source_span, ocr_confidence)
            results[field_name] = score
            total_score += score["score"]
            count += 1

        overall = total_score / count if count > 0 else 0.0

        return {
            "fields": results,
            "overall_confidence": round(overall, 3),
            "total_fields": count,
            "high_confidence_count": sum(1 for r in results.values() if r["level"] == "HIGH"),
            "low_confidence_count": sum(1 for r in results.values() if r["level"] in ("LOW", "UNSUPPORTED")),
        }

    def _flatten_fields(self, fields: dict, prefix: str = "") -> list[tuple]:
        """Recursively flatten nested field dict into (name, value) pairs."""
        items = []
        for key, value in fields.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and not any(k in value for k in ("value", "name", "source_span")):
                items.extend(self._flatten_fields(value, full_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        items.extend(self._flatten_fields(item, f"{full_key}[{i}]"))
                    else:
                        items.append((f"{full_key}[{i}]", item))
            else:
                items.append((full_key, value))
        return items
