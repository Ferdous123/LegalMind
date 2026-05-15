"""Output format templates — structures draft output for display."""

from typing import Optional


DRAFT_TYPE_LABELS = {
    "case_fact_summary": "Case Fact Summary",
    "title_review_summary": "Title Review Summary",
    "notice_summary": "Notice-Related Summary",
    "document_checklist": "Document Checklist",
}


def get_draft_label(draft_type: str) -> str:
    """Get human-readable label for a draft type."""
    return DRAFT_TYPE_LABELS.get(draft_type, draft_type.replace("_", " ").title())


def get_supported_types() -> list[dict]:
    """Return list of supported draft types with labels."""
    return [{"id": k, "label": v} for k, v in DRAFT_TYPE_LABELS.items()]


def format_confidence_badge(confidence: float) -> str:
    """Return HTML badge class for a confidence level."""
    if confidence >= 0.8:
        return "badge-verified"
    elif confidence >= 0.5:
        return "badge-uncertain"
    else:
        return "badge-unsupported"


def format_verification_status(status: str) -> dict:
    """Return display info for a verification status."""
    status_map = {
        "verified": {"label": "Verified", "class": "text-emerald-600", "icon": "check-circle"},
        "uncertain": {"label": "Uncertain", "class": "text-amber-600", "icon": "alert-circle"},
        "unsupported": {"label": "Unsupported", "class": "text-red-600", "icon": "x-circle"},
        "manual_review": {"label": "Needs Review", "class": "text-blue-600", "icon": "edit"},
    }
    return status_map.get(status, status_map["uncertain"])
