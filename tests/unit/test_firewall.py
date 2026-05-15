"""Unit tests for code.firewall.runner.FirewallRunner and VerificationResult.

SourceAnchorVerifier (which calls ModelManager) is mocked wherever needed.
Tests run fast (< 1 s each).
"""

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# test_firewall_imports
# ---------------------------------------------------------------------------

def test_firewall_imports():
    """Verify FirewallRunner and VerificationResult can be imported cleanly."""
    from code.firewall.runner import FirewallRunner, VerificationResult  # noqa: F401
    assert FirewallRunner is not None
    assert VerificationResult is not None


# ---------------------------------------------------------------------------
# test_verify_draft_empty_fields
# ---------------------------------------------------------------------------

def test_verify_draft_empty_fields():
    """verify_draft with empty structured_fields must return an empty list.

    When there are no fields to verify, the firewall should produce no results
    rather than raising or returning spurious entries.
    """
    from code.firewall.runner import FirewallRunner

    mock_anchor = MagicMock()
    mock_confidence = MagicMock()
    # score_document with empty fields returns empty fields dict
    mock_confidence.score_document.return_value = {
        "fields": {},
        "overall_confidence": 0.0,
        "total_fields": 0,
        "high_confidence_count": 0,
        "low_confidence_count": 0,
    }

    with (
        patch("code.firewall.runner.SourceAnchorVerifier", return_value=mock_anchor),
        patch("code.firewall.runner.ConfidenceScorer", return_value=mock_confidence),
    ):
        runner = FirewallRunner()
        results = runner.verify_draft(
            draft_content="",
            structured_fields={},
            citation_map={},
            ocr_confidence=1.0,
        )

    assert results == [], f"Expected [] for empty fields, got {results}"


# ---------------------------------------------------------------------------
# test_get_summary_all_verified
# ---------------------------------------------------------------------------

def test_get_summary_all_verified():
    """get_summary over all-verified results must report correct counts.

    Builds 3 VerificationResult objects all with status='verified' and
    checks that the summary dict has verified=3 and uncertain=0.
    """
    from code.firewall.runner import FirewallRunner, VerificationResult

    mock_anchor = MagicMock()
    mock_confidence = MagicMock()

    with (
        patch("code.firewall.runner.SourceAnchorVerifier", return_value=mock_anchor),
        patch("code.firewall.runner.ConfidenceScorer", return_value=mock_confidence),
    ):
        runner = FirewallRunner()

    verified_results = [
        VerificationResult(
            field_path=f"field_{i}",
            status="verified",
            confidence_score=0.95,
            confidence_level="HIGH",
            similarity=0.88,
            evidence_snippet="some evidence text",
            reasons=["All signals indicate high confidence"],
        )
        for i in range(3)
    ]

    summary = runner.get_summary(verified_results)

    assert summary["total"] == 3, f"total should be 3, got {summary['total']}"
    assert summary["verified"] == 3, f"verified should be 3, got {summary['verified']}"
    assert summary["uncertain"] == 0, f"uncertain should be 0, got {summary['uncertain']}"
    assert summary["unsupported"] == 0
    assert "overall_confidence" in summary


# ---------------------------------------------------------------------------
# test_get_summary_empty_results
# ---------------------------------------------------------------------------

def test_get_summary_empty_results():
    """get_summary over empty results list must return all-zero counts."""
    from code.firewall.runner import FirewallRunner

    mock_anchor = MagicMock()
    mock_confidence = MagicMock()

    with (
        patch("code.firewall.runner.SourceAnchorVerifier", return_value=mock_anchor),
        patch("code.firewall.runner.ConfidenceScorer", return_value=mock_confidence),
    ):
        runner = FirewallRunner()

    summary = runner.get_summary([])
    assert summary["total"] == 0
    assert summary["verified"] == 0


# ---------------------------------------------------------------------------
# test_combine_signals
# ---------------------------------------------------------------------------

def test_combine_signals():
    """_combine_signals must return correct status for each combination.

    Rule table (from runner.py docstring):
    - verified + HIGH      → verified
    - * + UNSUPPORTED      → unsupported
    - unsupported + *      → unsupported
    - * + LOW              → manual_review
    - uncertain + MEDIUM   → uncertain
    - verified + MEDIUM    → uncertain
    """
    from code.firewall.runner import FirewallRunner

    # (anchor_status, confidence_level, expected_final_status)
    cases = [
        ("verified", "HIGH", "verified"),
        ("verified", "MEDIUM", "uncertain"),
        ("verified", "LOW", "manual_review"),
        ("uncertain", "HIGH", "uncertain"),
        ("uncertain", "MEDIUM", "uncertain"),
        ("uncertain", "LOW", "manual_review"),
        ("unsupported", "HIGH", "unsupported"),
        ("unsupported", "MEDIUM", "unsupported"),
        ("unsupported", "LOW", "unsupported"),
        ("verified", "UNSUPPORTED", "unsupported"),
    ]

    for anchor, conf_level, expected in cases:
        result = FirewallRunner._combine_signals(
            anchor_status=anchor,
            confidence_level=conf_level,
        )
        assert result == expected, (
            f"_combine_signals({anchor!r}, {conf_level!r}) → {result!r}, "
            f"expected {expected!r}"
        )


# ---------------------------------------------------------------------------
# test_verify_draft_with_fields
# ---------------------------------------------------------------------------

def test_verify_draft_with_fields():
    """verify_draft with two structured fields should return 2 VerificationResults."""
    from code.firewall.runner import FirewallRunner, VerificationResult

    structured_fields = {
        "defendant": "Marcus Bell",
        "plaintiff": "Alice Corp",
    }

    mock_anchor = MagicMock()
    mock_anchor.verify.return_value = {
        "status": "verified",
        "similarity": 0.82,
        "evidence_snippet": "Marcus Bell filed a motion",
    }

    mock_confidence = MagicMock()
    mock_confidence.score_document.return_value = {
        "fields": {
            "defendant": {"level": "HIGH", "score": 0.9, "reasons": []},
            "plaintiff": {"level": "HIGH", "score": 0.9, "reasons": []},
        },
        "overall_confidence": 0.9,
        "total_fields": 2,
        "high_confidence_count": 2,
        "low_confidence_count": 0,
    }

    citation_map = {
        "E1": {
            "text": "Marcus Bell filed a motion",
            "document_id": "doc_001",
            "page_number": 1,
            "char_start": 0,
            "char_end": 30,
            "similarity_score": 0.9,
        }
    }

    with (
        patch("code.firewall.runner.SourceAnchorVerifier", return_value=mock_anchor),
        patch("code.firewall.runner.ConfidenceScorer", return_value=mock_confidence),
    ):
        runner = FirewallRunner()
        results = runner.verify_draft(
            draft_content="Defendant: Marcus Bell. Plaintiff: Alice Corp.",
            structured_fields=structured_fields,
            citation_map=citation_map,
            ocr_confidence=0.95,
        )

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    for r in results:
        assert isinstance(r, VerificationResult)
        assert r.field_path in ("defendant", "plaintiff")
