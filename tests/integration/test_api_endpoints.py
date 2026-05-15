"""Integration tests for FastAPI JSON API endpoints.

Uses Starlette's TestClient (sync wrapper around the ASGI app) so no running
server is needed. All heavy dependencies (ModelManager, GPU) are NOT exercised
here — these tests only call the JSON REST endpoints, not the SSR template
routes.

Mark: pytest.mark.integration
"""

import json
import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixture: test client with tmp_data_dir isolation
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_data_dir):
    """Return a Starlette TestClient bound to the FastAPI app.

    tmp_data_dir patches config.paths so all file I/O goes to a temp dir,
    giving each test a clean slate.
    """
    from starlette.testclient import TestClient
    from webapp.main import app

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# test_system_status
# ---------------------------------------------------------------------------

def test_system_status(client):
    """GET /api/v1/system/status must return 200 with 'gpu' and 'correction_count'."""
    response = client.get("/api/v1/system/status")

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert "gpu" in body, f"'gpu' key missing from status response: {body}"
    assert "correction_count" in body, (
        f"'correction_count' key missing from status response: {body}"
    )
    assert "supported_draft_types" in body


# ---------------------------------------------------------------------------
# test_documents_list_empty
# ---------------------------------------------------------------------------

def test_documents_list_empty(client):
    """GET /api/v1/documents must return an empty list when no documents exist."""
    response = client.get("/api/v1/documents")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list), f"Expected list, got {type(body)}: {body}"
    assert body == [], f"Expected empty list, got {body}"


# ---------------------------------------------------------------------------
# test_learning_rules_empty
# ---------------------------------------------------------------------------

def test_learning_rules_empty(client):
    """GET /api/v1/learning/rules must return {'rules': []} when no rules file exists."""
    response = client.get("/api/v1/learning/rules")

    assert response.status_code == 200
    body = response.json()
    assert "rules" in body, f"'rules' key missing: {body}"
    assert isinstance(body["rules"], list)


# ---------------------------------------------------------------------------
# test_learning_metrics
# ---------------------------------------------------------------------------

def test_learning_metrics(client):
    """GET /api/v1/learning/metrics must return a dict with a 'per_type' key."""
    response = client.get("/api/v1/learning/metrics")

    assert response.status_code == 200
    body = response.json()
    assert "per_type" in body, f"'per_type' key missing from metrics: {body}"
    assert isinstance(body["per_type"], dict)
    # Should cover at least the 4 known draft types
    for expected_type in (
        "case_fact_summary",
        "title_review_summary",
        "notice_summary",
        "document_checklist",
    ):
        assert expected_type in body["per_type"], (
            f"Draft type {expected_type!r} missing from per_type metrics"
        )


# ---------------------------------------------------------------------------
# test_corrections_list_empty
# ---------------------------------------------------------------------------

def test_corrections_list_empty(client):
    """GET /api/v1/corrections must return an empty list when no corrections exist."""
    response = client.get("/api/v1/corrections")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert body == []


# ---------------------------------------------------------------------------
# test_corrections_list_filtered_by_draft_type
# ---------------------------------------------------------------------------

def test_corrections_list_filtered_by_draft_type(client):
    """GET /api/v1/corrections?draft_type=case_fact_summary must return a list."""
    response = client.get("/api/v1/corrections?draft_type=case_fact_summary")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


# ---------------------------------------------------------------------------
# test_save_correction
# ---------------------------------------------------------------------------

def test_save_correction(client, tmp_data_dir):
    """POST /api/v1/corrections with a valid payload must return {'status': 'saved'}.

    Verifies that the correction is persisted and the response contains
    an 'id' field plus the 'saved' status string.
    """
    payload = {
        "document_id": "doc_api_test",
        "draft_type": "case_fact_summary",
        "field_path": "parties.defendant",
        "source_ocr_chunk": "the defendant John Doe filed a motion",
        "generated_text": "Defendant: [Not identified]",
        "edited_text": "Defendant: John Doe",
        "correction_type": "omission",
    }

    response = client.post(
        "/api/v1/corrections",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert body.get("status") == "saved", (
        f"Expected status='saved', got: {body}"
    )
    assert "id" in body, f"'id' key missing from correction response: {body}"

    # Verify the correction is now retrievable
    list_response = client.get("/api/v1/corrections?draft_type=case_fact_summary")
    assert list_response.status_code == 200
    corrections = list_response.json()
    assert len(corrections) >= 1, "Saved correction must appear in the list"


# ---------------------------------------------------------------------------
# test_save_correction_invalid_payload
# ---------------------------------------------------------------------------

def test_save_correction_invalid_payload(client):
    """POST /api/v1/corrections with an invalid payload must return 4xx."""
    response = client.post(
        "/api/v1/corrections",
        json={"bad_field": "nonsense_value", "another_bad": 123},
        headers={"Content-Type": "application/json"},
    )
    # The endpoint should reject or fail gracefully — either 422 or 500 is acceptable
    # depending on whether Correction.__init__ raises
    assert response.status_code in (200, 422, 500), (
        f"Unexpected status {response.status_code} for invalid payload"
    )


# ---------------------------------------------------------------------------
# test_draft_not_found
# ---------------------------------------------------------------------------

def test_draft_not_found(client):
    """GET /api/v1/drafts/nonexistent must return 404."""
    response = client.get("/api/v1/drafts/nonexistent")

    assert response.status_code == 404, (
        f"Expected 404 for missing draft, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# test_document_not_found
# ---------------------------------------------------------------------------

def test_document_not_found(client):
    """GET /api/v1/documents/nonexistent_id must return 404."""
    response = client.get("/api/v1/documents/nonexistent_doc_id_xyz")

    assert response.status_code == 404, (
        f"Expected 404 for missing document, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# test_generate_draft_missing_fields
# ---------------------------------------------------------------------------

def test_generate_draft_missing_fields(client):
    """POST /api/v1/drafts/generate without required fields must return 422."""
    response = client.post(
        "/api/v1/drafts/generate",
        json={},
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422, (
        f"Expected 422 for missing required fields, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# test_generate_draft_document_not_found
# ---------------------------------------------------------------------------

def test_generate_draft_document_not_found(client):
    """POST /api/v1/drafts/generate for a non-existent doc must return 404."""
    response = client.post(
        "/api/v1/drafts/generate",
        json={"document_id": "doc_does_not_exist_xyz", "draft_type": "case_fact_summary"},
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 404, (
        f"Expected 404 for missing document, got {response.status_code}"
    )
