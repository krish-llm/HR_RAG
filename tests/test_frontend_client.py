"""Unit tests for Frontend API Client (Phase 10)."""

from unittest.mock import MagicMock, patch
import httpx
import pytest

from frontend.api_client import APIClientError, HRPolicyAPIClient


@pytest.fixture
def api_client():
    """Instantiate a client pointing to a test base URL."""
    return HRPolicyAPIClient(base_url="http://test-server/api/v1", timeout=10.0)


def test_api_client_initialization(api_client):
    """Verify base URL stripping and timeout settings."""
    assert api_client.base_url == "http://test-server/api/v1"
    assert api_client.timeout == 10.0


def test_build_headers_with_tenant_scoping(api_client):
    """Verify that canonical tenant context headers are constructed properly."""
    school_ctx = {
        "school_id": "sch-01",
        "group_id": "grp-01",
        "branch_id": "br-01",
        "state": "Delhi",
    }
    headers = api_client._build_headers(school_ctx)

    assert headers["X-School-ID"] == "sch-01"
    assert headers["X-Group-ID"] == "grp-01"
    assert headers["X-Branch-ID"] == "br-01"
    assert headers["X-State"] == "Delhi"
    assert headers["Accept"] == "application/json"


def test_build_headers_empty(api_client):
    """Verify default headers when no school context is provided."""
    headers = api_client._build_headers(None)
    assert headers == {"Accept": "application/json"}


@patch("httpx.Client.get")
def test_check_health_success(mock_get, api_client):
    """Verify successful health check response parsing."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"success": True, "data": {"status": "healthy"}}
    mock_get.return_value = mock_resp

    res = api_client.check_health()
    assert res["success"] is True
    assert res["data"]["status"] == "healthy"


@patch("httpx.Client.get")
def test_check_health_connection_error(mock_get, api_client):
    """Verify connection error wraps into 503 APIClientError."""
    mock_get.side_effect = httpx.RequestError("Connection refused")

    with pytest.raises(APIClientError) as exc_info:
        api_client.check_health()

    assert exc_info.value.status_code == 503
    assert "Cannot connect to backend service" in str(exc_info.value)


@patch("httpx.Client.post")
def test_ask_question_success(mock_post, api_client):
    """Verify POST /query/answer client method."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "data": {
            "question": "What is the policy?",
            "answer": "Grounded answer text [Source 1].",
            "citations": [{"citation_label": "[Source 1]"}],
            "insufficient_evidence": False,
        },
    }
    mock_post.return_value = mock_resp

    res = api_client.ask_question(
        question="What is the policy?",
        school_context={"school_id": "sch-01"},
        top_k=5,
    )
    assert res["success"] is True
    assert "Grounded answer text" in res["data"]["answer"]


@patch("httpx.Client.post")
def test_non_200_error_envelope_handling(mock_post, api_client):
    """Verify standard ErrorResponse parsing on 404/400 errors."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 404
    mock_resp.json.return_value = {
        "success": False,
        "error_code": "RESOURCE_NOT_FOUND",
        "message": "Policy document not found",
        "details": None,
    }
    mock_post.return_value = mock_resp

    with pytest.raises(APIClientError) as exc_info:
        api_client.extract_document("non-existent-doc")

    assert exc_info.value.status_code == 404
    assert "[RESOURCE_NOT_FOUND]" in str(exc_info.value)
    assert "Policy document not found" in str(exc_info.value)


@patch("httpx.Client.post")
def test_document_pipeline_methods(mock_post, api_client):
    """Verify document upload, extract, process, chunk, and index methods."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"success": True, "data": {"status": "ok"}}
    mock_post.return_value = mock_resp

    upload_res = api_client.upload_document(b"%PDF-1.4...", "policy.pdf", {"school_id": "sch-01"})
    assert upload_res["success"] is True

    ext_res = api_client.extract_document("doc-123")
    assert ext_res["success"] is True

    proc_res = api_client.process_document("doc-123")
    assert proc_res["success"] is True

    chk_res = api_client.chunk_document("doc-123")
    assert chk_res["success"] is True

    idx_res = api_client.index_document("doc-123")
    assert idx_res["success"] is True


@patch("httpx.Client.get")
@patch("httpx.Client.delete")
def test_document_management_list_and_delete(mock_delete, mock_get, api_client):
    """Verify listing and deleting documents."""
    mock_get_resp = MagicMock(spec=httpx.Response)
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {
        "success": True,
        "data": {"items": [{"document_id": "doc-123", "document_name": "Test"}], "total": 1},
    }
    mock_get.return_value = mock_get_resp

    list_res = api_client.list_documents(school_id="sch-01")
    assert list_res["success"] is True
    assert len(list_res["data"]["items"]) == 1

    mock_del_resp = MagicMock(spec=httpx.Response)
    mock_del_resp.status_code = 200
    mock_del_resp.json.return_value = {"success": True, "data": {"deleted": True}}
    mock_delete.return_value = mock_del_resp

    del_res = api_client.delete_document("doc-123")
    assert del_res["success"] is True
    assert del_res["data"]["deleted"] is True
