"""Reusable API Client for communication between Streamlit UI and FastAPI backend."""

import logging
import os
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Custom exception raised when backend communication fails or returns an error."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Any = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class HRPolicyAPIClient:
    """Centralized HTTP client managing requests to the FastAPI backend with tenant scoping."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")).rstrip("/")
        self.timeout = float(timeout or os.getenv("API_TIMEOUT_SECONDS", "60.0"))

    def _build_headers(self, school_context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Construct request headers with canonical tenant scoping dimensions."""
        headers: Dict[str, str] = {
            "Accept": "application/json",
        }
        if school_context:
            if school_context.get("school_id"):
                headers["X-School-ID"] = str(school_context["school_id"])
            if school_context.get("group_id"):
                headers["X-Group-ID"] = str(school_context["group_id"])
            if school_context.get("branch_id"):
                headers["X-Branch-ID"] = str(school_context["branch_id"])
            if school_context.get("state"):
                headers["X-State"] = str(school_context["state"])
        return headers

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Parse response and handle standard error envelopes."""
        try:
            payload = response.json()
        except Exception:
            payload = {"message": response.text}

        if 200 <= response.status_code < 300:
            return payload

        # Extract structured error message if present in standard ErrorResponse envelope
        error_msg = payload.get("message") or f"HTTP {response.status_code} Error"
        error_code = payload.get("error_code") or "UNKNOWN_ERROR"
        details = payload.get("details") or payload

        logger.error("API error (%s): %s", error_code, error_msg)
        raise APIClientError(message=f"[{error_code}] {error_msg}", status_code=response.status_code, details=details)

    def check_health(self) -> Dict[str, Any]:
        """Verify FastAPI connectivity and health status."""
        url = f"{self.base_url}/health"
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(url)
                return self._handle_response(res)
        except httpx.RequestError as exc:
            raise APIClientError(
                f"Cannot connect to backend service at {self.base_url}. Ensure FastAPI is running.",
                status_code=503,
            ) from exc

    def ask_question(
        self,
        question: str,
        school_context: Optional[Dict[str, Any]] = None,
        employee_context: Optional[Dict[str, Any]] = None,
        as_of_date: Optional[str] = None,
        top_k: int = 10,
        token_budget: Optional[int] = 2000,
    ) -> Dict[str, Any]:
        """Submit question to POST /api/v1/query/answer for full grounded generation."""
        url = f"{self.base_url}/query/answer"
        headers = self._build_headers(school_context)
        body: Dict[str, Any] = {
            "question": question,
            "top_k": top_k,
            "token_budget": token_budget,
        }
        if employee_context:
            body["employee_context"] = employee_context
        if as_of_date:
            body["as_of_date"] = as_of_date

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=body, headers=headers)
                return self._handle_response(res)
        except httpx.RequestError as exc:
            raise APIClientError(f"Network error querying answer: {exc}", status_code=503) from exc

    def inspect_retrieval(
        self,
        question: str,
        school_context: Optional[Dict[str, Any]] = None,
        employee_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Query semantic retrieval directly via POST /api/v1/query/retrieve."""
        url = f"{self.base_url}/query/retrieve"
        headers = self._build_headers(school_context)
        body: Dict[str, Any] = {
            "question": question,
            "top_k": top_k,
        }
        if employee_context:
            body["employee_context"] = employee_context

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=body, headers=headers)
                return self._handle_response(res)
        except httpx.RequestError as exc:
            raise APIClientError(f"Network error inspecting retrieval: {exc}", status_code=503) from exc

    def inspect_context(
        self,
        question: str,
        school_context: Optional[Dict[str, Any]] = None,
        employee_context: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        token_budget: Optional[int] = 2000,
    ) -> Dict[str, Any]:
        """Query reranked context and source manifest via POST /api/v1/query/context."""
        url = f"{self.base_url}/query/context"
        headers = self._build_headers(school_context)
        body: Dict[str, Any] = {
            "question": question,
            "top_k": top_k,
            "token_budget": token_budget,
        }
        if employee_context:
            body["employee_context"] = employee_context

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=body, headers=headers)
                return self._handle_response(res)
        except httpx.RequestError as exc:
            raise APIClientError(f"Network error inspecting context: {exc}", status_code=503) from exc

    def upload_document(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Upload and ingest a PDF policy document via POST /api/v1/documents/upload."""
        url = f"{self.base_url}/documents/upload"
        files = {
            "file": (filename, file_bytes, "application/pdf"),
        }
        # Form fields
        data = {}
        for k, v in metadata.items():
            if v is not None:
                if isinstance(v, bool):
                    data[k] = "true" if v else "false"
                else:
                    data[k] = str(v)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, data=data, files=files)
                return self._handle_response(res)
        except httpx.RequestError as exc:
            raise APIClientError(f"Network error uploading document: {exc}", status_code=503) from exc

    def extract_document(self, document_id: str) -> Dict[str, Any]:
        """Trigger PDF extraction via POST /api/v1/documents/{document_id}/extract."""
        url = f"{self.base_url}/documents/{document_id}/extract"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url)
                return self._handle_response(res)
        except httpx.RequestError as exc:
            raise APIClientError(f"Network error extracting document: {exc}", status_code=503) from exc

    def process_document(self, document_id: str) -> Dict[str, Any]:
        """Trigger text cleaning/normalization via POST /api/v1/documents/{document_id}/process."""
        url = f"{self.base_url}/documents/{document_id}/process"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url)
                return self._handle_response(res)
        except httpx.RequestError as exc:
            raise APIClientError(f"Network error processing document: {exc}", status_code=503) from exc

    def chunk_document(self, document_id: str) -> Dict[str, Any]:
        """Trigger document chunking via POST /api/v1/documents/{document_id}/chunk."""
        url = f"{self.base_url}/documents/{document_id}/chunk"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url)
                return self._handle_response(res)
        except httpx.RequestError as exc:
            raise APIClientError(f"Network error chunking document: {exc}", status_code=503) from exc

    def index_document(self, document_id: str) -> Dict[str, Any]:
        """Trigger embedding and vector indexing via POST /api/v1/documents/{document_id}/index."""
        url = f"{self.base_url}/documents/{document_id}/index"
        try:
            with httpx.Client(timeout=self.timeout * 2) as client:
                res = client.post(url)
                return self._handle_response(res)
        except httpx.RequestError as exc:
            raise APIClientError(f"Network error indexing document: {exc}", status_code=503) from exc

    def get_indexing_status(self, document_id: str) -> Dict[str, Any]:
        """Fetch indexing status via GET /api/v1/documents/{document_id}/index-status."""
        url = f"{self.base_url}/documents/{document_id}/index-status"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(url)
                return self._handle_response(res)
        except httpx.RequestError as exc:
            raise APIClientError(f"Network error fetching indexing status: {exc}", status_code=503) from exc

    def list_documents(
        self,
        school_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        """List documents via GET /api/v1/documents."""
        url = f"{self.base_url}/documents"
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if school_id:
            params["school_id"] = school_id

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(url, params=params)
                return self._handle_response(res)
        except httpx.RequestError as exc:
            raise APIClientError(f"Network error listing documents: {exc}", status_code=503) from exc

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete document via DELETE /api/v1/documents/{document_id}."""
        url = f"{self.base_url}/documents/{document_id}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.delete(url)
                return self._handle_response(res)
        except httpx.RequestError as exc:
            raise APIClientError(f"Network error deleting document: {exc}", status_code=503) from exc
