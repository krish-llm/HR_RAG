"""Tests for shared schemas and multi-tenant document base models."""

from datetime import datetime
import pytest
from app.schemas.common import (
    ApiResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ServiceHealth,
)
from app.schemas.document import (
    BaseDocument,
    BaseDocumentCreate,
    BaseDocumentUpdate,
    SchoolScope,
)


def test_api_response_envelope():
    """Verify ApiResponse generic envelope."""
    payload = {"key": "value"}
    resp = ApiResponse[dict](data=payload, message="Success")

    assert resp.success is True
    assert resp.message == "Success"
    assert resp.data == payload
    assert isinstance(resp.timestamp, datetime)


def test_error_response_envelope():
    """Verify ErrorResponse structure."""
    detail = ErrorDetail(field="school_id", message="School ID is required")
    err = ErrorResponse(
        error_code="VALIDATION_ERROR",
        message="Invalid payload",
        details=[detail],
    )

    assert err.success is False
    assert err.error_code == "VALIDATION_ERROR"
    assert len(err.details) == 1
    assert err.details[0].field == "school_id"


def test_school_scope_validation():
    """Verify SchoolScope requires at least one scoping dimension."""
    # Successful cases
    scope1 = SchoolScope(school_id="SCH-001")
    assert scope1.school_id == "SCH-001"
    assert scope1.is_global is False

    scope2 = SchoolScope(group_id="GRP-100", state="Karnataka")
    assert scope2.group_id == "GRP-100"
    assert scope2.state == "Karnataka"

    scope_global = SchoolScope(is_global=True)
    assert scope_global.is_global is True

    # Empty scope should fail
    with pytest.raises(Exception):
        SchoolScope()


def test_base_document_canonical_scoping():
    """Verify BaseDocument uses canonical school hierarchy scope fields."""
    doc = BaseDocument(
        school_id="SCH-001",
        group_id="GRP-100",
        branch_id="BR-MAIN",
        state="Karnataka",
        metadata={"category": "leave_policy", "author": "HR"},
    )

    assert doc.id is not None
    assert doc.school_id == "SCH-001"
    assert doc.group_id == "GRP-100"
    assert doc.branch_id == "BR-MAIN"
    assert doc.state == "Karnataka"
    assert doc.is_global is False
    assert doc.version == 1
    assert doc.metadata["category"] == "leave_policy"
    assert isinstance(doc.created_at, datetime)
    assert isinstance(doc.updated_at, datetime)

    # Test scope property helper
    scope = doc.scope
    assert isinstance(scope, SchoolScope)
    assert scope.school_id == "SCH-001"
    assert scope.group_id == "GRP-100"


def test_base_document_global_policy():
    """Verify BaseDocument can represent global group-wide policies."""
    doc = BaseDocument(
        is_global=True,
        metadata={"category": "code_of_conduct"},
    )
    assert doc.is_global is True
    assert doc.school_id is None
    assert doc.version == 1


def test_base_document_validation_fails_without_scope():
    """Verify BaseDocument raises error when all scoping dimensions are omitted."""
    with pytest.raises(Exception):
        BaseDocument()
