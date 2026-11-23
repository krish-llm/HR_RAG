"""Tests for centralized exception handling and custom errors."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.core.exceptions import (
    AppException,
    DatabaseConnectionException,
    NotFoundException,
    TenantNotFoundException,
)
from app.core.handlers import (
    app_exception_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)


@pytest.fixture
def error_test_app():
    """Create a minimal FastAPI test app to verify error handlers."""
    test_app = FastAPI()
    test_app.add_exception_handler(AppException, app_exception_handler)

    @test_app.get("/trigger-not-found")
    def trigger_not_found():
        raise NotFoundException(message="Policy document not found")

    @test_app.get("/trigger-tenant-not-found")
    def trigger_tenant_not_found():
        raise TenantNotFoundException(tenant_id="tenant-xyz")

    @test_app.get("/trigger-db-error")
    def trigger_db_error():
        raise DatabaseConnectionException(message="Database ping timed out")

    return test_app


def test_not_found_exception_handling(error_test_app):
    """Verify 404 NotFoundException returns structured ErrorResponse."""
    client = TestClient(error_test_app)
    response = client.get("/trigger-not-found")
    assert response.status_code == 404
    data = response.json()

    assert data["success"] is False
    assert data["error_code"] == "RESOURCE_NOT_FOUND"
    assert "Policy document not found" in data["message"]
    assert "timestamp" in data


def test_tenant_not_found_exception_handling(error_test_app):
    """Verify TenantNotFoundException returns 404 with tenant details."""
    client = TestClient(error_test_app)
    response = client.get("/trigger-tenant-not-found")
    assert response.status_code == 404
    data = response.json()

    assert data["success"] is False
    assert data["error_code"] == "TENANT_NOT_FOUND"
    assert "tenant-xyz" in data["message"]
    assert len(data["details"]) == 1
    assert data["details"][0]["field"] == "tenant_id"


def test_database_connection_exception_handling(error_test_app):
    """Verify DatabaseConnectionException returns 503."""
    client = TestClient(error_test_app)
    response = client.get("/trigger-db-error")
    assert response.status_code == 503
    data = response.json()

    assert data["success"] is False
    assert data["error_code"] == "DATABASE_UNAVAILABLE"


def test_standard_404_handler(client: TestClient):
    """Verify nonexistent route triggers standard 404 ErrorResponse."""
    response = client.get("/nonexistent-endpoint-path")
    assert response.status_code == 404
    data = response.json()

    assert data["success"] is False
    assert data["error_code"] == "HTTP_404"
