"""Tests for application health and root endpoints."""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Verify root endpoint returns basic API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "HR Policy RAG API"
    assert data["status"] == "online"
    assert "version" in data
    assert "health_check" in data


def test_health_check_structure(client: TestClient):
    """Verify /health endpoint returns the standardized HealthResponse structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]
    assert "app_name" in data
    assert "environment" in data
    assert "version" in data
    assert "services" in data
    assert "mongodb" in data["services"]
    assert "chromadb" in data["services"]
    assert "timestamp" in data


def test_health_check_v1_alias(client: TestClient):
    """Verify /api/v1/health is routed and functions identically."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "services" in data
    assert "chromadb" in data["services"]


@patch("app.db.mongo.mongo_manager.ping", new_callable=AsyncMock)
def test_health_check_fully_healthy(mock_ping: AsyncMock, client: TestClient):
    """Verify status is 'healthy' when all backend services are responsive."""
    mock_ping.return_value = True

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["services"]["mongodb"]["status"] == "healthy"
    assert data["services"]["chromadb"]["status"] == "healthy"


@patch("app.db.mongo.mongo_manager.ping", new_callable=AsyncMock)
def test_health_check_degraded_when_mongo_fails(mock_ping: AsyncMock, client: TestClient):
    """Verify status reports 'degraded' gracefully when MongoDB is unavailable."""
    mock_ping.return_value = False

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "degraded"
    assert data["services"]["mongodb"]["status"] == "unhealthy"
