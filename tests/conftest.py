"""Pytest fixtures and test environment configuration."""

import os
import shutil
import tempfile
from typing import Generator
import pytest
from fastapi.testclient import TestClient

# Set testing environment before importing app settings
os.environ["APP_ENV"] = "test"
os.environ["DEBUG"] = "true"
os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
os.environ["MONGODB_DB_NAME"] = "hr_policy_rag_test"
os.environ["MONGODB_TIMEOUT_MS"] = "1000"

from app.config import Settings, get_settings
from app.db.chroma import chroma_service
from app.main import app


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Return test application settings."""
    return get_settings()


@pytest.fixture(autouse=True)
def temp_chroma_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for ChromaDB during tests to isolate test data."""
    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    chroma_service.initialize(persist_directory=temp_dir)
    yield temp_dir
    chroma_service.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provide a FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client
