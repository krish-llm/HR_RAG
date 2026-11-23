"""Tests for Embedding Generation & ChromaDB Vector Indexing (Phase 5)."""

from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
import pytest
from starlette.testclient import TestClient

from app.core.exceptions import (
    EmbeddingException,
    IndexingException,
    NotFoundException,
    ProcessingException,
)
from app.db.chroma import ChromaDBService
from app.main import app
from app.modules.chunking.schemas import DocumentChunk
from app.modules.indexing.provider import (
    AzureOpenAIEmbeddingProvider,
    BaseEmbeddingProvider,
)
from app.modules.indexing.repository import (
    IndexingRepository,
    VectorIndexRepository,
    serialize_chroma_metadata,
)
from app.modules.indexing.router import get_indexing_service
from app.modules.indexing.schemas import (
    ChunkEmbeddingRecord,
    DocumentIndexingRecord,
    IndexingStageStatus,
    IndexingStatusResponse,
)
from app.modules.indexing.service import IndexingService


# ============================================================================
# Test Fixtures & In-Memory Database Stubs
# ============================================================================


class _FakeCursor:
    def __init__(self, items):
        self._items = list(items)

    def sort(self, key, direction=1):
        self._items.sort(key=lambda x: x.get(key, 0), reverse=(direction == -1))
        return self

    async def to_list(self, length=None):
        if length is not None:
            return [dict(x) for x in self._items[:length]]
        return [dict(x) for x in self._items]


class _FakeAsyncCollection:
    def __init__(self, name="collection"):
        self.name = name
        self.docs = []

    async def create_indexes(self, indexes):
        return True

    async def count_documents(self, filter_doc):
        return len([d for d in self.docs if all(d.get(k) == v for k, v in filter_doc.items())])

    def find(self, filter_doc=None):
        filter_doc = filter_doc or {}
        matches = [d for d in self.docs if all(d.get(k) == v for k, v in filter_doc.items())]
        return _FakeCursor(matches)

    async def find_one(self, filter_doc):
        matches = [d for d in self.docs if all(d.get(k) == v for k, v in filter_doc.items())]
        return dict(matches[0]) if matches else None

    async def replace_one(self, filter_doc, replacement, upsert=False):
        for i, d in enumerate(self.docs):
            if all(d.get(k) == v for k, v in filter_doc.items()):
                self.docs[i] = dict(replacement)
                return True
        if upsert:
            self.docs.append(dict(replacement))
            return True
        return False

    async def delete_many(self, filter_doc):
        class _DeleteResult:
            deleted_count = 0
        result = _DeleteResult()
        initial_len = len(self.docs)
        self.docs = [d for d in self.docs if not all(d.get(k) == v for k, v in filter_doc.items())]
        result.deleted_count = initial_len - len(self.docs)
        return result

    async def delete_one(self, filter_doc):
        class _DeleteResult:
            deleted_count = 0
        result = _DeleteResult()
        for i, d in enumerate(self.docs):
            if all(d.get(k) == v for k, v in filter_doc.items()):
                del self.docs[i]
                result.deleted_count = 1
                break
        return result


class _FakeAsyncDatabase:
    def __init__(self):
        self._collections = {}

    def __getitem__(self, name):
        if name not in self._collections:
            self._collections[name] = _FakeAsyncCollection(name)
        return self._collections[name]


class _FakeChromaCollection:
    def __init__(self, name="hr_policy_chunks"):
        self.name = name
        self.items = {}

    def upsert(self, ids, embeddings=None, documents=None, metadatas=None):
        for i, id_val in enumerate(ids):
            self.items[id_val] = {
                "id": id_val,
                "embedding": embeddings[i] if embeddings else None,
                "document": documents[i] if documents else None,
                "metadata": metadatas[i] if metadatas else {},
            }

    def get(self, ids=None, where=None, include=None):
        matched = []
        for id_val, item in self.items.items():
            if ids is not None and id_val not in ids:
                continue
            if where:
                if not self._match_where(item["metadata"], where):
                    continue
            matched.append(item)
        res = {"ids": [m["id"] for m in matched]}
        if include and "metadatas" in include:
            res["metadatas"] = [m["metadata"] for m in matched]
        if include and "documents" in include:
            res["documents"] = [m["document"] for m in matched]
        return res

    def _match_where(self, meta, where):
        if not isinstance(where, dict):
            return True
        if "$and" in where:
            return all(self._match_where(meta, cond) for cond in where["$and"])
        for k, v in where.items():
            if isinstance(v, dict):
                for op, val in v.items():
                    if op == "$eq" and meta.get(k) != val:
                        return False
                    if op == "$ne" and meta.get(k) == val:
                        return False
            else:
                if meta.get(k) != v:
                    return False
        return True

    def delete(self, ids=None, where=None):
        if ids:
            for id_val in ids:
                self.items.pop(id_val, None)
        elif where:
            to_delete = [
                id_val for id_val, item in self.items.items()
                if self._match_where(item["metadata"], where)
            ]
            for id_val in to_delete:
                self.items.pop(id_val, None)

    def count(self):
        return len(self.items)


class _FakeChromaService:
    def __init__(self):
        self._collections = {}

    def get_or_create_collection(self, name, metadata=None):
        if name not in self._collections:
            self._collections[name] = _FakeChromaCollection(name)
        return self._collections[name]


def _build_test_chunk(doc_id: str, index: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc_id}_chunk_{index}",
        document_id=doc_id,
        chunk_index=index,
        content=text,
        token_count=15,
        heading=f"## Clause {index}",
        heading_path=["# Policy", f"## Clause {index}"],
        page_start=1,
        page_end=1,
        source_pages=[1],
        school_id="SCH-01",
        group_id="GRP-01",
        branch_id="BR-01",
        is_global=False,
        state="Karnataka",
        staff_category="teaching",
        employment_type="full-time",
        department="science",
        document_name="leave_policy.pdf",
        document_type="policy",
        version=1,
        effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        effective_to=datetime(2026, 12, 31, tzinfo=timezone.utc),
        created_at=datetime.now(timezone.utc),
    )


# ============================================================================
# 1. Embedding Provider Tests
# ============================================================================


@pytest.mark.asyncio
async def test_azure_provider_embedding_success_and_dimensions():
    """Verify provider calls Azure OpenAI client and validates 1536-dim vector output."""
    mock_client = MagicMock()
    mock_item_1 = MagicMock()
    mock_item_1.embedding = [0.01] * 1536
    mock_item_2 = MagicMock()
    mock_item_2.embedding = [0.02] * 1536

    mock_response = MagicMock()
    mock_response.data = [mock_item_1, mock_item_2]
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    provider = AzureOpenAIEmbeddingProvider(
        endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        api_version="2024-02-01",
        deployment_name="text-embedding-3-small",
        dimensions=1536,
        client=mock_client,
    )

    assert provider.get_model_name() == "text-embedding-3-small"
    assert provider.get_dimensions() == 1536

    vectors = await provider.embed_texts(["Section 1", "Section 2"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 1536
    assert len(vectors[1]) == 1536
    assert vectors[0][0] == 0.01

    mock_client.embeddings.create.assert_awaited_once_with(
        input=["Section 1", "Section 2"],
        model="text-embedding-3-small",
    )


@pytest.mark.asyncio
async def test_azure_provider_dimension_mismatch_raises_exception():
    """Verify provider rejects responses with incorrect dimensionality."""
    mock_client = MagicMock()
    mock_item = MagicMock()
    mock_item.embedding = [0.1] * 512  # Mismatched dimension (expected 1536)
    mock_response = MagicMock()
    mock_response.data = [mock_item]
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    provider = AzureOpenAIEmbeddingProvider(
        endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        api_version="2024-02-01",
        deployment_name="text-embedding-3-small",
        dimensions=1536,
        client=mock_client,
    )

    with pytest.raises(EmbeddingException) as exc_info:
        await provider.embed_texts(["Test clause"])
    assert exc_info.value.error_code == "INVALID_EMBEDDING_DIMENSION"
    assert "expected 1536, got 512" in str(exc_info.value)


@pytest.mark.asyncio
async def test_azure_provider_invalid_numerical_values_raises_exception():
    """Verify provider rejects NaN or Inf values in embeddings."""
    mock_client = MagicMock()
    mock_item = MagicMock()
    bad_vec = [0.1] * 1536
    bad_vec[10] = float("nan")
    mock_item.embedding = bad_vec

    mock_response = MagicMock()
    mock_response.data = [mock_item]
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    provider = AzureOpenAIEmbeddingProvider(
        endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        api_version="2024-02-01",
        deployment_name="text-embedding-3-small",
        dimensions=1536,
        client=mock_client,
    )

    with pytest.raises(EmbeddingException) as exc_info:
        await provider.embed_texts(["Test clause"])
    assert exc_info.value.error_code == "INVALID_EMBEDDING_VALUES"


@pytest.mark.asyncio
async def test_azure_provider_empty_input_returns_empty():
    """Verify empty input list returns empty result without calling API."""
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock()

    provider = AzureOpenAIEmbeddingProvider(
        endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        api_version="2024-02-01",
        deployment_name="text-embedding-3-small",
        dimensions=1536,
        client=mock_client,
    )

    result = await provider.embed_texts([])
    assert result == []
    mock_client.embeddings.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_azure_provider_api_error_raises_embedding_exception():
    """Verify Azure OpenAI client exceptions are caught and wrapped in EmbeddingException."""
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=RuntimeError("Connection timeout to Azure endpoint"))

    provider = AzureOpenAIEmbeddingProvider(
        endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        api_version="2024-02-01",
        deployment_name="text-embedding-3-small",
        dimensions=1536,
        client=mock_client,
    )

    with pytest.raises(EmbeddingException) as exc_info:
        await provider.embed_texts(["Some text"])
    assert exc_info.value.error_code == "AZURE_OPENAI_ERROR"
    assert exc_info.value.status_code == 502


# ============================================================================
# 2. Metadata Serialization & ChromaDB Vector Store Tests
# ============================================================================


def test_serialize_chroma_metadata_preserves_all_fields_and_scalar_types():
    """Verify metadata serialization satisfies strict ChromaDB scalar typing."""
    chunk = _build_test_chunk("doc_meta_1", 0, "Test clause content")
    meta = serialize_chroma_metadata(chunk, "text-embedding-3-small", generation=1)

    # Verify all 21 required fields are present
    required_fields = [
        "document_id",
        "chunk_id",
        "chunk_index",
        "generation",
        "school_id",
        "group_id",
        "branch_id",
        "is_global",
        "state",
        "staff_category",
        "employment_type",
        "department",
        "document_name",
        "document_type",
        "version",
        "effective_from",
        "effective_to",
        "page_start",
        "page_end",
        "heading",
        "embedding_model",
    ]
    for field in required_fields:
        assert field in meta, f"Missing required field: {field}"
        val = meta[field]
        assert isinstance(val, (str, int, float, bool)), f"Non-scalar type for {field}: {type(val)}"

    # Specific value checks
    assert meta["generation"] == 1
    assert meta["school_id"] == "SCH-01"
    assert meta["is_global"] is False
    assert meta["effective_from"].startswith("2026-01-01")
    assert meta["effective_to"].startswith("2026-12-31")
    assert meta["embedding_model"] == "text-embedding-3-small"


def test_serialize_chroma_metadata_handles_none_values():
    """Verify None values in optional fields are safely converted to empty strings."""
    chunk = DocumentChunk(
        chunk_id="doc_null_chunk_0",
        document_id="doc_null",
        chunk_index=0,
        content="Null test content",
        token_count=10,
        page_start=1,
        page_end=1,
        school_id=None,
        group_id=None,
        branch_id=None,
        state=None,
        staff_category=None,
        employment_type=None,
        department=None,
        heading=None,
        effective_from=None,
        effective_to=None,
        document_name="doc.pdf",
    )
    meta = serialize_chroma_metadata(chunk, "text-embedding-3-small", generation=1)
    assert meta["school_id"] == ""
    assert meta["heading"] == ""
    assert meta["effective_from"] == ""
    assert meta["effective_to"] == ""
    assert all(v is not None for v in meta.values())


def test_serialize_chroma_metadata_effective_dates_formats_and_missing_values():
    """Verify effective_from and effective_to conversion for diverse datetime and string formats."""
    # 1. Timezone-aware datetime
    chunk1 = _build_test_chunk("doc_d1", 0, "Text 1")
    chunk1.effective_from = datetime(2026, 4, 1, 9, 30, 0, tzinfo=timezone.utc)
    chunk1.effective_to = datetime(2027, 3, 31, 18, 0, 0, tzinfo=timezone.utc)
    meta1 = serialize_chroma_metadata(chunk1, "model", generation=1)
    assert meta1["effective_from"] == "2026-04-01T09:30:00+00:00"
    assert meta1["effective_to"] == "2027-03-31T18:00:00+00:00"

    # 2. Timezone-naive datetime
    chunk2 = _build_test_chunk("doc_d2", 0, "Text 2")
    chunk2.effective_from = datetime(2026, 6, 15, 0, 0, 0)
    chunk2.effective_to = datetime(2026, 12, 15, 23, 59, 59)
    meta2 = serialize_chroma_metadata(chunk2, "model", generation=2)
    assert meta2["effective_from"] == "2026-06-15T00:00:00"
    assert meta2["effective_to"] == "2026-12-15T23:59:59"
    assert meta2["generation"] == 2

    # 3. String date format (pre-formatted)
    chunk3 = _build_test_chunk("doc_d3", 0, "Text 3")
    chunk3.effective_from = "2026-01-01T00:00:00Z"  # type: ignore
    chunk3.effective_to = "2026-12-31T23:59:59Z"  # type: ignore
    meta3 = serialize_chroma_metadata(chunk3, "model", generation=3)
    assert meta3["effective_from"] == "2026-01-01T00:00:00Z"
    assert meta3["effective_to"] == "2026-12-31T23:59:59Z"

    # 4. None / missing dates safely convert to empty string
    chunk4 = _build_test_chunk("doc_d4", 0, "Text 4")
    chunk4.effective_from = None
    chunk4.effective_to = None
    meta4 = serialize_chroma_metadata(chunk4, "model")
    assert meta4["effective_from"] == ""
    assert meta4["effective_to"] == ""
    assert isinstance(meta4["effective_from"], str)
    assert isinstance(meta4["effective_to"], str)


def test_vector_repo_upsert_and_reindexing_idempotence():
    """Verify ChromaDB upsert behavior does not duplicate chunks on re-indexing."""
    fake_chroma = _FakeChromaService()
    repo = VectorIndexRepository(fake_chroma, collection_name="hr_policy_chunks")

    ids = ["doc1_chunk_0_g1", "doc1_chunk_1_g1"]
    embeddings = [[0.1] * 1536, [0.2] * 1536]
    docs = ["Clause 0", "Clause 1"]
    metadatas = [
        {"document_id": "doc1", "chunk_id": "doc1_chunk_0", "generation": 1},
        {"document_id": "doc1", "chunk_id": "doc1_chunk_1", "generation": 1},
    ]

    repo.upsert_vectors(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)
    assert repo.count() == 2

    # Re-upserting same IDs updates in-place without duplicating
    repo.upsert_vectors(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)
    assert repo.count() == 2


def test_vector_repo_generation_querying_and_cleanup():
    """Verify VectorIndexRepository generation-aware count, query, and cleanup methods."""
    fake_chroma = _FakeChromaService()
    repo = VectorIndexRepository(fake_chroma, collection_name="hr_policy_chunks")

    # 1. Upsert generation 1 vectors
    ids_g1 = ["doc_g_chunk_0_g1", "doc_g_chunk_1_g1"]
    meta_g1 = [
        {"document_id": "doc_g", "chunk_id": "doc_g_chunk_0", "generation": 1},
        {"document_id": "doc_g", "chunk_id": "doc_g_chunk_1", "generation": 1},
    ]
    repo.upsert_vectors(
        ids=ids_g1,
        embeddings=[[0.1] * 1536, [0.1] * 1536],
        documents=["text 0", "text 1"],
        metadatas=meta_g1,
    )
    assert repo.count_generation_vectors("doc_g", 1) == 2
    assert repo.count_generation_vectors("doc_g", 2) == 0

    # 2. Upsert generation 2 vectors while generation 1 remains present
    ids_g2 = ["doc_g_chunk_0_g2", "doc_g_chunk_1_g2"]
    meta_g2 = [
        {"document_id": "doc_g", "chunk_id": "doc_g_chunk_0", "generation": 2},
        {"document_id": "doc_g", "chunk_id": "doc_g_chunk_1", "generation": 2},
    ]
    repo.upsert_vectors(
        ids=ids_g2,
        embeddings=[[0.2] * 1536, [0.2] * 1536],
        documents=["text 0 v2", "text 1 v2"],
        metadatas=meta_g2,
    )
    assert repo.count_generation_vectors("doc_g", 1) == 2
    assert repo.count_generation_vectors("doc_g", 2) == 2
    assert repo.count() == 4

    # 3. Delete previous generations (keeping generation 2 active)
    purged = repo.delete_previous_generations("doc_g", active_generation=2)
    assert purged == 2
    assert repo.count_generation_vectors("doc_g", 1) == 0
    assert repo.count_generation_vectors("doc_g", 2) == 2
    assert repo.count() == 2

    # 4. Delete generation 2
    deleted = repo.delete_generation("doc_g", 2)
    assert deleted == 2
    assert repo.count() == 0


# ============================================================================
# 3. Service Layer Tests (Caching, Batching, Partial Failure, Stale Vectors)
# ============================================================================


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Predictable mock provider for service testing."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.call_count = 0
        self.embedded_texts: List[str] = []

    def get_model_name(self) -> str:
        return self.model_name

    def get_dimensions(self) -> int:
        return 1536

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        self.embedded_texts.extend(texts)
        # Deterministic vector based on text hash
        results = []
        for t in texts:
            val = float(int(hashlib.md5(t.encode("utf-8")).hexdigest()[:4], 16) % 100) / 100.0
            results.append([val] * 1536)
        return results


@pytest.mark.asyncio
async def test_service_successful_indexing_with_content_hash_caching():
    """Verify first run embeds chunks, second run reuses embeddings via content-hash caching."""
    db = _FakeAsyncDatabase()
    document_repo = MagicMock()
    chunk_repo = MagicMock()

    doc_metadata = {"document_id": "doc_cache_test", "document_name": "policy.pdf", "processing_status": "chunked"}
    document_repo.get_by_id = AsyncMock(return_value=doc_metadata)
    document_repo.update_processing_status = AsyncMock(return_value=True)

    chunks = [_build_test_chunk("doc_cache_test", i, f"Policy clause {i}") for i in range(3)]
    chunk_repo.get_all_by_document_id = AsyncMock(return_value=[c.model_dump(mode="python") for c in chunks])

    indexing_repo = IndexingRepository(db)
    fake_chroma = _FakeChromaService()
    vector_repo = VectorIndexRepository(fake_chroma)

    provider = MockEmbeddingProvider()

    service = IndexingService(
        document_repo=document_repo,
        chunk_repo=chunk_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
        batch_size=2,
    )

    # 1. First Run: all 3 chunks should be embedded under generation 1
    res1 = await service.index_document("doc_cache_test")
    assert res1.total_chunks == 3
    assert res1.embedded_chunks == 3
    assert res1.indexed_chunks == 3
    assert res1.active_generation == 1
    assert res1.indexing_status == "indexed"
    assert provider.call_count == 2  # batch_size=2 -> 2 calls (batch of 2, batch of 1)
    assert fake_chroma.get_or_create_collection("hr_policy_chunks").count() == 3

    # 2. Second Run with identical content: 100% Cache Hit, advances to generation 2!
    provider_calls_before = provider.call_count
    res2 = await service.index_document("doc_cache_test")
    assert res2.total_chunks == 3
    assert res2.embedded_chunks == 3
    assert res2.indexed_chunks == 3
    assert res2.active_generation == 2
    assert res2.indexing_status == "indexed"
    # Zero new API calls to embedding provider!
    assert provider.call_count == provider_calls_before
    # Previous generation 1 vectors pruned, generation 2 active
    coll = fake_chroma.get_or_create_collection("hr_policy_chunks")
    assert coll.count() == 3
    assert all("g2" in cid for cid in coll.items.keys())


@pytest.mark.asyncio
async def test_service_content_change_reembeds_only_modified_chunk():
    """Verify modifying 1 chunk re-embeds only that chunk, reusing unchanged chunks."""
    db = _FakeAsyncDatabase()
    document_repo = MagicMock()
    chunk_repo = MagicMock()

    doc_metadata = {"document_id": "doc_change_test", "document_name": "policy.pdf", "processing_status": "chunked"}
    document_repo.get_by_id = AsyncMock(return_value=doc_metadata)
    document_repo.update_processing_status = AsyncMock(return_value=True)

    chunks = [_build_test_chunk("doc_change_test", i, f"Clause {i}") for i in range(2)]
    chunk_repo.get_all_by_document_id = AsyncMock(return_value=[c.model_dump(mode="python") for c in chunks])

    indexing_repo = IndexingRepository(db)
    fake_chroma = _FakeChromaService()
    vector_repo = VectorIndexRepository(fake_chroma)

    provider = MockEmbeddingProvider()

    service = IndexingService(
        document_repo=document_repo,
        chunk_repo=chunk_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
    )

    # Initial indexing: 2 chunks embedded
    res1 = await service.index_document("doc_change_test")
    assert res1.active_generation == 1
    assert len(provider.embedded_texts) == 2

    # Modify ONLY chunk 1
    modified_chunks = [
        _build_test_chunk("doc_change_test", 0, "Clause 0"),  # Unchanged
        _build_test_chunk("doc_change_test", 1, "Clause 1 MODIFIED CONTENT"),  # Changed
    ]
    chunk_repo.get_all_by_document_id = AsyncMock(return_value=[c.model_dump(mode="python") for c in modified_chunks])

    provider.embedded_texts.clear()
    res2 = await service.index_document("doc_change_test")
    assert res2.active_generation == 2

    # Only the modified chunk was sent to the provider!
    assert len(provider.embedded_texts) == 1
    assert "Clause 1 MODIFIED CONTENT" in provider.embedded_texts[0]


@pytest.mark.asyncio
async def test_service_stale_vectors_pruned_on_rechunking():
    """Verify that when chunk count decreases, old vector IDs in ChromaDB are removed."""
    db = _FakeAsyncDatabase()
    document_repo = MagicMock()
    chunk_repo = MagicMock()

    document_repo.get_by_id = AsyncMock(return_value={"document_id": "doc_stale", "processing_status": "chunked"})
    document_repo.update_processing_status = AsyncMock(return_value=True)

    # Initial document has 4 chunks
    initial_chunks = [_build_test_chunk("doc_stale", i, f"Clause {i}") for i in range(4)]
    chunk_repo.get_all_by_document_id = AsyncMock(return_value=[c.model_dump(mode="python") for c in initial_chunks])

    indexing_repo = IndexingRepository(db)
    fake_chroma = _FakeChromaService()
    vector_repo = VectorIndexRepository(fake_chroma)

    provider = MockEmbeddingProvider()

    service = IndexingService(
        document_repo=document_repo,
        chunk_repo=chunk_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
    )

    # Initial indexing: 4 chunks active in generation 1
    res1 = await service.index_document("doc_stale")
    assert res1.active_generation == 1
    assert fake_chroma.get_or_create_collection("hr_policy_chunks").count() == 4

    # Document now only has 2 chunks (down from 4 in ChromaDB)
    current_chunks = [_build_test_chunk("doc_stale", i, f"Clause {i}") for i in range(2)]
    chunk_repo.get_all_by_document_id = AsyncMock(return_value=[c.model_dump(mode="python") for c in current_chunks])

    res2 = await service.index_document("doc_stale")
    assert res2.active_generation == 2
    assert res2.indexed_chunks == 2

    # ChromaDB must now have exactly 2 active vectors belonging to generation 2
    coll = fake_chroma.get_or_create_collection("hr_policy_chunks")
    assert coll.count() == 2
    assert set(coll.items.keys()) == {"doc_stale_chunk_0_g2", "doc_stale_chunk_1_g2"}


@pytest.mark.asyncio
async def test_service_safe_generations_lifecycle_activation_and_cleanup():
    """Verify safe generation transitions: write new -> verify -> activate -> delete previous."""
    db = _FakeAsyncDatabase()
    document_repo = MagicMock()
    chunk_repo = MagicMock()

    doc_metadata = {"document_id": "doc_lifecycle", "document_name": "policy.pdf", "processing_status": "chunked"}
    document_repo.get_by_id = AsyncMock(return_value=doc_metadata)
    document_repo.update_processing_status = AsyncMock(return_value=True)

    chunks = [_build_test_chunk("doc_lifecycle", i, f"Clause {i}") for i in range(3)]
    chunk_repo.get_all_by_document_id = AsyncMock(return_value=[c.model_dump(mode="python") for c in chunks])

    indexing_repo = IndexingRepository(db)
    fake_chroma = _FakeChromaService()
    vector_repo = VectorIndexRepository(fake_chroma)
    provider = MockEmbeddingProvider()

    service = IndexingService(
        document_repo=document_repo,
        chunk_repo=chunk_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
    )

    # 1. First index -> creates and activates Generation 1
    res1 = await service.index_document("doc_lifecycle")
    assert res1.active_generation == 1
    assert res1.indexed_chunks == 3
    assert vector_repo.count_generation_vectors("doc_lifecycle", 1) == 3

    # Check tracking record
    tracking1 = await indexing_repo.get_tracking_record("doc_lifecycle")
    assert tracking1["active_generation"] == 1
    assert tracking1["indexing_status"] == IndexingStageStatus.INDEXED

    # 2. Second index (re-index) -> creates Generation 2, keeps Gen 1 available until activation
    res2 = await service.index_document("doc_lifecycle")
    assert res2.active_generation == 2
    assert res2.indexed_chunks == 3

    # Check ChromaDB: Generation 1 has been cleaned up, Generation 2 is active
    assert vector_repo.count_generation_vectors("doc_lifecycle", 1) == 0
    assert vector_repo.count_generation_vectors("doc_lifecycle", 2) == 3

    tracking2 = await indexing_repo.get_tracking_record("doc_lifecycle")
    assert tracking2["active_generation"] == 2
    assert tracking2["indexing_status"] == IndexingStageStatus.INDEXED


@pytest.mark.asyncio
async def test_service_safe_generations_failure_retains_active_generation():
    """Verify interrupted/failed indexing leaves previous active generation intact."""
    db = _FakeAsyncDatabase()
    document_repo = MagicMock()
    chunk_repo = MagicMock()

    doc_metadata = {"document_id": "doc_fail_gen", "document_name": "policy.pdf", "processing_status": "chunked"}
    document_repo.get_by_id = AsyncMock(return_value=doc_metadata)
    document_repo.update_processing_status = AsyncMock(return_value=True)

    chunks = [_build_test_chunk("doc_fail_gen", i, f"Clause {i}") for i in range(2)]
    chunk_repo.get_all_by_document_id = AsyncMock(return_value=[c.model_dump(mode="python") for c in chunks])

    indexing_repo = IndexingRepository(db)
    fake_chroma = _FakeChromaService()
    vector_repo = VectorIndexRepository(fake_chroma)
    provider = MockEmbeddingProvider()

    service = IndexingService(
        document_repo=document_repo,
        chunk_repo=chunk_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
    )

    # 1. Establish valid Generation 1
    res1 = await service.index_document("doc_fail_gen")
    assert res1.active_generation == 1
    assert vector_repo.count_generation_vectors("doc_fail_gen", 1) == 2

    # 2. Inject failure during ChromaDB vector write for Generation 2
    orig_upsert = vector_repo.upsert_vectors

    def failing_upsert(*args, **kwargs):
        raise RuntimeError("ChromaDB connection interrupted during generation 2 write")

    vector_repo.upsert_vectors = failing_upsert

    with pytest.raises(IndexingException) as exc_info:
        await service.index_document("doc_fail_gen")
    assert exc_info.value.error_code == "INDEXING_FAILED"

    # 3. VERIFY: Generation 1 vectors remain 100% available and intact in ChromaDB!
    assert vector_repo.count_generation_vectors("doc_fail_gen", 1) == 2
    assert vector_repo.count_generation_vectors("doc_fail_gen", 2) == 0

    # 4. VERIFY: Tracking record retains active_generation = 1 with status = FAILED
    tracking_failed = await indexing_repo.get_tracking_record("doc_fail_gen")
    assert tracking_failed["active_generation"] == 1
    assert tracking_failed["indexing_status"] == IndexingStageStatus.FAILED
    assert "connection interrupted" in tracking_failed["error_message"]

    # 5. Restore normal upsert and retry -> successfully activates Generation 2
    vector_repo.upsert_vectors = orig_upsert
    res_retry = await service.index_document("doc_fail_gen")
    assert res_retry.active_generation == 2
    assert res_retry.indexing_status == "indexed"
    assert vector_repo.count_generation_vectors("doc_fail_gen", 1) == 0
    assert vector_repo.count_generation_vectors("doc_fail_gen", 2) == 2


@pytest.mark.asyncio
async def test_service_verification_failure_rolls_back_generation():
    """Verify that a vector count mismatch during verification rolls back the generation."""
    db = _FakeAsyncDatabase()
    document_repo = MagicMock()
    chunk_repo = MagicMock()

    doc_metadata = {"document_id": "doc_verify_fail", "document_name": "policy.pdf", "processing_status": "chunked"}
    document_repo.get_by_id = AsyncMock(return_value=doc_metadata)
    document_repo.update_processing_status = AsyncMock(return_value=True)

    chunks = [_build_test_chunk("doc_verify_fail", i, f"Clause {i}") for i in range(2)]
    chunk_repo.get_all_by_document_id = AsyncMock(return_value=[c.model_dump(mode="python") for c in chunks])

    indexing_repo = IndexingRepository(db)
    fake_chroma = _FakeChromaService()
    vector_repo = VectorIndexRepository(fake_chroma)
    provider = MockEmbeddingProvider()

    service = IndexingService(
        document_repo=document_repo,
        chunk_repo=chunk_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
    )

    # Establish Generation 1
    await service.index_document("doc_verify_fail")
    assert vector_repo.count_generation_vectors("doc_verify_fail", 1) == 2

    # Simulate verification mismatch on next run by making count_generation_vectors return 1 instead of 2
    vector_repo.count_generation_vectors = MagicMock(return_value=1)

    with pytest.raises(IndexingException) as exc_info:
        await service.index_document("doc_verify_fail")
    assert exc_info.value.error_code == "INDEXING_VERIFICATION_FAILED"

    # Tracking record retains active_generation = 1
    tracking = await indexing_repo.get_tracking_record("doc_verify_fail")
    assert tracking["active_generation"] == 1
    assert tracking["indexing_status"] == IndexingStageStatus.FAILED


@pytest.mark.asyncio
async def test_service_partial_batch_failure_and_retry():
    """Verify partial batch failure marks document failed, but preserves successful batch in cache."""
    db = _FakeAsyncDatabase()
    document_repo = MagicMock()
    chunk_repo = MagicMock()

    document_repo.get_by_id = AsyncMock(return_value={"document_id": "doc_fail_retry", "processing_status": "chunked"})
    document_repo.update_processing_status = AsyncMock(return_value=True)

    # 4 chunks with batch_size=2
    chunks = [_build_test_chunk("doc_fail_retry", i, f"Paragraph {i}") for i in range(4)]
    chunk_repo.get_all_by_document_id = AsyncMock(return_value=[c.model_dump(mode="python") for c in chunks])

    indexing_repo = IndexingRepository(db)
    fake_chroma = _FakeChromaService()
    vector_repo = VectorIndexRepository(fake_chroma)

    provider = MagicMock(spec=BaseEmbeddingProvider)
    provider.get_model_name.return_value = "text-embedding-3-small"
    provider.get_dimensions.return_value = 1536

    # Batch 1 succeeds, Batch 2 fails
    call_count = 0

    async def flaky_embed(texts):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Azure rate limit exceeded on second batch")
        return [[0.1] * 1536 for _ in texts]

    provider.embed_texts = AsyncMock(side_effect=flaky_embed)

    service = IndexingService(
        document_repo=document_repo,
        chunk_repo=chunk_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
        batch_size=2,
    )

    # 1. Attempt indexing -> raises EmbeddingException on batch 2
    with pytest.raises(EmbeddingException):
        await service.index_document("doc_fail_retry")

    # Document tracking status should be 'failed'
    status_rec = await indexing_repo.get_tracking_record("doc_fail_retry")
    assert status_rec["indexing_status"] == IndexingStageStatus.FAILED
    assert "Azure rate limit" in status_rec["error_message"]

    # BUT batch 1 (chunks 0 and 1) MUST be safely cached in MongoDB!
    cached_0 = await indexing_repo.get_chunk_embedding(
        "doc_fail_retry_chunk_0",
        hashlib.sha256(b"Paragraph 0").hexdigest(),
        "text-embedding-3-small",
    )
    assert cached_0 is not None
    assert cached_0["embedding_status"] == "completed"

    # 2. Retry indexing: fix provider to succeed
    provider.embed_texts = AsyncMock(return_value=[[0.2] * 1536, [0.2] * 1536])
    res_retry = await service.index_document("doc_fail_retry")

    assert res_retry.indexing_status == "indexed"
    assert res_retry.total_chunks == 4
    assert res_retry.indexed_chunks == 4
    # On retry, provider was ONLY called once for the remaining 2 chunks!
    assert provider.embed_texts.call_count == 1


@pytest.mark.asyncio
async def test_service_empty_document_handling():
    """Verify empty document produces clean zero-chunk indexed status."""
    db = _FakeAsyncDatabase()
    document_repo = MagicMock()
    chunk_repo = MagicMock()

    document_repo.get_by_id = AsyncMock(return_value={"document_id": "doc_empty", "processing_status": "completed"})
    chunk_repo.get_all_by_document_id = AsyncMock(return_value=[])

    indexing_repo = IndexingRepository(db)
    mock_chroma_service = MagicMock(spec=ChromaDBService)
    vector_repo = VectorIndexRepository(mock_chroma_service)
    provider = MockEmbeddingProvider()

    service = IndexingService(
        document_repo=document_repo,
        chunk_repo=chunk_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
    )

    res = await service.index_document("doc_empty")
    assert res.total_chunks == 0
    assert res.indexing_status == "indexed"


@pytest.mark.asyncio
async def test_service_missing_document_raises_404():
    """Verify NotFoundException raised if document UUID is not found."""
    document_repo = MagicMock()
    document_repo.get_by_id = AsyncMock(return_value=None)

    service = IndexingService(
        document_repo=document_repo,
        chunk_repo=MagicMock(),
        indexing_repo=MagicMock(),
        vector_repo=MagicMock(),
        embedding_provider=MockEmbeddingProvider(),
    )

    with pytest.raises(NotFoundException) as exc_info:
        await service.index_document("doc_missing")
    assert exc_info.value.status_code == 404
    assert exc_info.value.error_code == "DOCUMENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_service_un_chunked_document_raises_400():
    """Verify ProcessingException raised if document has not been chunked."""
    document_repo = MagicMock()
    document_repo.get_by_id = AsyncMock(return_value={"document_id": "doc_unchunked", "processing_status": "uploaded"})

    chunk_repo = MagicMock()
    chunk_repo.get_all_by_document_id = AsyncMock(return_value=[])

    service = IndexingService(
        document_repo=document_repo,
        chunk_repo=chunk_repo,
        indexing_repo=MagicMock(),
        vector_repo=MagicMock(),
        embedding_provider=MockEmbeddingProvider(),
    )

    with pytest.raises(ProcessingException) as exc_info:
        await service.index_document("doc_unchunked")
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "DOCUMENT_NOT_CHUNKED"


# ============================================================================
# 4. API Endpoint Tests
# ============================================================================


def test_api_index_document_success():
    """Verify POST /api/v1/documents/{document_id}/index returns 200 with indexing summary."""
    client = TestClient(app)
    mock_service = MagicMock()

    mock_res = IndexingStatusResponse(
        document_id="doc_api_index",
        total_chunks=5,
        embedded_chunks=5,
        indexed_chunks=5,
        failed_chunks=0,
        active_generation=1,
        embedding_model="text-embedding-3-small",
        indexing_status="indexed",
        last_indexed_at=datetime.now(timezone.utc),
    )
    mock_service.index_document = AsyncMock(return_value=mock_res)

    app.dependency_overrides[get_indexing_service] = lambda: mock_service
    try:
        response = client.post("/api/v1/documents/doc_api_index/index")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["document_id"] == "doc_api_index"
        assert data["data"]["indexing_status"] == "indexed"
        assert data["data"]["indexed_chunks"] == 5
        assert data["data"]["active_generation"] == 1
    finally:
        app.dependency_overrides.pop(get_indexing_service, None)


def test_api_index_document_not_found():
    """Verify POST /api/v1/documents/{document_id}/index returns 404 for nonexistent document."""
    client = TestClient(app)
    mock_service = MagicMock()
    mock_service.index_document = AsyncMock(
        side_effect=NotFoundException("Document with ID 'doc_miss' not found", "DOCUMENT_NOT_FOUND")
    )

    app.dependency_overrides[get_indexing_service] = lambda: mock_service
    try:
        response = client.post("/api/v1/documents/doc_miss/index")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error_code"] == "DOCUMENT_NOT_FOUND"
    finally:
        app.dependency_overrides.pop(get_indexing_service, None)


def test_api_get_indexing_status_success():
    """Verify GET /api/v1/documents/{document_id}/index-status returns 200 with status."""
    client = TestClient(app)
    mock_service = MagicMock()

    mock_res = IndexingStatusResponse(
        document_id="doc_api_status",
        total_chunks=3,
        embedded_chunks=3,
        indexed_chunks=3,
        failed_chunks=0,
        active_generation=2,
        embedding_model="text-embedding-3-small",
        indexing_status="indexed",
        last_indexed_at=datetime.now(timezone.utc),
    )
    mock_service.get_indexing_status = AsyncMock(return_value=mock_res)

    app.dependency_overrides[get_indexing_service] = lambda: mock_service
    try:
        response = client.get("/api/v1/documents/doc_api_status/index-status")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["document_id"] == "doc_api_status"
        assert data["data"]["indexing_status"] == "indexed"
        assert data["data"]["active_generation"] == 2
    finally:
        app.dependency_overrides.pop(get_indexing_service, None)


def test_api_get_indexing_status_not_found():
    """Verify GET /api/v1/documents/{document_id}/index-status returns 404 for nonexistent document."""
    client = TestClient(app)
    mock_service = MagicMock()
    mock_service.get_indexing_status = AsyncMock(
        side_effect=NotFoundException("Document with ID 'doc_miss' not found", "DOCUMENT_NOT_FOUND")
    )

    app.dependency_overrides[get_indexing_service] = lambda: mock_service
    try:
        response = client.get("/api/v1/documents/doc_miss/index-status")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error_code"] == "DOCUMENT_NOT_FOUND"
    finally:
        app.dependency_overrides.pop(get_indexing_service, None)
