"""Tests for Query Understanding & Semantic Retrieval (Phase 6)."""

from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
import pytest
from starlette.testclient import TestClient

from app.api.dependencies import SchoolContext
from app.core.exceptions import AppException, EmbeddingException, NotFoundException
from app.main import app
from app.modules.indexing.provider import BaseEmbeddingProvider
from app.modules.indexing.repository import IndexingRepository, VectorIndexRepository
from app.modules.ingestion.repository import DocumentRepository
from app.modules.retrieval.filtering import PolicyScopeFilter
from app.modules.retrieval.router import get_retrieval_service
from app.modules.retrieval.schemas import (
    EmployeeContext,
    QueryUnderstandingResult,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)
from app.modules.retrieval.service import RetrievalService
from app.modules.retrieval.understanding import QueryUnderstandingService


# ============================================================================
# In-Memory Stubs & Fixtures
# ============================================================================


class _FakeCursor:
    def __init__(self, items):
        self._items = list(items)

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
            if where and not self._match_where(item["metadata"], where):
                continue
            matched.append(item)
        res = {"ids": [m["id"] for m in matched]}
        if include and "metadatas" in include:
            res["metadatas"] = [m["metadata"] for m in matched]
        if include and "documents" in include:
            res["documents"] = [m["document"] for m in matched]
        return res

    def query(self, query_embeddings, n_results=10, where=None, include=None):
        matched = []
        q_vec = query_embeddings[0] if query_embeddings else []
        for id_val, item in self.items.items():
            if where and not self._match_where(item["metadata"], where):
                continue
            emb = item.get("embedding")
            if emb and q_vec:
                diff = sum(abs(a - b) for a, b in zip(q_vec[:10], emb[:10])) / 10.0
                dist = round(diff, 4)
            else:
                dist = 0.5
            matched.append((dist, item))

        matched.sort(key=lambda x: x[0])
        top_items = matched[:n_results]

        return {
            "ids": [[m[1]["id"] for m in top_items]],
            "documents": [[m[1]["document"] for m in top_items]],
            "metadatas": [[m[1]["metadata"] for m in top_items]],
            "distances": [[m[0] for m in top_items]],
        }

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

    def count(self):
        return len(self.items)


class _FakeChromaService:
    def __init__(self):
        self._collections = {}

    def get_or_create_collection(self, name, metadata=None):
        if name not in self._collections:
            self._collections[name] = _FakeChromaCollection(name)
        return self._collections[name]


class MockEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.call_count = 0

    def get_model_name(self) -> str:
        return self.model_name

    def get_dimensions(self) -> int:
        return 1536

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        results = []
        for t in texts:
            # Deterministic float based on text hash
            val = float(int(hashlib.md5(t.encode("utf-8")).hexdigest()[:4], 16) % 100) / 100.0
            results.append([val] * 1536)
        return results


def _build_test_document(
    doc_id: str,
    name: str = "leave_policy.pdf",
    school_id: Optional[str] = "SCH-01",
    group_id: Optional[str] = "GRP-01",
    branch_id: Optional[str] = None,
    state: Optional[str] = "Karnataka",
    status: str = "ready",
    effective_from: Optional[datetime] = datetime(2026, 1, 1, tzinfo=timezone.utc),
    effective_to: Optional[datetime] = datetime(2026, 12, 31, tzinfo=timezone.utc),
) -> Dict[str, Any]:
    return {
        "document_id": doc_id,
        "document_name": name,
        "school_id": school_id,
        "group_id": group_id,
        "branch_id": branch_id,
        "state": state,
        "status": status,
        "processing_status": "completed",
        "version": 1,
        "effective_from": effective_from,
        "effective_to": effective_to,
    }


def _build_test_chunk_meta(
    doc_id: str,
    chunk_index: int,
    generation: int = 1,
    school_id: Optional[str] = "SCH-01",
    group_id: Optional[str] = "GRP-01",
    branch_id: Optional[str] = None,
    state: Optional[str] = "Karnataka",
    is_global: bool = False,
    staff_category: Optional[str] = None,
    employment_type: Optional[str] = None,
    department: Optional[str] = None,
    effective_from: str = "2026-01-01T00:00:00+00:00",
    effective_to: str = "2026-12-31T00:00:00+00:00",
) -> Dict[str, Any]:
    return {
        "chunk_id": f"{doc_id}_chunk_{chunk_index}",
        "document_id": doc_id,
        "chunk_index": chunk_index,
        "generation": generation,
        "document_name": "leave_policy.pdf",
        "document_type": "policy",
        "version": 1,
        "school_id": school_id or "",
        "group_id": group_id or "",
        "branch_id": branch_id or "",
        "state": state or "",
        "is_global": is_global,
        "staff_category": staff_category or "",
        "employment_type": employment_type or "",
        "department": department or "",
        "heading": f"## Section {chunk_index}",
        "heading_path": "Policy > Section",
        "page_start": 1,
        "page_end": 1,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "embedding_model": "text-embedding-3-small",
    }


# ============================================================================
# 1. Query Understanding Tests
# ============================================================================


def test_query_understanding_normalization_and_whitespace():
    """Verify whitespace collapsing and empty query rejection."""
    service = QueryUnderstandingService()
    ctx = SchoolContext(school_id="SCH-01")

    # Empty queries raise 400
    with pytest.raises(AppException) as exc1:
        service.understand_query("   ", ctx)
    assert exc1.value.status_code == 400
    assert exc1.value.error_code == "QUERY_EMPTY"

    with pytest.raises(AppException) as exc2:
        service.understand_query("", ctx)
    assert exc2.value.status_code == 400

    # Whitespace normalization
    res = service.understand_query("  How   many  casual   leaves\n\nare\tavailable?  ", ctx)
    assert res.normalized_query == "How many casual leaves are available?"
    assert res.original_query == "  How   many  casual   leaves\n\nare\tavailable?  "


def test_query_understanding_intent_classification():
    """Verify deterministic intent classification across key HR policy categories."""
    service = QueryUnderstandingService()
    ctx = SchoolContext(school_id="SCH-01")

    assert service.understand_query("How many casual leaves can I take?", ctx).intent == "leave"
    assert service.understand_query("What is the maternity leave policy for teachers?", ctx).intent == "leave"
    assert service.understand_query("When is monthly salary credited and what are the deductions?", ctx).intent == "compensation"
    assert service.understand_query("Is health insurance medical cover provided?", ctx).intent == "benefits"
    assert service.understand_query("What are the working hours and biometric rules?", ctx).intent == "attendance"
    assert service.understand_query("What constitutes disciplinary misconduct under the code of conduct?", ctx).intent == "conduct"
    assert service.understand_query("What is the probation period before confirmation?", ctx).intent == "probation"
    assert service.understand_query("What is the notice period on resignation?", ctx).intent == "termination"
    assert service.understand_query("What are the eligibility requirements for hiring?", ctx).intent == "recruitment"
    assert service.understand_query("General question without keywords", ctx).intent == "general_inquiry"


def test_query_understanding_keyword_extraction_and_stop_words():
    """Verify deterministic stop word removal and keyword extraction."""
    service = QueryUnderstandingService()
    ctx = SchoolContext(school_id="SCH-01")

    res = service.understand_query("What are the maternity leave rules for permanent faculty?", ctx)
    assert "maternity" in res.extracted_keywords
    assert "leave" in res.extracted_keywords
    assert "faculty" in res.extracted_keywords
    assert "what" not in res.extracted_keywords
    assert "the" not in res.extracted_keywords
    assert "for" not in res.extracted_keywords


def test_query_understanding_explicit_employee_context():
    """Verify employee applicability is extracted only when explicitly stated or supplied."""
    service = QueryUnderstandingService()
    ctx = SchoolContext(school_id="SCH-01")

    # Extracted from natural language
    res1 = service.understand_query("How many leaves for permanent teaching staff in mathematics?", ctx)
    assert res1.employee_context is not None
    assert res1.employee_context.staff_category == "teaching"
    assert res1.employee_context.employment_type == "permanent"

    # Supplied in EmployeeContext
    emp_ctx = EmployeeContext(staff_category="non_teaching", employment_type="contract", department="science")
    res2 = service.understand_query("What is the policy?", ctx, employee_context=emp_ctx)
    assert res2.employee_context.staff_category == "non_teaching"
    assert res2.employee_context.employment_type == "contract"
    assert res2.employee_context.department == "science"

    # General query without explicit employee details -> None (do not invent)
    res3 = service.understand_query("What is the general holiday calendar?", ctx)
    assert res3.employee_context is None


# ============================================================================
# 2. Scope & Applicability Filtering Tests (PolicyScopeFilter)
# ============================================================================


def test_tenant_scope_isolation_school_and_branch():
    """Verify branch and school isolation rules."""
    scope_delhi = SchoolContext(school_id="SCH-DELHI", branch_id="BR-NORTH")
    scope_mumbai = SchoolContext(school_id="SCH-MUMBAI", branch_id="BR-SOUTH")

    # 1. Branch specific policy
    meta_branch_delhi = {"school_id": "SCH-DELHI", "branch_id": "BR-NORTH"}
    assert PolicyScopeFilter.is_tenant_scope_applicable(meta_branch_delhi, scope_delhi) is True
    assert PolicyScopeFilter.is_tenant_scope_applicable(meta_branch_delhi, scope_mumbai) is False

    # 2. School wide policy (matches any branch in same school)
    meta_school_delhi = {"school_id": "SCH-DELHI", "branch_id": ""}
    assert PolicyScopeFilter.is_tenant_scope_applicable(meta_school_delhi, scope_delhi) is True
    assert PolicyScopeFilter.is_tenant_scope_applicable(meta_school_delhi, scope_mumbai) is False


def test_tenant_scope_isolation_group_and_state():
    """Verify group-level and state-level policy matching."""
    scope_dps = SchoolContext(school_id="SCH-DPS-01", group_id="GRP-DPS", state="Delhi")
    scope_ryan = SchoolContext(school_id="SCH-RYAN-01", group_id="GRP-RYAN", state="Maharashtra")

    # Group policy must never leak to unrelated group
    meta_dps_group = {"group_id": "GRP-DPS", "school_id": "", "branch_id": ""}
    assert PolicyScopeFilter.is_tenant_scope_applicable(meta_dps_group, scope_dps) is True
    assert PolicyScopeFilter.is_tenant_scope_applicable(meta_dps_group, scope_ryan) is False

    # State policy
    meta_delhi_state = {"state": "Delhi", "group_id": "GRP-DPS"}
    assert PolicyScopeFilter.is_tenant_scope_applicable(meta_delhi_state, scope_dps) is True
    assert PolicyScopeFilter.is_tenant_scope_applicable(meta_delhi_state, scope_ryan) is False


def test_tenant_scope_global_isolation():
    """Verify is_global=True cannot be used standalone to bypass tenant context."""
    scope_unrelated = SchoolContext(school_id="SCH-99", group_id="GRP-99")
    meta_global_with_group = {"is_global": True, "group_id": "GRP-DPS"}

    # Must match group even if global flag is true
    assert PolicyScopeFilter.is_tenant_scope_applicable(meta_global_with_group, scope_unrelated) is False


def test_employee_applicability_filtering():
    """Verify staff_category, employment_type, and department filtering."""
    teaching_ctx = EmployeeContext(staff_category="teaching", employment_type="full_time", department="science")

    # Exact match
    meta1 = {"staff_category": "teaching", "employment_type": "full_time", "department": "science"}
    assert PolicyScopeFilter.is_employee_applicable(meta1, teaching_ctx) is True

    # Generic policy (empty applicability) matches
    meta2 = {"staff_category": "", "employment_type": "", "department": ""}
    assert PolicyScopeFilter.is_employee_applicable(meta2, teaching_ctx) is True

    # Mismatch staff category
    meta3 = {"staff_category": "non_teaching"}
    assert PolicyScopeFilter.is_employee_applicable(meta3, teaching_ctx) is False

    # Mismatch department
    meta4 = {"staff_category": "teaching", "department": "mathematics"}
    assert PolicyScopeFilter.is_employee_applicable(meta4, teaching_ctx) is False


def test_policy_active_status_and_effective_dates():
    """Verify draft, superseded, expired, and future policies are rejected."""
    as_of = datetime(2026, 6, 1, tzinfo=timezone.utc)

    # Active valid policy
    doc_valid = {"status": "ready", "effective_from": "2026-01-01T00:00:00Z", "effective_to": "2026-12-31T00:00:00Z"}
    active, _ = PolicyScopeFilter.is_policy_active(doc_valid, as_of)
    assert active is True

    # Inactive status
    doc_draft = {"status": "draft", "effective_from": "2026-01-01T00:00:00Z"}
    active, reason = PolicyScopeFilter.is_policy_active(doc_draft, as_of)
    assert active is False
    assert "draft" in reason

    # Expired policy
    doc_expired = {"status": "ready", "effective_to": "2026-05-01T00:00:00Z"}
    active, reason = PolicyScopeFilter.is_policy_active(doc_expired, as_of)
    assert active is False
    assert "expired" in reason

    # Future policy (not yet effective)
    doc_future = {"status": "ready", "effective_from": "2026-08-01T00:00:00Z"}
    active, reason = PolicyScopeFilter.is_policy_active(doc_future, as_of)
    assert active is False
    assert "not effective until" in reason


def test_active_generation_validation():
    """Verify candidate generation matches MongoDB authoritative active_generation."""
    tracking_gen2 = {"indexing_status": "indexed", "active_generation": 2}

    # Matches active generation 2
    valid, _ = PolicyScopeFilter.is_generation_active("doc1", 2, tracking_gen2)
    assert valid is True

    # Stale generation 1 rejected
    valid, reason = PolicyScopeFilter.is_generation_active("doc1", 1, tracking_gen2)
    assert valid is False
    assert "does not match" in reason

    # Missing tracking record rejected
    valid, reason = PolicyScopeFilter.is_generation_active("doc1", 1, None)
    assert valid is False
    assert "No tracking record" in reason


# ============================================================================
# 3. Retrieval Service End-to-End Tests
# ============================================================================


@pytest.mark.asyncio
async def test_retrieval_service_successful_query_flow():
    """Verify semantic retrieval ranks candidates by similarity and enforces scoping."""
    db = _FakeAsyncDatabase()
    doc_repo = DocumentRepository(db)
    indexing_repo = IndexingRepository(db)
    fake_chroma = _FakeChromaService()
    vector_repo = VectorIndexRepository(fake_chroma)
    provider = MockEmbeddingProvider()
    # Explicitly return vector matching chunk 0 to test similarity rank ordering
    provider.embed_texts = AsyncMock(return_value=[[0.05] * 1536])

    # 1. Setup Document in MongoDB
    doc_data = _build_test_document("doc_ret_1", "Leave Policy 2026.pdf", school_id="SCH-01")
    await db["documents"].replace_one({"document_id": "doc_ret_1"}, doc_data, upsert=True)

    # 2. Setup Tracking in MongoDB (active_generation = 1)
    tracking_data = {"document_id": "doc_ret_1", "active_generation": 1, "indexing_status": "indexed"}
    await db["document_indexing_tracking"].replace_one({"document_id": "doc_ret_1"}, tracking_data, upsert=True)

    # 3. Setup Vectors in ChromaDB (generation = 1)
    meta_0 = _build_test_chunk_meta("doc_ret_1", 0, generation=1, school_id="SCH-01", staff_category="teaching")
    meta_1 = _build_test_chunk_meta("doc_ret_1", 1, generation=1, school_id="SCH-01", staff_category="teaching")

    # Let chunk 0 have embedding closer to query
    vector_repo.upsert_vectors(
        ids=["doc_ret_1_chunk_0_g1", "doc_ret_1_chunk_1_g1"],
        embeddings=[[0.05] * 1536, [0.80] * 1536],
        documents=[
            "Permanent teachers receive 12 casual leaves annually.",
            "General attendance policy for school operations.",
        ],
        metadatas=[meta_0, meta_1],
    )

    service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
    )

    req = RetrievalRequest(
        question="How many casual leaves are available for teachers?",
        employee_context=EmployeeContext(staff_category="teaching"),
        top_k=2,
    )
    school_ctx = SchoolContext(school_id="SCH-01")

    res = await service.retrieve(req, school_ctx)

    assert res.retrieval_status == "success"
    assert res.total_results == 2
    assert res.query_understanding.intent == "leave"
    assert res.chunks[0].chunk_id == "doc_ret_1_chunk_0"
    assert res.chunks[0].similarity_score >= res.chunks[1].similarity_score
    assert "casual leaves" in res.chunks[0].content


@pytest.mark.asyncio
async def test_retrieval_service_multi_document_different_generations():
    """Verify documents with different active generations are properly validated."""
    db = _FakeAsyncDatabase()
    doc_repo = DocumentRepository(db)
    indexing_repo = IndexingRepository(db)
    fake_chroma = _FakeChromaService()
    vector_repo = VectorIndexRepository(fake_chroma)
    provider = MockEmbeddingProvider()

    # Doc A is on generation 1
    doc_a = _build_test_document("doc_A", "Doc A.pdf", school_id="SCH-01")
    await db["documents"].replace_one({"document_id": "doc_A"}, doc_a, upsert=True)
    await db["document_indexing_tracking"].replace_one(
        {"document_id": "doc_A"},
        {"document_id": "doc_A", "active_generation": 1, "indexing_status": "indexed"},
        upsert=True,
    )

    # Doc B is on generation 2
    doc_b = _build_test_document("doc_B", "Doc B.pdf", school_id="SCH-01")
    await db["documents"].replace_one({"document_id": "doc_B"}, doc_b, upsert=True)
    await db["document_indexing_tracking"].replace_one(
        {"document_id": "doc_B"},
        {"document_id": "doc_B", "active_generation": 2, "indexing_status": "indexed"},
        upsert=True,
    )

    meta_a = _build_test_chunk_meta("doc_A", 0, generation=1, school_id="SCH-01")
    meta_b = _build_test_chunk_meta("doc_B", 0, generation=2, school_id="SCH-01")
    meta_b_stale = _build_test_chunk_meta("doc_B", 1, generation=1, school_id="SCH-01")  # Stale generation for Doc B!

    vector_repo.upsert_vectors(
        ids=["doc_A_chunk_0_g1", "doc_B_chunk_0_g2", "doc_B_chunk_1_g1"],
        embeddings=[[0.1] * 1536, [0.1] * 1536, [0.1] * 1536],
        documents=["Content A", "Content B", "Stale Content B"],
        metadatas=[meta_a, meta_b, meta_b_stale],
    )

    service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
    )

    req = RetrievalRequest(question="Policy inquiry", top_k=5)
    res = await service.retrieve(req, SchoolContext(school_id="SCH-01"))

    returned_ids = [c.chunk_id for c in res.chunks]
    assert "doc_A_chunk_0" in returned_ids
    assert "doc_B_chunk_0" in returned_ids
    # Stale generation chunk from Doc B must be rejected!
    assert "doc_B_chunk_1" not in returned_ids


@pytest.mark.asyncio
async def test_retrieval_service_cross_tenant_isolation_filter():
    """Verify policies from School B are completely excluded when querying for School A."""
    db = _FakeAsyncDatabase()
    doc_repo = DocumentRepository(db)
    indexing_repo = IndexingRepository(db)
    fake_chroma = _FakeChromaService()
    vector_repo = VectorIndexRepository(fake_chroma)
    provider = MockEmbeddingProvider()

    # School A policy
    doc_a = _build_test_document("doc_school_A", school_id="SCH-A")
    await db["documents"].replace_one({"document_id": "doc_school_A"}, doc_a, upsert=True)
    await db["document_indexing_tracking"].replace_one(
        {"document_id": "doc_school_A"},
        {"document_id": "doc_school_A", "active_generation": 1, "indexing_status": "indexed"},
        upsert=True,
    )

    # School B policy
    doc_b = _build_test_document("doc_school_B", school_id="SCH-B")
    await db["documents"].replace_one({"document_id": "doc_school_B"}, doc_b, upsert=True)
    await db["document_indexing_tracking"].replace_one(
        {"document_id": "doc_school_B"},
        {"document_id": "doc_school_B", "active_generation": 1, "indexing_status": "indexed"},
        upsert=True,
    )

    meta_a = _build_test_chunk_meta("doc_school_A", 0, generation=1, school_id="SCH-A")
    meta_b = _build_test_chunk_meta("doc_school_B", 0, generation=1, school_id="SCH-B")

    vector_repo.upsert_vectors(
        ids=["doc_school_A_chunk_0_g1", "doc_school_B_chunk_0_g1"],
        embeddings=[[0.05] * 1536, [0.01] * 1536],  # School B is closer in vector space
        documents=["School A Policy Content", "School B Secret Policy Content"],
        metadatas=[meta_a, meta_b],
    )

    service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
    )

    # Query with School A scope
    req = RetrievalRequest(question="Tell me the policy", top_k=5)
    res = await service.retrieve(req, SchoolContext(school_id="SCH-A"))

    assert res.total_results == 1
    assert res.chunks[0].document_id == "doc_school_A"
    assert "School B" not in res.chunks[0].content


@pytest.mark.asyncio
async def test_retrieval_service_overfetching_when_filtering_removes_candidates():
    """Verify that when first candidates are filtered out, overfetching satisfies top_k."""
    db = _FakeAsyncDatabase()
    doc_repo = DocumentRepository(db)
    indexing_repo = IndexingRepository(db)
    fake_chroma = _FakeChromaService()
    vector_repo = VectorIndexRepository(fake_chroma)
    provider = MockEmbeddingProvider()

    # Valid doc
    doc_valid = _build_test_document("doc_val", school_id="SCH-01")
    await db["documents"].replace_one({"document_id": "doc_val"}, doc_valid, upsert=True)
    await db["document_indexing_tracking"].replace_one(
        {"document_id": "doc_val"},
        {"document_id": "doc_val", "active_generation": 1, "indexing_status": "indexed"},
        upsert=True,
    )

    # Expired doc
    doc_exp = _build_test_document("doc_exp", school_id="SCH-01", status="expired")
    await db["documents"].replace_one({"document_id": "doc_exp"}, doc_exp, upsert=True)
    await db["document_indexing_tracking"].replace_one(
        {"document_id": "doc_exp"},
        {"document_id": "doc_exp", "active_generation": 1, "indexing_status": "indexed"},
        upsert=True,
    )

    # Upsert expired candidates with lower distance, and valid candidates with slightly higher distance
    vector_repo.upsert_vectors(
        ids=["doc_exp_chunk_0_g1", "doc_exp_chunk_1_g1", "doc_val_chunk_0_g1"],
        embeddings=[[0.01] * 1536, [0.02] * 1536, [0.05] * 1536],
        documents=["Expired 1", "Expired 2", "Valid Policy Content"],
        metadatas=[
            _build_test_chunk_meta("doc_exp", 0, 1, school_id="SCH-01"),
            _build_test_chunk_meta("doc_exp", 1, 1, school_id="SCH-01"),
            _build_test_chunk_meta("doc_val", 0, 1, school_id="SCH-01"),
        ],
    )

    service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
        candidate_pool_size=5,
    )

    req = RetrievalRequest(question="Valid policy question", top_k=1)
    res = await service.retrieve(req, SchoolContext(school_id="SCH-01"))

    assert res.total_results == 1
    assert res.chunks[0].chunk_id == "doc_val_chunk_0"


@pytest.mark.asyncio
async def test_retrieval_service_empty_results():
    """Verify empty response envelope when no vectors match."""
    db = _FakeAsyncDatabase()
    doc_repo = DocumentRepository(db)
    indexing_repo = IndexingRepository(db)
    fake_chroma = _FakeChromaService()
    vector_repo = VectorIndexRepository(fake_chroma)
    provider = MockEmbeddingProvider()

    service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
    )

    req = RetrievalRequest(question="Question with no indexed docs", top_k=5)
    res = await service.retrieve(req, SchoolContext(school_id="SCH-EMPTY"))

    assert res.retrieval_status == "empty"
    assert res.total_results == 0
    assert len(res.chunks) == 0


@pytest.mark.asyncio
async def test_retrieval_service_embedding_failure_raises_502():
    """Verify provider failure raises EmbeddingException with 502 status."""
    db = _FakeAsyncDatabase()
    doc_repo = DocumentRepository(db)
    indexing_repo = IndexingRepository(db)
    fake_chroma = _FakeChromaService()
    vector_repo = VectorIndexRepository(fake_chroma)

    provider = MagicMock(spec=BaseEmbeddingProvider)
    provider.embed_texts = AsyncMock(side_effect=RuntimeError("Azure connection error"))

    service = RetrievalService(
        document_repo=doc_repo,
        indexing_repo=indexing_repo,
        vector_repo=vector_repo,
        embedding_provider=provider,
    )

    req = RetrievalRequest(question="Will fail on embedding")
    with pytest.raises(EmbeddingException) as exc_info:
        await service.retrieve(req, SchoolContext(school_id="SCH-01"))

    assert exc_info.value.status_code == 502
    assert exc_info.value.error_code == "AZURE_OPENAI_ERROR"


# ============================================================================
# 4. API Endpoint Integration Tests
# ============================================================================


def test_api_retrieve_success():
    """Verify POST /api/v1/query/retrieve returns 200 with complete envelope."""
    client = TestClient(app)
    mock_service = MagicMock()

    mock_response = RetrievalResponse(
        query_understanding=QueryUnderstandingResult(
            original_query="What is the leave policy?",
            normalized_query="What is the leave policy?",
            intent="leave",
            extracted_keywords=["leave"],
            requested_scope={"school_id": "SCH-API"},
            as_of_date=datetime.now(timezone.utc),
        ),
        chunks=[
            RetrievedChunk(
                chunk_id="doc_1_chunk_0",
                document_id="doc_1",
                document_name="Leave.pdf",
                content="Casual leave is 12 days.",
                page_start=1,
                page_end=1,
                similarity_score=0.92,
                school_id="SCH-API",
            )
        ],
        total_candidates=5,
        total_results=1,
        retrieval_status="success",
        retrieved_at=datetime.now(timezone.utc),
    )
    mock_service.retrieve = AsyncMock(return_value=mock_response)

    app.dependency_overrides[get_retrieval_service] = lambda: mock_service
    try:
        response = client.post(
            "/api/v1/query/retrieve",
            headers={"X-School-ID": "SCH-API"},
            json={
                "question": "What is the leave policy?",
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["retrieval_status"] == "success"
        assert len(data["data"]["chunks"]) == 1
        assert data["data"]["chunks"][0]["similarity_score"] == 0.92
    finally:
        app.dependency_overrides.pop(get_retrieval_service, None)


def test_api_retrieve_empty_question_validation_error():
    """Verify POST /api/v1/query/retrieve rejects empty question with 422/400."""
    client = TestClient(app)
    mock_service = MagicMock()
    app.dependency_overrides[get_retrieval_service] = lambda: mock_service
    try:
        response = client.post(
            "/api/v1/query/retrieve",
            headers={"X-School-ID": "SCH-01"},
            json={"question": ""},
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_retrieval_service, None)
