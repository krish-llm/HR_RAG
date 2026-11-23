# HR Policy RAG — System Architecture & Development Standards

## 1. Architectural Philosophy: Modular Monolith

The HR Policy RAG platform is structured as a **Modular Monolith**. It delivers the maintainability, clear boundaries, and team scalability of microservices while eliminating network latency, deployment fragmentation, and operational complexity.

### Core Architectural Principles
- **Loose Coupling, High Cohesion**: Functional domains (e.g., policies, school tenants, search/RAG) are self-contained modules under `app/modules/`.
- **Single Process Execution**: The entire system runs within a single unified FastAPI process managed by an asynchronous lifespan handler.
- **Fixed Boundaries**: Modules interact exclusively through well-defined public contracts, never reaching into the private internals or database collections of other modules.

```
+-----------------------------------------------------------------------------------+
|                               FastAPI Application                                 |
|                                                                                   |
|  [ API Layer: /api/v1 (Routers, Versioning, Request Validation, Error Handling) ] |
+-----------------------------------------+-----------------------------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                 |                                 |
        v                                 v                                 v
+-------------------+            +-------------------+            +-------------------+
| Policies Module   |            |  Tenants Module   |            |  Future Modules   |
| (Contracts, DTOs, |            | (School Hierarchy,|            | (RAG, Ingestion,  |
|  Business Logic)  |            |  Scope Resolver)  |            |  Chat Services)   |
+---------+---------+            +---------+---------+            +---------+---------+
          |                                |                                |
          +--------------------------------+--------------------------------+
                                          |
                      +-------------------+-------------------+
                      |   Dependency Injection (FastAPI Depends)
                      v                                       v
         +--------------------------+           +--------------------------+
         | MongoDB Manager (Async)  |           | ChromaDB Service (Local) |
         | - Official PyMongo Async |           | - Persistent local store |
         | - Authoritative state    |           | - Semantic vector index  |
         | - Versioned documents    |           | - Scoped metadata filters|
         +--------------------------+           +--------------------------+
```

---

## 2. Independent Modules & Fixed Contracts

### 2.1 Module Structure
Every business domain module inside `app/modules/<module_name>/` must follow this structure:
```
app/modules/<module_name>/
├── __init__.py          # Public exports (Schemas, Services, Router)
├── router.py            # FastAPI endpoints specific to this module
├── schemas.py           # Module-specific Pydantic input/output contracts
├── service.py           # Core business logic and operations
└── repository.py        # Database queries (MongoDB / ChromaDB)
```

### 2.2 Fixed Pydantic Contracts
- Modules communicate using immutable Pydantic models for both inputs (Commands/Queries) and outputs (DTOs/Responses).
- Internal database representations (e.g., Raw BSON documents or ChromaDB result dictionaries) must **never** leak across module boundaries. They must be parsed into explicit Pydantic response models at the repository/service layer.

---

## 3. Dependency Injection & Circular Dependency Prevention

### 3.1 Dependency Injection Standards
- All database connections, configurations, and shared services are injected via FastAPI's `Depends()` framework.
- Handlers must not instantiate database clients or configuration objects directly inside route functions:
  ```python
  # CORRECT:
  @router.get("/example")
  async def get_example(
      db: AsyncDatabase = Depends(get_mongo_db),
      chroma: ChromaDBService = Depends(get_chroma_service),
      settings: Settings = Depends(get_settings),
  ):
      ...
  ```

### 3.2 Preventing Circular Dependencies
- **Unidirectional Layering**: 
  `main.py` ➔ `api/` ➔ `modules/` ➔ `db/` / `schemas/` ➔ `core/` / `config.py`
- Shared types and cross-cutting models belong in `app/schemas/` or `app/core/`, never imported from sibling modules.
- Utility functions in `app/utils/` must remain pure and free from database or API route imports.

---

## 4. Database Roles & Responsibilities

The system deliberately separates document storage from vector indexing.

| Feature / Responsibility | MongoDB (PyMongo Async API) | ChromaDB (PersistentClient) |
| :--- | :--- | :--- |
| **Role** | Authoritative Source of Truth | Semantic Search & Vector Cache |
| **Data Stored** | Full policy text, metadata, school hierarchy, revision history, audit trails | Text chunk embeddings, chunk text, scope metadata for vector filtering |
| **API Client** | Official `pymongo.AsyncMongoClient` | `chromadb.PersistentClient` |
| **Consistency** | Strong document consistency | Eventual consistency (re-indexable from MongoDB) |
| **Lifecycle** | Connection pool opened at startup, closed at shutdown | Persistent filesystem directory (`./data/chroma`) |

> [!IMPORTANT]
> If ChromaDB storage is corrupted or deleted, the system can rebuild all vector collections and embeddings directly from the authoritative documents stored in MongoDB.

### 4.1 Document Pipeline & MongoDB Collections

The ingestion, extraction, and processing pipeline follows a non-destructive stage progression across dedicated collections:

```
[ Upload PDF ] ──> `documents` (Ingestion & Scoping Metadata)
                        │
                        ▼ (Extract)
                   `document_extractions` (Raw Page-level Markdown via PyMuPDF4LLM)
                        │
                        ▼ (Clean & Normalize)
                   `document_processed_content` (Cleaned, Header/Footer-stripped, Structure-tagged)
                        │
                        ▼ (Chunk)
                   `document_chunks` (Heading-aware, Section-isolated, Hierarchical Metadata)
                        │
                        ▼ (Embed & Index)
                   `document_chunk_embeddings` (MongoDB Cache) & `hr_policy_chunks` (ChromaDB Vector Store)
```

1. **`documents`**: Ingested file registrations, file hashes (SHA-256), scope dimensions, and lifecycle status (`uploaded`, `processing`, `ready`, `failed`).
2. **`document_extractions`**: Original extracted Markdown page content, page boundaries, and `extraction_status` (`completed`, `failed`). Unique index on `document_id`.
3. **`document_processed_content`**: Rule-based normalized text, stripped repeated headers/footers and noise, detected structural metadata (headings, tables, lists, paragraphs), linked via `document_id` and `source_extraction_id`. Unique index on `document_id`. Safe upsert supports idempotent reprocessing. Extraction records remain intact upon reprocessing.
4. **`document_chunks`**: Heading-aware semantic chunks with recursive fallback and repeated table headers. Each chunk embeds 22 metadata fields including canonical school scope, employee applicability, `heading`, `heading_path` breadcrumbs, and exact `source_pages` (`page_start`, `page_end`). Safe dual-path persistence (MongoDB transactions with staging fallback) preserves valid chunks during re-chunking. Unique index on `chunk_id`.
5. **`document_chunk_embeddings` & `document_indexing_tracking` (MongoDB)**: Authoritative vector caching and progress tracking. Chunk embeddings are cached against their SHA-256 `content_hash` and `embedding_model`, preventing redundant Azure OpenAI API calls when chunk content has not changed. Document-level tracking manages active/target index generations and pipeline stages: `chunked` ➔ `embedding` ➔ `indexing` ➔ `indexed` (or `failed`).
6. **`hr_policy_chunks` (ChromaDB Vector Store)**: Dedicated vector collection containing 1536-dimensional `text-embedding-3-small` vectors, original Markdown chunk documents, and sanitized scalar metadata (21 dimensions including `generation`, tenant scopes, applicability, validity dates in ISO 8601, and section headings). Employs lightweight generation-based indexing:
   - Vectors are written under a generation-scoped identifier (`{document_id}_chunk_{index}_g{generation}`).
   - Existing active generation remains available for retrieval during re-indexing.
   - All written vectors are verified against expected count before activation.
   - The new generation is activated in MongoDB tracking upon successful verification.
   - Previous generation vectors are purged only after activation.
   - On write/verification failure, unactivated generation vectors are purged while the existing active generation is retained intact.
7. **Query Understanding & Semantic Retrieval (`app/modules/retrieval`)**:
   - **Deterministic Query Understanding**: Whitespace normalization, HR intent classification (`leave`, `compensation`, `benefits`, `attendance`, `conduct`, `probation`, `termination`, `recruitment`), stop-word filtered keyword extraction, and explicit employee context parsing.
   - **Query Embedding**: Generates 1536-dim vector via official `BaseEmbeddingProvider` reusing `AzureOpenAIEmbeddingProvider`.
   - **ChromaDB Vector Search with Candidate Over-fetching**: Queries nearest neighbors with a configurable candidate pool size (`DEFAULT_CANDIDATE_POOL_SIZE`, default 20) larger than `top_k`, with automated over-fetching retry when filtering reduces candidates below `top_k`.
   - **Authoritative Multi-Layer MongoDB Validation**:
     - Document policy status (rejects draft, superseded, expired, failed).
     - Temporal validity: checks `effective_from` and `effective_to` against `as_of_date`.
     - Active index generation: validates candidate vector generation against document's authoritative `active_generation` in `document_indexing_tracking`.
     - Strict tenant isolation: branch, school, group, state, and global isolation.
     - Employee applicability: staff category, employment type, and department matching.
   - **Ranking & Deduplication**: Avoids duplicate chunks and ranks valid candidates by similarity score in $[0.0, 1.0]$.
8. **Reranking & Context Building (`app/modules/reranking`, `app/modules/context`)**:
   - **Deterministic V1 Reranker (`DeterministicV1Reranker`)**: Lightweight, model-free composite reranker computing a normalized relevance score $S_{\text{rerank}} = \min(1.0, w_{\text{sem}} S_{\text{sem}} + w_{\text{lex}} S_{\text{lex}} + w_{\text{head}} S_{\text{head}})$. Combines vector similarity ($S_{\text{sem}}$), lexical term coverage and keyword density ($S_{\text{lex}}$), and heading breadcrumb alignment ($S_{\text{head}}$). Both rerank scores and semantic scores are explicitly documented as ranking heuristics, not calibrated probabilities.
   - **Replaceable Strategy Pattern (`BaseReranker`)**: Clean abstract interface allowing future neural or cross-encoder rerankers to be plugged in without contract changes.
   - **Conservative Deduplication**: Prunes exact duplicates and near-duplicate chunks (high token overlap under identical headings) while strictly safeguarding distinct policy clauses across sections.
   - **Strict Token Budget Enforcement (`ContextBuilder`)**: Evaluates token limits using `tiktoken` (`cl100k_base`). Ensures complete chunks are preserved—oversized chunks that exceed the remaining token budget are safely skipped rather than truncated mid-clause, mid-sentence, or mid-table.
   - **Structured Source Manifest & Citations**: Formats standardized citation headers (`[Source 1: Policy Name, pages 2–3, Heading Path]`) and compiles a structured manifest mapping citations to authoritative document UUIDs, chunk IDs, versions, page ranges, and tenant scoping dimensions.
   - **Context API Endpoint**: `POST /api/v1/query/context` executing the full `Query ➔ Retrieval ➔ Reranking ➔ Context Assembly` pipeline while preserving `POST /api/v1/query/retrieve`.
9. **Grounded LLM Answer Generation (`app/modules/generation`)**:
   - **Replaceable Provider Abstraction (`BaseLLMProvider`)**: Extensible interface for generative completion engines. Supports `AzureOpenAIChatProvider` using the official `AsyncAzureOpenAI` client, configured via environment settings without hardcoded credentials.
   - **Strict Grounding & Prompt Injection Defense**: System instructions enforce factual answers drawn exclusively from `<context>`. Context documents are treated strictly as untrusted data, neutralizing adversarial instructions or prompt overrides embedded within policy documents.
   - **Mandatory In-Text Citations**: Every factual claim requires citation tags (e.g. `[Source 1]`). The service validates all cited labels against the Phase 7 `source_manifest`, stripping unsupported or fabricated labels and recording pruning actions in `limitations`.
   - **Insufficient & Conflicting Evidence Handling**:
     - Fast-path insufficient evidence: If retrieval yields 0 chunks or empty context, the service returns an explicit insufficient-evidence response immediately without calling the LLM.
     - Contextual insufficient evidence: If the retrieved policy does not answer the question, the LLM sets `insufficient_evidence: true` and describes gaps in `limitations`.
     - Conflicting policies: When overlapping documents have contradictory rules, the LLM sets `conflicting_evidence: true`, highlights both provisions, cites both sources, and logs details in `limitations` without arbitrarily choosing one.
   - **Answer API Endpoint**: `POST /api/v1/query/answer` reuses the entire pipeline: `Query ➔ Retrieval ➔ Reranking ➔ Context Assembly ➔ LLM Generation ➔ Citation Validation ➔ Answer Response`.
10. **RAG Evaluation & Quality Measurement (`app/modules/evaluation`)**:
    - **Versioned Evaluation Dataset**: Versioned synthetic HR benchmark datasets under `data/evaluation/dataset_v1.json` with test case definitions covering canonical school scoping, expected documents, expected answer points, expected citations, insufficient evidence conditions, and conflicting evidence flags.
    - **Retrieval & Ranking Evaluation (`RetrievalEvaluator`)**: Evaluates Phase 6/7 retrieval quality using standard information retrieval metrics:
      - Precision@K ($\frac{|\text{Retrieved@K} \cap \text{Relevant}|}{K}$)
      - Recall@K ($\frac{|\text{Retrieved@K} \cap \text{Relevant}|}{|\text{Relevant}|}$)
      - Mean Reciprocal Rank (MRR, $\frac{1}{\text{rank of first relevant document}}$)
      - Normalized Discounted Cumulative Gain (nDCG@K)
    - **Answer & Citation Evaluation (`AnswerEvaluator`)**: Evaluates Phase 8 grounded answer generation using deterministic multi-aspect scoring:
      - Point Coverage: proportion of key factual claims present in the answer.
      - Citation Validity: ratio of cited source tags that strictly exist within the authoritative Phase 7 context manifest (no fabricated references).
      - Citation Coverage: proportion of expected citations referenced in the output.
      - Insufficient & Conflicting Evidence Accuracy: evaluates whether the system abstains or highlights conflicts appropriately.
      - Unsupported Claims Detection: flags fabricated or ungrounded citation tags.
      - Human Review Flagging: clearly distinguishes automatically measurable metrics from ambiguous cases where human review is advised.
    - **CLI Runner & Report Exporter (`cli.py`, `runner.py`)**: Standalone evaluation runner generating summary CLI tables, latency percentiles (Mean, P95), and exporting timestamped JSON and CSV reports to `data/evaluation/reports/`.
11. **Hybrid Retrieval (Dense + BM25 + RRF) (`app/modules/bm25`, `app/modules/retrieval`)**:
    - **Persistent SQLite FTS5 BM25 Provider (`SQLiteFTSBM25IndexProvider`)**: Replaceable `BaseBM25IndexProvider` implementation using SQLite's native FTS5 extension with Porter stemming (`tokenize='porter unicode61'`). Stores chunk text and rich metadata (`chunk_id`, `document_id`, `generation`, scoping, employee applicability, heading, dates) in a persistent database file with WAL mode.
    - **Generation Consistency & Atomic Activation**: Document indexing writes target generation chunks to both ChromaDB and SQLite BM25. Generation counts are verified in both stores before atomic activation in MongoDB tracking. Stale generations are purged only after activation; failed runs roll back unactivated chunks without affecting the active generation.
    - **Rebuild Capability from Source of Truth**: Full and per-document index rebuilding (`POST /api/v1/documents/rebuild-bm25`) reconstructs the SQLite BM25 index directly from authoritative MongoDB chunks.
    - **Concurrent Hybrid Execution**: In `RETRIEVAL_MODE="hybrid"`, `RetrievalService` executes dense vector search and BM25 full-text search concurrently via `asyncio.to_thread` and `asyncio.gather`.
    - **Reciprocal Rank Fusion (RRF)**: Merges dense and BM25 candidate lists using $RRF_{\text{score}} = \sum \frac{1}{k + \text{rank}_i}$ with configurable smoothing parameter $k$ (`HYBRID_RRF_K`, default 60). Candidates are deduplicated by `chunk_id`, and original modality scores and ranks are preserved for transparency and debugging.
    - **Authoritative MongoDB Scoping for Both Streams**: Both dense and BM25 candidates undergo strict policy lifecycle validation, effective date verification, active generation matching, educational hierarchy scoping, and employee applicability filtering.
    - **Structured Logging & Execution Timing**: Granular `INFO`-level structured logs measure `total_retrieval`, `dense_retrieval`, `bm25_retrieval`, `mongo_validation`, and `rrf_fusion` in milliseconds with correlation request IDs.

---

## 5. Multi-Tenancy & School Scoping Hierarchy

### 5.1 Canonical School Scope Dimensions
Our domain models educational networks consisting of trusts, schools, and branches. We avoid opaque generic `tenant_id` strings and use explicit canonical scope dimensions:

- `is_global` (*bool*): When `True`, the policy applies universally across all institutions.
- `group_id` (*Optional[str]*): The overarching educational trust or society (e.g., DPS Society, Ryan Group).
- `school_id` (*Optional[str]*): The specific school institution within a group or standalone.
- `branch_id` (*Optional[str]*): Specific campus or branch within a school.
- `state` (*Optional[str]*): Geographical jurisdiction (e.g., Karnataka, Maharashtra, Delhi) governing state-level labor, leave, and holiday compliance.

### 5.2 Separation of Tenant Context from Authentication
- **Request Context is NOT an Identity**: Context parameters passed via headers (`X-School-ID`, `X-Group-ID`, `X-State`) or query parameters represent *requested search or routing filters*.
- **No Independent Source of Truth**: Request context headers must **never** be treated as verified authorization credentials. Authentication and permission verification must always occur against verified authentication sessions (e.g., signed JWT tokens or user session records) in dedicated authentication guards.
- **Hierarchical Policy Resolution Order**:
  When resolving policies for a staff member at Branch $B$, School $S$, Group $G$, in State $St$:
  $$\text{Branch Specific } (B) \succ \text{School Specific } (S) \succ \text{Group Specific } (G) \succ \text{Global Policy } (\text{is\_global}) \quad [\text{Filtered by State } St]$$

---

## 6. Document Versioning & Auditability

Policies undergo regular revisions and regulatory amendments.
1. **Monotonic Version Number**: Every document in `BaseDocument` possesses an integer `version` field starting at `1`.
2. **Immutability of Historical Versions**: When a policy is amended, the prior version record in MongoDB is archived rather than overwritten.
3. **Audit Trail**: Every record automatically maintains UTC timestamps (`created_at`, `updated_at`) and arbitrary `metadata` tracking policy authors, approval references, and change summaries.

---

## 7. Error Handling, Logging & Testing Standards

### 7.1 Error Handling
- All business exceptions must inherit from `AppException` located in `app.core.exceptions`.
- All errors returned by the API conform strictly to the `ErrorResponse` envelope schema:
  ```json
  {
    "success": false,
    "error_code": "RESOURCE_NOT_FOUND",
    "message": "Policy document not found",
    "details": null,
    "timestamp": "2026-11-18T05:00:00.000000Z"
  }
  ```
- Unhandled server exceptions are intercepted by `unhandled_exception_handler` to log stack traces internally while returning a sanitized 500 error to clients.

### 7.2 Centralized Logging
- Python's standard `logging` module is configured in `app/logging_config.py`.
- Format: `%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s`.
- Secrets, credentials, and PII must never be written to logs.

### 7.3 Testing Standards
- Test suites are built using `pytest` and `pytest-asyncio`.
- External resources are isolated during testing:
  - ChromaDB uses a per-test temporary directory (`tempfile.mkdtemp`), cleaned up after each run.
  - MongoDB connection methods (`connect`, `ping`, `close`) are verified both with live local instances and mocked unit tests.
- Every API endpoint must have automated test coverage verifying both 2xx success cases and 4xx/5xx failure responses.

---

## 8. Development Discipline & Constraints

- **No Unnecessary Abstractions**: Do not introduce heavy wrapper frameworks (e.g. LangChain, LlamaIndex, or heavyweight ORMs). Utilize native, performant SDKs (`pymongo`, `chromadb`, Azure OpenAI SDK).
- **No Secret Hardcoding**: Secrets and credentials must strictly load through `pydantic-settings` from `.env`. `.env` is git-ignored, and only `.env.example` with non-sensitive placeholders is committed.
- **Do Not Rewrite Working Code**: Refinements must preserve established, tested functionality while enhancing compliance with architectural standards.
