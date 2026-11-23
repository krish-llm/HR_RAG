# HR Policy RAG Platform

Enterprise Multi-Tenant HR Policy Retrieval-Augmented Generation (RAG) platform built with **FastAPI**, **MongoDB**, **ChromaDB**, **Azure OpenAI**, and a **Streamlit** testing playground.

---

## 🏛️ Architecture Overview

The system follows a strict **Modular Monolith** architecture designed for educational networks (trusts, schools, and branches). It features multi-tenancy, canonical scoping, deterministic query understanding, composite reranking, prompt-injection defense, and grounded question answering with verified citations.

### End-to-End Pipeline
```
[ PDF Policy Upload ]
        │
        ▼ (Stage 1: Ingestion & Storage)
   `documents` (MongoDB Metadata & PDF Storage)
        │
        ▼ (Stage 2: Extraction via PyMuPDF4LLM)
   `document_extractions` (Page-level Markdown)
        │
        ▼ (Stage 3: Normalization & Cleaning)
   `document_processed_content` (Header/Footer-stripped Structure)
        │
        ▼ (Stage 4: Semantic Chunking via tiktoken)
   `document_chunks` (Heading-aware, Hierarchical Metadata)
        │
        ▼ (Stage 5: Embeddings & Vector Indexing)
   `hr_policy_chunks` (ChromaDB 1536-dim text-embedding-3-small)
        │
        ▼ (Stage 6: Query Understanding & Vector Search)
   Semantic Nearest Neighbors + Multi-layer MongoDB Validation
        │
        ▼ (Stage 7: Reranking & Context Assembly)
   Deterministic V1 Reranker + Budget Packing + Source Manifest
        │
        ▼ (Stage 8: Grounded LLM Generation)
   Grounded Answer + Citation Validation against Manifest
        │
        ▼ (Stage 9: Quality Measurement & Benchmarking)
   Evaluation Runner (P@K, R@K, MRR, nDCG, Point Coverage, Citation Validity)
        │
        ▼ (Stage 10: Streamlit UI Testing Playground)
   Interactive Multi-Page Frontend
```

---

## 📁 Directory Structure

```
HRPolicyRAG/
├── app/                         # FastAPI Monolithic Backend
│   ├── api/                     # Central API routing & dependencies
│   ├── core/                    # Centralized error handling (AppException, GenerationException)
│   ├── db/                      # Asynchronous MongoDB manager & ChromaDB persistent service
│   ├── modules/                 # Cohesive domain modules
│   │   ├── ingestion/           # Phase 1: PDF upload, file storage, document registry
│   │   ├── extraction/          # Phase 2: PyMuPDF4LLM page extraction
│   │   ├── processing/          # Phase 3: Rule-based text cleaning & noise removal
│   │   ├── chunking/            # Phase 4: Heading-aware semantic chunking
│   │   ├── indexing/            # Phase 5: Azure OpenAI embedding & ChromaDB vector store
│   │   ├── retrieval/           # Phase 6: Query understanding & vector retrieval
│   │   ├── reranking/           # Phase 7: Deterministic V1 composite reranker
│   │   ├── context/             # Phase 7: Context builder, token budget & source manifest
│   │   ├── generation/          # Phase 8: Azure OpenAI grounded answering & citation validator
│   │   └── evaluation/          # Phase 9: Evaluation runner, metric calculators, CLI
│   └── schemas/                 # Shared envelopes (ApiResponse, ErrorResponse) & BaseDocument
├── data/
│   ├── chroma/                  # Persistent ChromaDB vector database
│   ├── uploads/                 # Ingested PDF policy files
│   └── evaluation/              # Benchmark datasets (dataset_v1.json) & generated reports
├── frontend/                    # Streamlit Testing Playground (Phase 10)
│   ├── app.py                   # Main Streamlit dashboard & tenant scope navigation
│   ├── api_client.py            # Reusable HTTP API client with tenant headers
│   ├── .env.example             # Frontend environment variables template
│   └── pages/                   # Multi-page dashboard
│       ├── 1_Ask_HR_Policy.py         # Grounded Q&A, citations, evidence warning alerts
│       ├── 2_Document_Management.py   # Upload PDF, 5-stage pipeline, library table
│       ├── 3_Retrieval_Inspector.py   # Vector candidates, reranking & context inspection
│       └── 4_Evaluation_Dashboard.py  # Benchmark runner, metrics tables, CSV/JSON exports
├── tests/                       # Comprehensive automated test suite (193 tests)
├── .env.example                 # Backend environment variables template
├── requirements.txt             # Backend dependencies
├── requirements-ui.txt          # Frontend dependencies
└── ARCHITECTURE.md              # Architectural standards & domain specifications
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.11+
- Local or remote MongoDB instance (e.g. `mongodb://localhost:27017`)
- Azure OpenAI instance (for embeddings and chat completions)

### 2. Install Backend Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 3. Install Frontend Dependencies
```powershell
python -m pip install -r requirements-ui.txt
```

### 4. Configure Environment Variables

**Backend (`.env`):**
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
Key configuration settings:
- `MONGODB_URL`: MongoDB connection URI (default: `mongodb://localhost:27017`)
- `MONGODB_DB_NAME`: Database name (default: `hr_policy_rag`)
- `CHROMA_PERSIST_DIRECTORY`: Path for ChromaDB vector storage (default: `./data/chroma`)
- `AZURE_OPENAI_ENDPOINT`: Endpoint URL for Azure OpenAI
- `AZURE_OPENAI_API_KEY`: API Key
- `AZURE_OPENAI_API_VERSION`: API version (e.g. `2024-12-01-preview`)
- `AZURE_OPENAI_CHAT_DEPLOYMENT`: Chat deployment (e.g. `gpt-4o` or `gpt-4.1-mini`)
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`: Embedding deployment (`text-embedding-3-small`)
- `CONTEXT_TOKEN_BUDGET`: Default context token budget (default: `2000`)
- `RERANK_TOP_K`: Candidate count fed to reranker (default: `10`)

**Frontend (`frontend/.env`):**
Copy `frontend/.env.example` to `frontend/.env`:
```powershell
cp frontend/.env.example frontend/.env
```
Settings:
- `BACKEND_API_URL`: Backend URL (default: `http://localhost:8000/api/v1`)
- `API_TIMEOUT_SECONDS`: Request timeout in seconds (default: `60.0`)

---

## 🏃 Launching the Application

### 1. Start the FastAPI Backend
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### 2. Start the Streamlit Testing Playground
In a separate terminal:
```powershell
streamlit run frontend/app.py
```
Opens in your browser at: [http://localhost:8501](http://localhost:8501)

---

## 📖 End-to-End Workflow Guide

### A. Upload and Index a Policy Document
1. Navigate to **Document Management** in Streamlit (`http://localhost:8501`).
2. Select a PDF file (e.g., `Leave_Policy.pdf`).
3. Set the metadata:
   - **School ID**: `sch-delhi-01`
   - **Group ID**: `grp-national-01`
   - **Classification**: `policy`
   - **Version**: `1`
4. Click **Start Ingestion Pipeline**. The UI sequentially executes:
   - `POST /api/v1/documents/upload`
   - `POST /api/v1/documents/{document_id}/extract`
   - `POST /api/v1/documents/{document_id}/process`
   - `POST /api/v1/documents/{document_id}/chunk`
   - `POST /api/v1/documents/{document_id}/index`
5. The policy is now stored in MongoDB and indexed in ChromaDB under generation `g1`.

### B. Ask HR Policy Questions
1. Navigate to **Ask HR Policy**.
2. Set your school/tenant scope in the sidebar (`School ID: sch-delhi-01`).
3. Enter your question:
   > *"How many days of casual leave are full-time teachers entitled to annually?"*
4. Click **Submit Question**.
5. The system returns:
   - Grounded natural language answer.
   - Structured citations mapping claims to document name, version, section heading, and page range.
   - Evidence state flags (highlights if insufficient evidence or conflicting policies are detected).

### C. Inspect Retrieval and Context Packing
1. Navigate to **Retrieval Inspector**.
2. Query a term to inspect:
   - Deterministic query normalization and keyword extraction.
   - Raw ChromaDB vector search similarity scores.
   - Composite reranking scores (Semantic + Lexical + Heading match).
   - Complete chunk token packing within the configured token budget.

### D. Run Quality Evaluations
1. Navigate to **Evaluation Dashboard**.
2. Click **Run Benchmark** (or run via CLI below).
3. Inspect aggregate metrics:
   - **Precision@K**, **Recall@K**, **MRR**, and **nDCG@K**.
   - **Factual Point Coverage**, **Citation Validity**, and **Abstention Accuracy**.
4. Download full timestamped JSON and CSV reports.

---

## 🛠️ CLI Evaluation Runner

To run benchmark evaluations headless (e.g. in CI/CD pipelines):

```powershell
# Live pipeline evaluation:
python -m app.modules.evaluation.cli --dataset data/evaluation/dataset_v1.json --output-dir data/evaluation/reports

# Offline dry-run with mock services:
python -m app.modules.evaluation.cli --mock

# Retrieval-only evaluation:
python -m app.modules.evaluation.cli --no-answer-eval
```

---

## 🧪 Automated Test Suite

Run the full pytest suite covering all 10 phases:

```powershell
python -m pytest -v
```

All 193 automated tests verify:
- Phase 0: Foundations, DB lifecycles, configuration, and multi-tenant scoping.
- Phase 1: Ingestion, SHA-256 deduplication, atomic rollbacks.
- Phase 2: PyMuPDF4LLM extraction and page boundary tracking.
- Phase 3: Rule-based text cleaning and structural normalization.
- Phase 4: Heading-aware chunking and transaction/staging re-chunking safety.
- Phase 5: Azure OpenAI embeddings, ChromaDB generation-based indexing.
- Phase 6: Query understanding, tenant isolation filters, and semantic search.
- Phase 7: Deterministic V1 reranker and token-budgeted context building.
- Phase 8: Grounded LLM answering, citation validation, and prompt injection defense.
- Phase 9: Evaluation metrics (P@K, R@K, MRR, nDCG), runners, and JSON/CSV reports.
- Phase 10: Streamlit frontend API client and tenant header routing.
