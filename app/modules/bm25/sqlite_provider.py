"""SQLite FTS5 BM25 index provider implementation."""

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Tuple

from app.core.exceptions import AppException
from app.modules.bm25.base import BaseBM25IndexProvider
from app.modules.bm25.schemas import BM25ChunkRecord, BM25SearchResult
from app.schemas.common import current_utc_time

logger = logging.getLogger(__name__)


class SQLiteFTSBM25IndexProvider(BaseBM25IndexProvider):
    """BM25 index provider leveraging SQLite FTS5 for fast, persistent keyword retrieval."""

    def __init__(self, db_path: str = "./data/bm25/fts5_index.db") -> None:
        self.db_path = db_path
        self._is_memory = db_path == ":memory:"
        self._memory_con: Optional[sqlite3.Connection] = None

        # 1. Verify SQLite FTS5 availability
        self._verify_fts5_support()

        # 2. Ensure database directory exists if file-backed
        if not self._is_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # 3. Initialize schema and configure PRAGMA
        self._init_schema()
        logger.info("SQLiteFTSBM25IndexProvider initialized (path: %s)", self.db_path)

    def _verify_fts5_support(self) -> None:
        """Verify that SQLite was compiled with the FTS5 module enabled."""
        try:
            test_con = sqlite3.connect(":memory:")
            test_con.execute("CREATE VIRTUAL TABLE _test_fts5 USING fts5(x);")
            test_con.execute("DROP TABLE _test_fts5;")
            test_con.close()
        except Exception as exc:
            logger.error("SQLite FTS5 check failed: %s", exc)
            raise AppException(
                message=f"SQLite FTS5 extension is not available or unsupported: {exc}",
                error_code="FTS5_UNSUPPORTED",
                status_code=500,
            ) from exc

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a connection context manager, reusing memory connection if :memory:."""
        if self._is_memory:
            if self._memory_con is None:
                self._memory_con = sqlite3.connect(":memory:", check_same_thread=False)
                self._memory_con.row_factory = sqlite3.Row
            yield self._memory_con
        else:
            con = sqlite3.connect(self.db_path, timeout=30.0)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            try:
                yield con
            finally:
                con.close()

    def _init_schema(self) -> None:
        """Create metadata table and FTS5 virtual table if they do not exist."""
        with self._get_connection() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS bm25_metadata (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    chunk_index INTEGER DEFAULT 0,
                    content TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    document_type TEXT DEFAULT 'policy',
                    version INTEGER DEFAULT 1,
                    school_id TEXT,
                    group_id TEXT,
                    branch_id TEXT,
                    state TEXT,
                    is_global INTEGER DEFAULT 0,
                    staff_category TEXT,
                    employment_type TEXT,
                    department TEXT,
                    heading TEXT,
                    heading_path TEXT,
                    page_start INTEGER DEFAULT 1,
                    page_end INTEGER DEFAULT 1,
                    effective_from TEXT,
                    effective_to TEXT,
                    created_at TEXT
                );
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bm25_doc_gen ON bm25_metadata(document_id, generation);
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bm25_doc ON bm25_metadata(document_id);
                """
            )
            con.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS bm25_fts USING fts5(
                    chunk_id UNINDEXED,
                    heading,
                    content,
                    tokenize='porter unicode61'
                );
                """
            )
            con.commit()

    @staticmethod
    def _build_fts5_query(query: str) -> str:
        """Sanitize and format natural language query into an FTS5 MATCH expression.

        Extracts words and quoted phrases, escaping syntax characters to prevent FTS5 errors.
        """
        if not query or not query.strip():
            return ""

        # Extract quoted phrases
        phrases = re.findall(r'"([^"]+)"', query)
        clean = re.sub(r'"[^"]+"', " ", query)

        # Extract alphanumeric words
        words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", clean)
        if not words and not phrases:
            words = re.findall(r"\b[a-zA-Z0-9]\b", clean)

        parts: List[str] = []
        for p in phrases:
            clean_p = " ".join(re.findall(r"\b[a-zA-Z0-9_-]+\b", p))
            if clean_p:
                parts.append(f'"{clean_p}"')

        for w in words:
            clean_w = w.replace('"', "").strip()
            if clean_w and clean_w.upper() not in ("AND", "OR", "NOT"):
                parts.append(f'"{clean_w}"')

        if not parts:
            return ""

        # Join with OR so that any matching keyword produces a candidate hit
        return " OR ".join(parts)

    def index_chunks(self, chunks: List[BM25ChunkRecord]) -> None:
        """Idempotently index chunks into both metadata and FTS5 tables."""
        if not chunks:
            return

        with self._get_connection() as con:
            cur = con.cursor()
            for chunk in chunks:
                # 1. Delete prior entries for chunk_id if re-indexing
                cur.execute("DELETE FROM bm25_fts WHERE chunk_id = ?", (chunk.chunk_id,))
                cur.execute("DELETE FROM bm25_metadata WHERE chunk_id = ?", (chunk.chunk_id,))

                # 2. Insert into metadata
                heading_path_str = json.dumps(chunk.heading_path or [])
                now_iso = current_utc_time().isoformat()
                cur.execute(
                    """
                    INSERT INTO bm25_metadata (
                        chunk_id, document_id, generation, chunk_index, content,
                        document_name, document_type, version, school_id, group_id,
                        branch_id, state, is_global, staff_category, employment_type,
                        department, heading, heading_path, page_start, page_end,
                        effective_from, effective_to, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.generation,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.document_name,
                        chunk.document_type,
                        chunk.version,
                        chunk.school_id,
                        chunk.group_id,
                        chunk.branch_id,
                        chunk.state,
                        1 if chunk.is_global else 0,
                        chunk.staff_category,
                        chunk.employment_type,
                        chunk.department,
                        chunk.heading,
                        heading_path_str,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.effective_from,
                        chunk.effective_to,
                        now_iso,
                    ),
                )

                # 3. Insert into FTS5 index
                cur.execute(
                    """
                    INSERT INTO bm25_fts (chunk_id, heading, content)
                    VALUES (?, ?, ?)
                    """,
                    (chunk.chunk_id, chunk.heading or "", chunk.content),
                )
            con.commit()
            logger.info("Indexed %d chunks into SQLite BM25 index", len(chunks))

    @staticmethod
    def _build_tenant_scope_clause(filters: Dict[str, Any]) -> Tuple[Optional[str], List[Any]]:
        """Construct SQL WHERE clause for tenant hierarchy scoping aligned with PolicyScopeFilter.

        Enforces non-bypass tenant isolation in the SQLite pre-filter:
        - If r_school is provided, chunks belonging to a different school are excluded:
            (m.school_id = ? OR m.school_id IS NULL OR m.school_id = '')
        - If r_group is provided, chunks belonging to a different group are excluded:
            (m.group_id = ? OR m.group_id IS NULL OR m.group_id = '')
        - At least one positive scope dimension or global scope must match:
            (m.school_id = ? OR m.group_id = ? OR m.branch_id = ? OR LOWER(m.state) = ? OR m.is_global = 1)
        """
        clauses: List[str] = []
        params: List[Any] = []

        r_school = (filters.get("school_id") or "").strip()
        r_group = (filters.get("group_id") or "").strip()
        r_branch = (filters.get("branch_id") or "").strip()
        r_state = (filters.get("state") or "").strip().lower()
        r_global = bool(filters.get("is_global", False))

        if not r_school and not r_group and not r_branch and not r_state and not r_global:
            return None, []

        # 1. School isolation: chunk must not specify a different school
        if r_school:
            clauses.append("(m.school_id = ? OR m.school_id IS NULL OR m.school_id = '')")
            params.append(r_school)

        # 2. Group isolation: chunk must not specify a different group
        if r_group:
            clauses.append("(m.group_id = ? OR m.group_id IS NULL OR m.group_id = '')")
            params.append(r_group)

        # 3. Positive match dimension: must match school, group, branch, state, or be global
        positive_clauses: List[str] = ["m.is_global = 1"]
        positive_params: List[Any] = []

        if r_school:
            positive_clauses.append("m.school_id = ?")
            positive_params.append(r_school)
        if r_group:
            positive_clauses.append("m.group_id = ?")
            positive_params.append(r_group)
        if r_branch:
            positive_clauses.append("m.branch_id = ?")
            positive_params.append(r_branch)
        if r_state:
            positive_clauses.append("LOWER(m.state) = ?")
            positive_params.append(r_state)

        clauses.append(f"({' OR '.join(positive_clauses)})")
        params.extend(positive_params)

        return " AND ".join(clauses), params

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BM25SearchResult]:
        """Perform BM25 keyword search using FTS5 match and rank."""
        fts_query = self._build_fts5_query(query)
        if not fts_query:
            logger.debug("BM25 query '%s' yielded no searchable tokens", query)
            return []

        limit = max(1, top_k)
        where_clauses = ["bm25_fts MATCH ?"]
        params: List[Any] = [fts_query]

        if filters:
            if "document_id" in filters:
                where_clauses.append("m.document_id = ?")
                params.append(filters["document_id"])
            if "generation" in filters:
                where_clauses.append("m.generation = ?")
                params.append(int(filters["generation"]))
            scope_clause, scope_params = self._build_tenant_scope_clause(filters)
            if scope_clause:
                where_clauses.append(scope_clause)
                params.extend(scope_params)

        where_sql = " AND ".join(where_clauses)
        sql = f"""
            SELECT
                m.chunk_id,
                m.document_id,
                m.generation,
                m.content,
                m.document_name,
                m.document_type,
                m.version,
                m.school_id,
                m.group_id,
                m.branch_id,
                m.state,
                m.is_global,
                m.staff_category,
                m.employment_type,
                m.department,
                m.heading,
                m.heading_path,
                m.page_start,
                m.page_end,
                m.effective_from,
                m.effective_to,
                bm25(bm25_fts, 3.0, 1.0) AS score
            FROM bm25_fts f
            JOIN bm25_metadata m ON f.chunk_id = m.chunk_id
            WHERE {where_sql}
            ORDER BY score ASC
            LIMIT ?
        """
        params.append(limit)

        results: List[BM25SearchResult] = []
        try:
            with self._get_connection() as con:
                rows = con.execute(sql, params).fetchall()
                for rank_idx, row in enumerate(rows, start=1):
                    # FTS5 bm25 returns negative values where lower/more negative is better match
                    raw_score = float(row["score"])
                    positive_score = round(abs(raw_score), 6)

                    heading_path: List[str] = []
                    raw_hp = row["heading_path"]
                    if raw_hp:
                        try:
                            heading_path = json.loads(raw_hp)
                        except Exception:
                            heading_path = []

                    res = BM25SearchResult(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        generation=int(row["generation"]),
                        content=row["content"],
                        document_name=row["document_name"],
                        document_type=row["document_type"] or "policy",
                        version=int(row["version"] or 1),
                        bm25_score=positive_score,
                        bm25_rank=rank_idx,
                        school_id=row["school_id"],
                        group_id=row["group_id"],
                        branch_id=row["branch_id"],
                        state=row["state"],
                        is_global=bool(row["is_global"]),
                        staff_category=row["staff_category"],
                        employment_type=row["employment_type"],
                        department=row["department"],
                        heading=row["heading"],
                        heading_path=heading_path,
                        page_start=int(row["page_start"] or 1),
                        page_end=int(row["page_end"] or 1),
                        effective_from=row["effective_from"],
                        effective_to=row["effective_to"],
                    )
                    results.append(res)
        except Exception as exc:
            logger.error("Error executing SQLite BM25 search for query '%s': %s", query, exc)
            return []

        return results

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks belonging to a document."""
        with self._get_connection() as con:
            cur = con.cursor()
            cur.execute(
                "DELETE FROM bm25_fts WHERE chunk_id IN (SELECT chunk_id FROM bm25_metadata WHERE document_id = ?)",
                (document_id,),
            )
            cur.execute("DELETE FROM bm25_metadata WHERE document_id = ?", (document_id,))
            deleted = cur.rowcount
            con.commit()
            logger.info("Deleted document %s from BM25 index (%d chunks removed)", document_id, deleted)
            return deleted

    def delete_generation(self, document_id: str, generation: int) -> int:
        """Remove all chunks for a specific document and generation (e.g. aborting failed run)."""
        with self._get_connection() as con:
            cur = con.cursor()
            cur.execute(
                """
                DELETE FROM bm25_fts WHERE chunk_id IN (
                    SELECT chunk_id FROM bm25_metadata WHERE document_id = ? AND generation = ?
                )
                """,
                (document_id, generation),
            )
            cur.execute(
                "DELETE FROM bm25_metadata WHERE document_id = ? AND generation = ?",
                (document_id, generation),
            )
            deleted = cur.rowcount
            con.commit()
            logger.info("Purged %d unactivated BM25 chunks for doc=%s gen=%d", deleted, document_id, generation)
            return deleted

    def delete_previous_generations(self, document_id: str, active_generation: int) -> int:
        """Purge chunks from older generations than active_generation for a document."""
        with self._get_connection() as con:
            cur = con.cursor()
            cur.execute(
                """
                DELETE FROM bm25_fts WHERE chunk_id IN (
                    SELECT chunk_id FROM bm25_metadata WHERE document_id = ? AND generation != ?
                )
                """,
                (document_id, active_generation),
            )
            cur.execute(
                "DELETE FROM bm25_metadata WHERE document_id = ? AND generation != ?",
                (document_id, active_generation),
            )
            deleted = cur.rowcount
            con.commit()
            if deleted > 0:
                logger.info(
                    "Purged %d older generation BM25 chunks for doc=%s (active gen: %d)",
                    deleted,
                    document_id,
                    active_generation,
                )
            return deleted

    def count_generation_chunks(self, document_id: str, generation: int) -> int:
        """Return the number of chunks stored in BM25 for a document and generation."""
        with self._get_connection() as con:
            row = con.execute(
                "SELECT COUNT(*) AS total FROM bm25_metadata WHERE document_id = ? AND generation = ?",
                (document_id, generation),
            ).fetchone()
            return int(row["total"]) if row else 0

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Return total number of chunks currently in the metadata index."""
        sql = "SELECT COUNT(*) AS total FROM bm25_metadata m"
        params: List[Any] = []
        if filters:
            clauses = []
            if "document_id" in filters:
                clauses.append("m.document_id = ?")
                params.append(filters["document_id"])
            if "generation" in filters:
                clauses.append("m.generation = ?")
                params.append(filters["generation"])
            scope_clause, scope_params = self._build_tenant_scope_clause(filters)
            if scope_clause:
                clauses.append(scope_clause)
                params.extend(scope_params)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)

        with self._get_connection() as con:
            row = con.execute(sql, params).fetchone()
            return int(row["total"]) if row else 0

    def health_check(self) -> Dict[str, Any]:
        """Perform diagnostic self-check on the SQLite FTS5 index."""
        try:
            with self._get_connection() as con:
                total_chunks = con.execute("SELECT COUNT(*) AS c FROM bm25_metadata").fetchone()["c"]
                total_docs = con.execute("SELECT COUNT(DISTINCT document_id) AS d FROM bm25_metadata").fetchone()["d"]
                fts_count = con.execute("SELECT COUNT(*) AS f FROM bm25_fts").fetchone()["f"]

            db_size_bytes = 0
            if not self._is_memory and Path(self.db_path).exists():
                db_size_bytes = Path(self.db_path).stat().st_size

            return {
                "status": "healthy",
                "provider": "sqlite_fts5",
                "db_path": str(self.db_path),
                "total_documents": int(total_docs),
                "total_chunks": int(total_chunks),
                "fts_indexed_rows": int(fts_count),
                "db_size_bytes": db_size_bytes,
                "fts5_available": True,
            }
        except Exception as exc:
            logger.error("BM25 health check failed: %s", exc)
            return {
                "status": "unhealthy",
                "provider": "sqlite_fts5",
                "error": str(exc),
                "fts5_available": False,
            }

    def rebuild_index(self, chunks: List[BM25ChunkRecord]) -> None:
        """Completely drop and recreate the index with the provided chunk records."""
        with self._get_connection() as con:
            con.execute("DROP TABLE IF EXISTS bm25_fts;")
            con.execute("DROP TABLE IF EXISTS bm25_metadata;")
            con.commit()

        self._init_schema()
        self.index_chunks(chunks)
        logger.info("Rebuilt BM25 index with %d chunks", len(chunks))

    def close(self) -> None:
        """Close any cached memory connection."""
        if self._memory_con is not None:
            try:
                self._memory_con.close()
            except Exception:
                pass
            self._memory_con = None
