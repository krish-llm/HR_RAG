"""Common API dependencies for route handlers.

IMPORTANT ARCHITECTURE RULE:
Tenant and school context handling is strictly separated from authentication.
Context parameters extracted from request headers or query parameters represent
requested scoping filters (e.g. for search, policy routing, or document scoping).
They MUST NOT be treated as an independent source of truth or authorization credentials.
Authorization and permission validation must always be verified against authenticated
user credentials / tokens in their respective authentication middleware/guards.
"""

from typing import Optional
from fastapi import Depends, Header
from pydantic import BaseModel, ConfigDict, Field
from app.config import Settings, get_settings
from app.db.chroma import ChromaDBService, get_chroma_service
from app.db.mongo import AsyncDatabase, AsyncMongoClient, get_mongo_client, get_mongo_db
from app.modules.bm25.base import BaseBM25IndexProvider

_bm25_provider_instance: Optional[BaseBM25IndexProvider] = None


def get_bm25_provider(settings: Settings = Depends(get_settings)) -> BaseBM25IndexProvider:
    """Dependency provider for the singleton BM25 index provider."""
    global _bm25_provider_instance
    if _bm25_provider_instance is None:
        from app.modules.bm25.sqlite_provider import SQLiteFTSBM25IndexProvider

        _bm25_provider_instance = SQLiteFTSBM25IndexProvider(db_path=settings.BM25_INDEX_PATH)
    return _bm25_provider_instance


def reset_bm25_provider() -> None:
    """Close and reset the BM25 index provider instance (used during shutdown and testing)."""
    global _bm25_provider_instance
    if _bm25_provider_instance is not None:
        _bm25_provider_instance.close()
        _bm25_provider_instance = None


class SchoolContext(BaseModel):
    """Encapsulates requested school scope parameters for request-level routing.

    NOTE: This is a scoping filter hint, NOT an authentication token or security boundary.
    It does not grant access permissions on its own.
    """

    model_config = ConfigDict(populate_by_name=True)

    school_id: Optional[str] = Field(default=None, description="Requested school ID filter")
    group_id: Optional[str] = Field(default=None, description="Requested group ID filter")
    branch_id: Optional[str] = Field(default=None, description="Requested branch ID filter")
    state: Optional[str] = Field(default=None, description="Requested state/geographical filter")
    is_global: bool = Field(default=False, description="Flag requesting global policy scope")


async def get_school_context(
    x_school_id: Optional[str] = Header(default=None, alias="X-School-ID"),
    x_group_id: Optional[str] = Header(default=None, alias="X-Group-ID"),
    x_branch_id: Optional[str] = Header(default=None, alias="X-Branch-ID"),
    x_state: Optional[str] = Header(default=None, alias="X-State"),
    x_is_global: bool = Header(default=False, alias="X-Is-Global"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID", deprecated=True),
) -> SchoolContext:
    """Extract requested school scoping parameters from headers.

    Fallback: If legacy X-Tenant-ID is provided and X-School-ID is omitted,
    X-Tenant-ID is mapped to school_id for backward compatibility.
    """
    from fastapi.params import Header as HeaderParam

    clean_school = None if isinstance(x_school_id, HeaderParam) else x_school_id
    clean_group = None if isinstance(x_group_id, HeaderParam) else x_group_id
    clean_branch = None if isinstance(x_branch_id, HeaderParam) else x_branch_id
    clean_state = None if isinstance(x_state, HeaderParam) else x_state
    clean_global = False if isinstance(x_is_global, HeaderParam) else bool(x_is_global)
    clean_legacy = None if isinstance(x_tenant_id, HeaderParam) else x_tenant_id

    return SchoolContext(
        school_id=clean_school or clean_legacy,
        group_id=clean_group,
        branch_id=clean_branch,
        state=clean_state,
        is_global=clean_global,
    )


async def get_tenant_id_header(
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID", deprecated=True),
) -> Optional[str]:
    """Legacy helper: Extract optional X-Tenant-ID header.

    WARNING: Deprecated in favor of get_school_context. This header is a client hint only
    and MUST NOT be used as an independent source of truth for authorization.
    """
    return x_tenant_id


__all__ = [
    "get_settings",
    "get_mongo_db",
    "get_mongo_client",
    "get_chroma_service",
    "get_bm25_provider",
    "reset_bm25_provider",
    "get_school_context",
    "get_tenant_id_header",
    "SchoolContext",
    "Settings",
    "AsyncDatabase",
    "AsyncMongoClient",
    "ChromaDBService",
    "BaseBM25IndexProvider",
]
