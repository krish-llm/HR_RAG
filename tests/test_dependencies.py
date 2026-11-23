"""Tests for common API dependencies including school context scoping."""

import pytest
from app.api.dependencies import SchoolContext, get_school_context, get_tenant_id_header


@pytest.mark.asyncio
async def test_get_school_context_headers():
    """Verify get_school_context extracts canonical headers."""
    ctx = await get_school_context(
        x_school_id="SCH-123",
        x_group_id="GRP-999",
        x_branch_id="BR-01",
        x_state="Maharashtra",
        x_is_global=False,
    )
    assert ctx.school_id == "SCH-123"
    assert ctx.group_id == "GRP-999"
    assert ctx.branch_id == "BR-01"
    assert ctx.state == "Maharashtra"
    assert ctx.is_global is False


@pytest.mark.asyncio
async def test_get_school_context_fallback_legacy_header():
    """Verify legacy X-Tenant-ID falls back to school_id when X-School-ID is absent."""
    ctx = await get_school_context(
        x_school_id=None,
        x_group_id="GRP-999",
        x_tenant_id="SCH-FALLBACK",
    )
    assert ctx.school_id == "SCH-FALLBACK"
    assert ctx.group_id == "GRP-999"


@pytest.mark.asyncio
async def test_legacy_tenant_id_header():
    """Verify legacy get_tenant_id_header functions as a hint."""
    tenant = await get_tenant_id_header(x_tenant_id="legacy-tenant-1")
    assert tenant == "legacy-tenant-1"
