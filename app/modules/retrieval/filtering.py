"""Centralized filtering and validation rules for multi-tenant policy retrieval (Phase 6)."""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional, Tuple

from app.api.dependencies import SchoolContext
from app.modules.retrieval.schemas import EmployeeContext

logger = logging.getLogger(__name__)


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure datetime is timezone-aware in UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_date(val: Any) -> Optional[datetime]:
    """Parse date from datetime, ISO string, or None."""
    if not val:
        return None
    if isinstance(val, datetime):
        return _to_utc(val)
    if isinstance(val, str):
        try:
            cleaned = val.replace("Z", "+00:00").strip()
            return _to_utc(datetime.fromisoformat(cleaned))
        except Exception:
            return None
    return None


class PolicyScopeFilter:
    """Encapsulates strict multi-tenancy, employee applicability, and policy validity rules."""

    @staticmethod
    def is_tenant_scope_applicable(
        chunk_meta: Dict[str, Any],
        requested_scope: SchoolContext,
    ) -> bool:
        """Verify that a candidate chunk belongs to the requested educational hierarchy.

        Rules:
        - Branch-specific policies must match the requested school and branch.
        - School-wide policies must match the requested school.
        - Group-wide policies must match the requested group_id.
        - A group-wide policy must never be treated as globally applicable across unrelated groups.
        - State-level policies must match the requested state and must not bypass school/group isolation.
        - Never use is_global=true as a standalone condition that exposes policies across tenants.
        """
        c_school = (chunk_meta.get("school_id") or "").strip()
        c_group = (chunk_meta.get("group_id") or "").strip()
        c_branch = (chunk_meta.get("branch_id") or "").strip()
        c_state = (chunk_meta.get("state") or "").strip().lower()
        c_global = bool(chunk_meta.get("is_global", False))

        r_school = (requested_scope.school_id or "").strip()
        r_group = (requested_scope.group_id or "").strip()
        r_branch = (requested_scope.branch_id or "").strip()
        r_state = (requested_scope.state or "").strip().lower()

        # 1. Branch-specific policy validation
        if c_branch:
            if not r_branch or c_branch != r_branch:
                return False
            if c_school and r_school and c_school != r_school:
                return False
            return True

        # 2. School-wide policy validation
        if c_school:
            if not r_school or c_school != r_school:
                return False
            if c_group and r_group and c_group != r_group:
                return False
            return True

        # 3. Group-wide policy validation (no school or branch defined)
        if c_group:
            if not r_group or c_group != r_group:
                return False
            # If state is also specified on group policy, state must match
            if c_state and r_state and c_state != r_state:
                return False
            return True

        # 4. State-level policy validation
        if c_state:
            if not r_state or c_state != r_state:
                return False
            # Must not bypass school/group isolation rules
            if c_group and r_group and c_group != r_group:
                return False
            if c_school and r_school and c_school != r_school:
                return False
            return True

        # 5. Global policy validation
        if c_global:
            # If policy explicitly defines group or school, tenant isolation must hold
            if c_group and r_group and c_group != r_group:
                return False
            if c_school and r_school and c_school != r_school:
                return False
            if c_state and r_state and c_state != r_state:
                return False
            # Disallow standalone global leakage without matching tenant context
            if not r_school and not r_group and not r_state and not requested_scope.is_global:
                return False
            return True

        # If chunk defines no recognized scope dimension, reject
        return False

    @staticmethod
    def is_employee_applicable(
        chunk_meta: Dict[str, Any],
        employee_context: Optional[EmployeeContext],
    ) -> bool:
        """Validate employee applicability filters (staff_category, employment_type, department)."""
        c_staff = (chunk_meta.get("staff_category") or "").strip().lower()
        c_emp_type = (chunk_meta.get("employment_type") or "").strip().lower()
        c_dept = (chunk_meta.get("department") or "").strip().lower()

        if not employee_context:
            return True

        r_staff = (employee_context.staff_category or "").strip().lower()
        r_emp_type = (employee_context.employment_type or "").strip().lower()
        r_dept = (employee_context.department or "").strip().lower()

        # Staff category filtering
        if r_staff and c_staff and c_staff not in ("all", "general", "any"):
            if c_staff != r_staff:
                return False

        # Employment type filtering
        if r_emp_type and c_emp_type and c_emp_type not in ("all", "general", "any"):
            synonyms = {
                "full_time": {"full_time", "permanent", "regular"},
                "permanent": {"full_time", "permanent", "regular"},
                "contract": {"contract", "contractual", "temporary"},
                "part_time": {"part_time", "part-time"},
            }
            allowed = synonyms.get(r_emp_type, {r_emp_type})
            if c_emp_type not in allowed:
                return False

        # Department filtering
        if r_dept and c_dept and c_dept not in ("all", "general", "any"):
            if c_dept != r_dept:
                return False

        return True

    @staticmethod
    def is_policy_active(
        doc_meta: Dict[str, Any],
        as_of_date: datetime,
    ) -> Tuple[bool, Optional[str]]:
        """Verify that document lifecycle status is active and effective dates cover as_of_date."""
        status = str(doc_meta.get("status") or doc_meta.get("policy_status") or "").lower()
        if status in ("draft", "superseded", "expired", "failed", "archived"):
            return False, f"Policy status '{status}' is inactive"

        as_of_utc = _to_utc(as_of_date)

        # Check effective_from boundary
        eff_from = _parse_date(doc_meta.get("effective_from"))
        if eff_from and as_of_utc and as_of_utc < eff_from:
            return False, f"Policy is not effective until {eff_from.isoformat()}"

        # Check effective_to boundary
        eff_to = _parse_date(doc_meta.get("effective_to"))
        if eff_to and as_of_utc and as_of_utc > eff_to:
            return False, f"Policy expired on {eff_to.isoformat()}"

        return True, None

    @staticmethod
    def is_generation_active(
        doc_id: str,
        candidate_generation: Any,
        tracking_record: Optional[Dict[str, Any]],
    ) -> Tuple[bool, Optional[str]]:
        """Verify that the candidate's vector generation matches the document's active generation."""
        if not tracking_record:
            return False, f"No tracking record found for document '{doc_id}'"

        status = tracking_record.get("indexing_status")
        if status != "indexed":
            return False, f"Document '{doc_id}' indexing status is '{status}', not 'indexed'"

        active_gen = tracking_record.get("active_generation")
        if active_gen is None or int(active_gen) == 0:
            return False, f"Document '{doc_id}' has no active vector generation"

        try:
            cand_gen_int = int(candidate_generation)
        except (ValueError, TypeError):
            return False, f"Candidate generation '{candidate_generation}' is invalid"

        if cand_gen_int != int(active_gen):
            return (
                False,
                f"Candidate generation {cand_gen_int} does not match document active generation {active_gen}",
            )

        return True, None
