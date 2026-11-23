"""Query Understanding module for deterministic normalization, intent classification, and scoping."""

from datetime import datetime
import logging
import re
from typing import Any, Dict, List, Optional, Set

from app.api.dependencies import SchoolContext
from app.core.exceptions import AppException
from app.modules.retrieval.schemas import EmployeeContext, QueryUnderstandingResult
from app.schemas.common import current_utc_time

logger = logging.getLogger(__name__)

# Common English stop words to filter out during deterministic keyword extraction
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "won't", "would", "wouldn't", "you", "your",
    "yours", "yourself", "yourselves", "available", "applicable", "rule", "rules",
    "policy", "policies", "tell", "explain", "give", "know", "much", "many",
}

# HR Intent classification keyword dictionary (ordered by specificity)
INTENT_PATTERNS: Dict[str, List[str]] = {
    "leave": [
        "casual leave", "sick leave", "earned leave", "annual leave", "maternity leave",
        "paternity leave", "bereavement leave", "study leave", "compensatory off",
        "comp off", "leave policy", "leaves", "leave", "holiday", "vacation",
        "time off", "half day", "cl", "sl", "el", "absence", "privilege leave",
        "duty leave", "unpaid leave", "loss of pay", "lop",
    ],
    "compensation": [
        "salary", "payroll", "wages", "stipend", "allowance", "increment", "bonus",
        "deduction", "provident fund", "pf", "gratuity", "esi", "reimbursement",
        "pay scale", "hra", "da", "travel allowance", "per diem", "compensation",
        "overtime pay", "appraisal", "pay slip",
    ],
    "benefits": [
        "insurance", "health insurance", "medical", "mediclaim", "wellness",
        "pension", "transport", "bus", "fee concession", "child concession",
        "tuition fee", "subsidy", "meal", "gratuity", "fringe benefits",
    ],
    "attendance": [
        "working hours", "biometric", "punctuality", "late arrival", "late coming",
        "shift", "overtime", "working days", "work from home", "wfh", "remote work",
        "login", "logout", "half day", "timings", "grace period",
    ],
    "conduct": [
        "code of conduct", "ethics", "discipline", "disciplinary", "harassment",
        "posh", "sexual harassment", "grievance", "misconduct", "dress code",
        "confidentiality", "conflict of interest", "social media", "warning",
        "penalty", "investigation", "inquiry",
    ],
    "probation": [
        "probation", "confirmation", "probationary", "probation period",
        "probation extension", "probationer",
    ],
    "termination": [
        "resignation", "notice period", "exit", "termination", "relieving",
        "severance", "retirement", "clearance", "relieving letter", "handover",
        "dismissal", "separation",
    ],
    "recruitment": [
        "recruitment", "hiring", "interview", "eligibility", "qualification",
        "onboarding", "induction", "background verification", "bgv",
        "appointment letter", "job description",
    ],
}

# Explicit employee context patterns detectable in natural language
STAFF_CATEGORY_PATTERNS = [
    ("non-teaching", "non_teaching"),
    ("non teaching", "non_teaching"),
    ("administrative staff", "non_teaching"),
    ("support staff", "non_teaching"),
    ("teaching staff", "teaching"),
    ("teaching", "teaching"),
    ("teachers", "teaching"),
    ("teacher", "teaching"),
    ("faculty", "teaching"),
    ("academic staff", "teaching"),
]

EMPLOYMENT_TYPE_PATTERNS = [
    ("full-time", "full_time"),
    ("full time", "full_time"),
    ("part-time", "part_time"),
    ("part time", "part_time"),
    ("permanent", "permanent"),
    ("regular", "permanent"),
    ("contractual", "contract"),
    ("contract", "contract"),
    ("temporary", "contract"),
    ("visiting", "contract"),
    ("ad-hoc", "contract"),
    ("probationary", "probationary"),
    ("on probation", "probationary"),
]


class QueryUnderstandingService:
    """Service providing deterministic query normalization, intent detection, and context mapping."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Strip and collapse excessive whitespace."""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def extract_intent(self, text_lower: str) -> str:
        """Classify HR intent deterministically based on keyword and phrase matching."""
        for intent, patterns in INTENT_PATTERNS.items():
            for pat in patterns:
                # Word boundary match for short acronyms like cl, sl, el, pf, da
                if len(pat) <= 3:
                    if re.search(rf"\b{re.escape(pat)}\b", text_lower):
                        return intent
                elif pat in text_lower:
                    return intent
        return "general_inquiry"

    def extract_keywords(self, text: str) -> List[str]:
        """Extract deterministic list of domain keywords from normalized text, filtering stop words."""
        # Clean out punctuation except hyphens inside words
        cleaned = re.sub(r"[^\w\s-]", " ", text)
        tokens = cleaned.split()
        keywords: List[str] = []
        seen: Set[str] = set()

        for token in tokens:
            t_lower = token.lower()
            if len(t_lower) > 1 and t_lower not in STOP_WORDS and not t_lower.isdigit():
                if t_lower not in seen:
                    seen.add(t_lower)
                    keywords.append(t_lower)

        return keywords

    def extract_employee_context(
        self,
        text_lower: str,
        supplied_context: Optional[EmployeeContext] = None,
    ) -> Optional[EmployeeContext]:
        """Extract explicit employee applicability attributes without inventing details."""
        extracted_staff: Optional[str] = None
        extracted_emp_type: Optional[str] = None

        for pattern, category in STAFF_CATEGORY_PATTERNS:
            if pattern in text_lower:
                extracted_staff = category
                break

        for pattern, emp_type in EMPLOYMENT_TYPE_PATTERNS:
            if pattern in text_lower:
                extracted_emp_type = emp_type
                break

        final_staff = (
            supplied_context.staff_category
            if supplied_context and supplied_context.staff_category
            else extracted_staff
        )
        final_emp_type = (
            supplied_context.employment_type
            if supplied_context and supplied_context.employment_type
            else extracted_emp_type
        )
        final_dept = (
            supplied_context.department
            if supplied_context and supplied_context.department
            else None
        )

        if not final_staff and not final_emp_type and not final_dept:
            return None

        return EmployeeContext(
            staff_category=final_staff,
            employment_type=final_emp_type,
            department=final_dept,
        )

    def understand_query(
        self,
        query: str,
        school_context: SchoolContext,
        employee_context: Optional[EmployeeContext] = None,
        as_of_date: Optional[datetime] = None,
    ) -> QueryUnderstandingResult:
        """Execute deterministic query understanding pipeline."""
        normalized = self.normalize_text(query)
        if not normalized:
            raise AppException(
                message="User query cannot be empty or blank",
                error_code="QUERY_EMPTY",
                status_code=400,
            )

        text_lower = normalized.lower()
        intent = self.extract_intent(text_lower)
        keywords = self.extract_keywords(normalized)
        resolved_employee_context = self.extract_employee_context(text_lower, employee_context)

        # Scoping representation from SchoolContext
        requested_scope: Dict[str, Any] = {
            "school_id": school_context.school_id,
            "group_id": school_context.group_id,
            "branch_id": school_context.branch_id,
            "state": school_context.state,
            "is_global": school_context.is_global,
        }

        eval_date = as_of_date or current_utc_time()

        return QueryUnderstandingResult(
            original_query=query,
            normalized_query=normalized,
            intent=intent,
            extracted_keywords=keywords,
            employee_context=resolved_employee_context,
            requested_scope=requested_scope,
            as_of_date=eval_date,
        )
