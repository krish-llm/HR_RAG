"""General utility functions and helpers."""

from datetime import datetime, timezone
from typing import Any, Dict


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of dictionary removing None values."""
    return {k: v for k, v in data.items() if v is not None}
