"""Structured logging and execution timing utilities for the HR Policy RAG platform."""

from contextlib import contextmanager
import contextvars
import logging
import time
from typing import Any, Dict, Optional
import uuid

# Context variable for propagating request_id across async tasks and function calls
_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Return the current request_id from context, or '-' if unset."""
    return _request_id_ctx.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set the request_id context variable.

    If None or empty, generates an 8-character hex UUID.
    """
    req_id = request_id.strip() if request_id and request_id.strip() else uuid.uuid4().hex[:8]
    _request_id_ctx.set(req_id)
    return req_id


def clear_request_id() -> None:
    """Reset the request_id context variable to default."""
    _request_id_ctx.set("-")


def format_log_msg(
    module: str,
    status: Optional[str] = None,
    duration_ms: Optional[float] = None,
    request_id: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Format a consistent structured log message.

    Format:
    request_id=<id> | module=<module> | status=<status> [| duration_ms=<ms>] [| key=value ...]
    """
    resolved_status = kwargs.pop("status", None) or status or "completed"
    req_id = request_id or get_request_id()
    parts = [
        f"request_id={req_id}",
        f"module={module}",
        f"status={resolved_status}",
    ]
    if duration_ms is not None:
        parts.append(f"duration_ms={int(round(duration_ms))}")

    for k, v in kwargs.items():
        if v is not None:
            # Clean string values to prevent delimiter collision
            val_str = str(v).replace("\n", " ").strip()
            parts.append(f"{k}={val_str}")

    return " | ".join(parts)


def log_operation_start(
    logger: logging.Logger,
    module: str,
    request_id: Optional[str] = None,
    **kwargs: Any,
) -> float:
    """Log the start of an operation and return the start timestamp."""
    start_time = time.perf_counter()
    logger.info(format_log_msg(module=module, status="started", request_id=request_id, **kwargs))
    return start_time


def log_operation_completed(
    logger: logging.Logger,
    module: str,
    start_time: float,
    request_id: Optional[str] = None,
    **kwargs: Any,
) -> float:
    """Log the completion of an operation with elapsed duration in milliseconds."""
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        format_log_msg(
            module=module,
            status="completed",
            duration_ms=duration_ms,
            request_id=request_id,
            **kwargs,
        )
    )
    return duration_ms


def log_operation_failed(
    logger: logging.Logger,
    module: str,
    start_time: float,
    error: Any,
    is_warning: bool = False,
    request_id: Optional[str] = None,
    **kwargs: Any,
) -> float:
    """Log an operation failure with elapsed duration in milliseconds."""
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    msg = format_log_msg(
        module=module,
        status="failed",
        duration_ms=duration_ms,
        error=str(error),
        request_id=request_id,
        **kwargs,
    )
    if is_warning:
        logger.warning(msg)
    else:
        logger.error(msg)
    return duration_ms


class TimedBlock:
    """Context manager for timing and structured logging of synchronous and asynchronous blocks."""

    def __init__(
        self,
        logger: logging.Logger,
        module: str,
        log_start: bool = True,
        start_kwargs: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        self.logger = logger
        self.module = module
        self.log_start = log_start
        self.start_kwargs = start_kwargs or {}
        self.request_id = request_id
        self.start_time: float = 0.0
        self.duration_ms: float = 0.0
        self.extra_completion_kwargs: Dict[str, Any] = {}
        self._suppress_completed_log = False

    def set_extra(self, **kwargs: Any) -> None:
        """Add additional metadata fields to the completion log message."""
        self.extra_completion_kwargs.update(kwargs)

    def _enter(self) -> "TimedBlock":
        self.start_time = time.perf_counter()
        if self.log_start:
            self.logger.info(
                format_log_msg(
                    module=self.module,
                    status="started",
                    request_id=self.request_id,
                    **self.start_kwargs,
                )
            )
        return self

    def _exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        extra_kwargs = dict(self.extra_completion_kwargs)
        if exc_type is not None:
            status_val = extra_kwargs.pop("status", "failed")
            self.logger.error(
                format_log_msg(
                    module=self.module,
                    status=status_val,
                    duration_ms=self.duration_ms,
                    error=str(exc_val),
                    request_id=self.request_id,
                    **extra_kwargs,
                )
            )
        elif not self._suppress_completed_log:
            status_val = extra_kwargs.pop("status", "completed")
            self.logger.info(
                format_log_msg(
                    module=self.module,
                    status=status_val,
                    duration_ms=self.duration_ms,
                    request_id=self.request_id,
                    **extra_kwargs,
                )
            )

    def __enter__(self) -> "TimedBlock":
        return self._enter()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._exit(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> "TimedBlock":
        return self._enter()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._exit(exc_type, exc_val, exc_tb)
