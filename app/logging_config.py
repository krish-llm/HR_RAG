"""Centralized logging configuration module."""

import logging
import sys
from typing import Optional


def setup_logging(log_level: Optional[str] = "INFO") -> None:
    """Configure centralized application-wide logging.

    Args:
        log_level: Desired log level string (e.g. DEBUG, INFO, WARNING, ERROR)
    """
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)

    # Formatter with timestamp, level, module name, line number, message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
    else:
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)

    # Quiet overly chatty third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance configured for the given component name."""
    return logging.getLogger(name)


# Re-export core structured logging utilities
from app.core.logging_utils import (  # noqa: E402
    TimedBlock,
    clear_request_id,
    format_log_msg,
    get_request_id,
    log_operation_completed,
    log_operation_failed,
    log_operation_start,
    set_request_id,
)

