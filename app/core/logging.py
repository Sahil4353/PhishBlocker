# app/core/logging.py
from __future__ import annotations

import logging
from contextvars import ContextVar
from logging import Filter, LogRecord
from typing import Optional

from app.core.config import settings

# Include request id in every line (rid=- when not in a request context)
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [rid=%(request_id)s]: %(message)s"

# Context var to carry request id across the request lifecycle
_request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def set_request_id(rid: Optional[str]) -> None:
    _request_id_ctx.set(rid)


def get_request_id() -> Optional[str]:
    return _request_id_ctx.get()


def clear_request_id() -> None:
    _request_id_ctx.set(None)


class _DisableWhenOff(Filter):
    """Drop all records when logging is disabled via settings."""

    def filter(self, record: LogRecord) -> bool:
        return settings.ENABLE_LOGGING


class _RequestIdInjector(Filter):
    """Ensure every record has a request_id attribute to satisfy the format string."""

    def filter(self, record: LogRecord) -> bool:
        rid = get_request_id() or "-"
        if not hasattr(record, "request_id"):
            setattr(record, "request_id", rid)
        else:
            if getattr(record, "request_id", None) in (None, ""):
                setattr(record, "request_id", rid)
        return True


def _ensure_root_handler() -> logging.Handler:
    """
    Ensure the root logger has at least one handler configured with our formatter.
    Returns that handler (first handler on root).
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    # If the first handler has no formatter, set one.
    if not root.handlers[0].formatter:
        root.handlers[0].setFormatter(logging.Formatter(LOG_FORMAT))
    return root.handlers[0]


def setup_logging(level: Optional[str] = None) -> None:
    """
    Initialize root logging once, idempotently.
    """
    h = _ensure_root_handler()

    eff_level = (level or settings.LOG_LEVEL).upper()
    logging.getLogger().setLevel(getattr(logging, eff_level, logging.INFO))

    # Attach our filters to all root handlers (once)
    disable_flt = _DisableWhenOff()
    rid_flt = _RequestIdInjector()

    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(f, _DisableWhenOff) for f in handler.filters):
            handler.addFilter(disable_flt)
        if not any(isinstance(f, _RequestIdInjector) for f in handler.filters):
            handler.addFilter(rid_flt)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def attach_filters_to(logger: logging.Logger) -> None:
    """
    Attach the same filters the root logger uses to a specific logger (e.g., uvicorn.*).
    If the target logger has no handlers, add a StreamHandler with the root formatter.
    Idempotent: won’t duplicate filters.
    """
    setup_logging()

    root = logging.getLogger()
    root_handler = _ensure_root_handler()

    # Try to reuse existing filter instances from the root handler to avoid duplication.
    existing_disable = (
        next((f for f in root_handler.filters if isinstance(f, _DisableWhenOff)), None)
        or _DisableWhenOff()
    )
    existing_rid = (
        next(
            (f for f in root_handler.filters if isinstance(f, _RequestIdInjector)), None
        )
        or _RequestIdInjector()
    )

    # If the target logger has no handlers, attach one with the same formatter.
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(root_handler.formatter or logging.Formatter(LOG_FORMAT))
        logger.addHandler(h)

    # Attach filters to each handler (once)
    for h in logger.handlers:
        if not any(isinstance(f, _DisableWhenOff) for f in h.filters):
            h.addFilter(existing_disable)
        if not any(isinstance(f, _RequestIdInjector) for f in h.filters):
            h.addFilter(existing_rid)
