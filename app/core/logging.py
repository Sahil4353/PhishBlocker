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
            # If something else set it, keep theirs
            if getattr(record, "request_id", None) in (None, ""):
                setattr(record, "request_id", rid)
        return True


def setup_logging(level: Optional[str] = None) -> None:
    """
    Initialize root logging once, idempotently.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    eff_level = (level or settings.LOG_LEVEL).upper()
    root.setLevel(getattr(logging, eff_level, logging.INFO))

    disable_flt = _DisableWhenOff()
    rid_flt = _RequestIdInjector()

    for h in root.handlers:
        # Attach filters once
        if not any(isinstance(f, _DisableWhenOff) for f in h.filters):
            h.addFilter(disable_flt)
        if not any(isinstance(f, _RequestIdInjector) for f in h.filters):
            h.addFilter(rid_flt)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
