import logging
from logging import Filter, LogRecord
from typing import Optional

from app.core.config import settings

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _DisableWhenOff(Filter):
    """Drop all records when logging is disabled via settings."""

    def filter(self, record: LogRecord) -> bool:
        return settings.ENABLE_LOGGING


def setup_logging(level: Optional[str] = None) -> None:
    """
    Initialize root logging once.
    Safe to call multiple times (idempotent-ish).
    """
    # If already configured, just ensure our filter exists.
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    # Set level from settings or override param
    eff_level = (level or settings.LOG_LEVEL).upper()
    try:
        root.setLevel(getattr(logging, eff_level, logging.INFO))
    except Exception:
        root.setLevel(logging.INFO)

    # Ensure our filter is attached to all handlers
    flt = _DisableWhenOff()
    for h in root.handlers:
        # avoid adding duplicates
        if not any(isinstance(f, _DisableWhenOff) for f in h.filters):
            h.addFilter(flt)


def get_logger(name: str) -> logging.Logger:
    """
    Get a module-level logger consistently.
    Usage:
        from app.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("hello")
    """
    setup_logging()  # ensure configured
    return logging.getLogger(name)
