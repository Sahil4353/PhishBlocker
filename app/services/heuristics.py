from __future__ import annotations

import re
from typing import List, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)

# --- Patterns (keep simple & fast for now) ---
PHISH_PATTERNS = [
    r"\bverify your (?:account|password)\b",
    r"\burgent action required\b",
    r"\bclick (?:here|the link)\b",
    r"\bconfirm your (?:details|identity)\b",
]
SPAM_PATTERNS = [
    r"\bfree\s+money\b",
    r"\bwinner\b",
    r"\bwork from home\b",
    r"\bweight loss\b",
]

# Try to compile regexes once at import time. If anything fails, we degrade gracefully.
try:
    PHISH_REGEX = [re.compile(p, re.I) for p in PHISH_PATTERNS]
    SPAM_REGEX = [re.compile(p, re.I) for p in SPAM_PATTERNS]
except re.error as e:
    logger.exception("Regex compilation failed; disabling heuristics: %s", e)
    PHISH_REGEX, SPAM_REGEX = [], []


def _excel_reason(patterns: List[str]) -> List[str]:
    """Map raw regex patterns to human-readable reasons (kept simple)."""
    return [f"matched: {p}" for p in patterns]


def classify_text(text: str) -> Tuple[str, float, List[str]]:
    """
    Heuristic classifier with defensive guards.

    Returns:
        (label, confidence, reasons[])
    Notes:
        - Prefers 'phishing' over 'spam' if both match.
        - Confidence scales with number of hits.
        - Never raises: on any error, returns ('safe', 0.50, [...]) and logs.
    """
    try:
        if not PHISH_REGEX and not SPAM_REGEX:
            logger.warning("Heuristics disabled (regex not compiled)")
            return "safe", 0.50, ["heuristics unavailable"]

        # Normalize defensively; cap extremely long bodies to avoid pathological regex time.
        t = (text or "").strip()
        if len(t) > 200_000:
            logger.info("Input truncated from %d to 200000 chars for safety", len(t))
            t = t[:200_000]

        hits_phish = [r.pattern for r in PHISH_REGEX if r.search(t)]
        hits_spam = [r.pattern for r in SPAM_REGEX if r.search(t)]

        if hits_phish:
            conf = min(0.70 + 0.10 * len(hits_phish), 0.99)
            reasons = _excel_reason(hits_phish)
            logger.debug(
                "Heuristic result: phishing conf=%.2f hits=%s", conf, hits_phish
            )
            return "phishing", conf, reasons

        if hits_spam:
            conf = min(0.60 + 0.10 * len(hits_spam), 0.95)
            reasons = _excel_reason(hits_spam)
            logger.debug("Heuristic result: spam conf=%.2f hits=%s", conf, hits_spam)
            return "spam", conf, reasons

        logger.debug("Heuristic result: safe")
        return "safe", 0.50, ["no suspicious patterns found"]

    except Exception as e:
        # Absolute safety: never break the request because of heuristics.
        logger.exception("Heuristic classification failed: %s", e)
        return "safe", 0.50, ["heuristic error; defaulted to safe"]
