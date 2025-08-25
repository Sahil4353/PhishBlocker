from __future__ import annotations

import re
from typing import Dict, List, Tuple
from urllib.parse import urlparse


# --- Public API kept the same for pages.py ---
def classify_text(text: str) -> Tuple[str, float, List[str]]:
    """
    Very light baseline heuristics for text-only classification.
    Returns (label, confidence, reasons). Labels: safe|spam|phishing
    """
    text_l = (text or "").lower()
    reasons: List[str] = []
    score = 0.0

    # Signals
    urgent = any(
        p in text_l
        for p in (
            "urgent",
            "immediately",
            "verify your account",
            "password expiry",
            "final notice",
            "suspend",
            "limited time",
            "unauthorized",
            "confirm your identity",
        )
    )
    if urgent:
        score += 0.35
        reasons.append("urgent language")

    # Links that look odd (IP / punycode / at-signs before domain)
    urls = re.findall(r"https?://\S+", text or "")
    has_ip_url = any(_looks_like_ip(url) for url in urls)
    has_puny = any("xn--" in url for url in urls)
    if has_ip_url:
        score += 0.35
        reasons.append("url uses raw ip")
    if has_puny:
        score += 0.2
        reasons.append("punycode url")

    # Credential bait
    cred_bait = any(
        p in text_l
        for p in ("update your password", "login now", "re-enter your details")
    )
    if cred_bait:
        score += 0.25
        reasons.append("credential lure")

    # Marketing-y (spammy) signals
    spammy = any(
        p in text_l
        for p in ("free trial", "special offer", "unsubscribe", "act now", "win a")
    )
    if spammy and score < 0.5:
        reasons.append("marketing language")

    # Decide
    if score >= 0.6:
        return ("phishing", min(0.99, 0.5 + score / 2), reasons)
    if spammy:
        return ("spam", 0.6, reasons)
    return ("safe", 0.7 if not (urgent or cred_bait or urls) else 0.5, reasons)


def _looks_like_ip(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))
    except Exception:
        return False


# --- Header/URL analyzer for pipeline JSON (booleans/ints only) ---
def analyze(parsed: Dict) -> Dict:
    """
    Analyze parsed email dict (from services.parser.parse_rfc822).
    Returns JSON-serializable flags for storage in Scan.header_flags.
    """
    sender = parsed.get("sender") or ""
    reply_to = parsed.get("reply_to") or ""
    return_path = parsed.get("return_path") or ""

    sender_dom = _domain(sender)
    reply_dom = _domain(reply_to)
    ret_dom = _domain(return_path)

    reply_mismatch = bool(sender_dom and reply_dom and sender_dom != reply_dom)
    return_mismatch = bool(sender_dom and ret_dom and sender_dom != ret_dom)

    spf = (parsed.get("spf_result") or "").lower()
    dkim = (parsed.get("dkim_result") or "").lower()
    dmarc = (parsed.get("dmarc_result") or "").lower()

    urls = parsed.get("urls") or []
    has_ip = any(_looks_like_ip(u) for u in urls)
    has_puny = any("xn--" in (u or "") for u in urls)

    return {
        "has_html": int(bool(parsed.get("has_html"))),
        "has_text": int(bool(parsed.get("has_text"))),
        "urls_count": int(len(urls)),
        "url_has_ip": int(has_ip),
        "url_has_punycode": int(has_puny),
        "reply_to_mismatch": int(reply_mismatch),
        "return_path_mismatch": int(return_mismatch),
        "spf_fail": int(spf == "fail"),
        "dkim_fail": int(dkim == "fail"),
        "dmarc_fail": int(dmarc == "fail"),
    }


def _domain(addr: str) -> str:
    m = re.search(r"@([A-Za-z0-9.-]+)", addr or "")
    return m.group(1).lower() if m else ""
