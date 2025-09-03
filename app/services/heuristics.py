# app/services/heuristics.py
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
    text = text or ""
    text_l = text.lower()
    reasons: List[str] = []
    phish_score = 0.0
    spam_score = 0.0

    # --- Signals (use word boundaries where possible) ---
    def has_any(patterns: List[str]) -> bool:
        return any(re.search(p, text_l) for p in patterns)

    urgent = has_any(
        [
            r"\burgent\b",
            r"\bimmediately\b",
            r"\bverify (your )?account\b",
            r"\b(password|passcode)\s*(expiry|expires?)\b",
            r"\bfinal notice\b",
            r"\bsuspend(ed|ing)?\b",
            r"\blimited time\b",
            r"\bunauthorized\b",
            r"\bconfirm your identity\b",
        ]
    )
    if urgent:
        phish_score += 0.35
        reasons.append("urgent language")

    # Credential bait
    cred_bait = has_any(
        [
            r"\bupdate (your )?password\b",
            r"\blog[\s-]*in now\b",
            r"\bre[- ]?enter your details\b",
            r"\b2fa (reset|disable)\b",
        ]
    )
    if cred_bait:
        phish_score += 0.25
        reasons.append("credential lure")

    # URLs
    urls = _extract_urls(text)
    has_ip_url = any(_looks_like_ip(u) for u in urls)
    has_ipv6_url = any(_looks_like_ipv6(u) for u in urls)
    has_puny = any("xn--" in u for u in urls)
    has_at_userinfo = any(_url_has_userinfo(u) for u in urls)  # e.g., http://user@host

    if has_ip_url or has_ipv6_url:
        phish_score += 0.35
        reasons.append("url uses raw ip")
    if has_at_userinfo:
        phish_score += 0.2
        reasons.append("url has @ userinfo")
    if has_puny:
        phish_score += 0.2
        reasons.append("punycode url")

    # Marketing-y (spammy) signals
    spammy = has_any(
        [
            r"\bfree trial\b",
            r"\bspecial offer\b",
            r"\bunsubscribe\b",
            r"\bact now\b",
            r"\bwin a\b",
        ]
    )
    if spammy:
        spam_score += 0.35
        reasons.append("marketing language")

    # Shoutiness / sales-y formatting
    exclaim_heavy = text.count("!") >= 3
    caps_ratio = _caps_ratio(text)
    money_lure = has_any([r"\$\d", r"\b\d+% off\b", r"\bdiscount\b"])
    if exclaim_heavy:
        spam_score += 0.1
        reasons.append("excess punctuation")
    if caps_ratio >= 0.35:
        spam_score += 0.1
        reasons.append("shouty text")
    if money_lure:
        spam_score += 0.1
        reasons.append("money lure")

    # --- Decide ---
    # Basic precedence: phishing if phish_score strong; else spam; else safe.
    if phish_score >= 0.6 or (phish_score >= 0.45 and spam_score < 0.4):
        conf = _clip01(0.5 + phish_score / 2)
        return ("phishing", conf, reasons[:8])

    if spam_score >= 0.5 or (spam_score >= 0.35 and phish_score < 0.45):
        conf = _clip01(0.55 + spam_score / 2.5)
        return ("spam", conf, reasons[:8])

    # Default safe; lower confidence if any risky signals present
    risky = urgent or cred_bait or bool(urls)
    conf = 0.7 if not risky else 0.5
    return ("safe", conf, reasons[:8])


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# Robust URL extractor for http/https (handles trailing punctuation)
_URL_RE = re.compile(r"""https?://[^\s'">)]+""", flags=re.IGNORECASE)


def _extract_urls(text: str) -> List[str]:
    raw = _URL_RE.findall(text or "")
    # strip common trailing punctuation
    return [u.rstrip(").,;:!?]") for u in raw]


def _looks_like_ip(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))
    except Exception:
        return False


def _looks_like_ipv6(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        # very loose IPv6 check (colon presence + hex groups)
        return ":" in host and bool(
            re.fullmatch(r"[0-9a-f:]+", host, flags=re.IGNORECASE)
        )
    except Exception:
        return False


def _url_has_userinfo(url: str) -> bool:
    try:
        # Detect presence of userinfo@host in URL
        parsed = urlparse(url)
        return "@" in (parsed.netloc or "") and parsed.username is not None
    except Exception:
        return False


def _caps_ratio(s: str) -> float:
    if not s:
        return 0.0
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return 0.0
    caps = sum(1 for ch in letters if ch.isupper())
    return caps / len(letters)


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
    has_ip = any(_looks_like_ip(u) or _looks_like_ipv6(u) for u in urls)
    has_puny = any("xn--" in (u or "") for u in urls)
    has_at = any(_url_has_userinfo(u) for u in urls)

    return {
        "has_html": int(bool(parsed.get("has_html"))),
        "has_text": int(bool(parsed.get("has_text"))),
        "urls_count": int(len(urls)),
        "url_has_ip": int(has_ip),
        "url_has_punycode": int(has_puny),
        "url_has_userinfo": int(has_at),
        "reply_to_mismatch": int(reply_mismatch),
        "return_path_mismatch": int(return_mismatch),
        "spf_fail": int(spf == "fail"),
        "dkim_fail": int(dkim == "fail"),
        "dmarc_fail": int(dmarc == "fail"),
    }


def _domain(addr: str) -> str:
    m = re.search(r"@([A-Za-z0-9.-]+)", addr or "")
    return m.group(1).lower() if m else ""
