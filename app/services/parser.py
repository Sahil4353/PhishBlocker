# app/services/parser.py
from __future__ import annotations

import re
from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser
from typing import Any, Dict, List, Optional


# --- Header decoding helpers ---
def _decode_unfold(value: str | None) -> str:
    """
    Decode RFC 2047-encoded headers and unfold whitespace.
    Returns a safe str.
    """
    if not value:
        return ""
    try:
        # make_header handles multiple encoded-word segments
        txt = str(make_header(decode_header(value)))
    except Exception:
        txt = value
    # Unfold CRLF-wrapped header lines and collapse runs of whitespace
    return re.sub(r"\s+", " ", txt).strip()


# --- HTML → text (tolerant if bs4 isn't installed) ---
def html_to_text(html: str | None) -> str:
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        # fallback: strip tags crudely
        return re.sub(r"<[^>]+>", " ", html or "").strip()
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style"]):
        t.extract()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


# --- URL extraction ---
_URL_RE = re.compile(r"https?://[^\s)>\]\"']+", re.IGNORECASE)


def extract_urls(text: str | None) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    seen = set()
    for m in _URL_RE.finditer(text):
        u = m.group(0).strip(").,>]}\"'")
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    # Cap pathological cases to keep JSON reasonable
    return out[:500]


# --- Auth & Received parsing ---
_AUTH_RE = re.compile(r"(spf|dkim|dmarc)\s*=\s*([a-zA-Z]+)", re.I)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6_RE = re.compile(r"\b[0-9a-fA-F:]{2,}\b")  # loose but practical


def parse_auth_results(hdr: str | None) -> Dict[str, Optional[str]]:
    res: Dict[str, Optional[str]] = {
        "spf_result": None,
        "dkim_result": None,
        "dmarc_result": None,
    }
    if not hdr:
        return res
    for mech, val in _AUTH_RE.findall(hdr):
        k = f"{mech.lower()}_result"
        if res.get(k) is None:  # first hit wins
            res[k] = val.lower()
    return res


def part_is_attachment(p) -> bool:
    cd = p.get_content_disposition()
    return bool(cd == "attachment" or p.get_filename())


# --- Main parser ---
def parse_rfc822(raw_bytes: bytes) -> Dict[str, Any]:
    """
    Parse raw RFC-822 message bytes into a provider-agnostic dict
    matching the canonical Email schema (strings/lists only).
    """
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)

    # Headers (decoded & unfolded)
    subject = _decode_unfold(msg.get("Subject", ""))
    sender = _decode_unfold(msg.get("From", "")).strip()
    to = _decode_unfold(msg.get("To", ""))
    cc = _decode_unfold(msg.get("Cc", ""))
    bcc = _decode_unfold(msg.get("Bcc", ""))
    reply_to = _decode_unfold(msg.get("Reply-To", ""))
    return_path = _decode_unfold(msg.get("Return-Path", ""))
    message_id = _decode_unfold(msg.get("Message-ID", ""))
    date_hdr = _decode_unfold(msg.get("Date", ""))
    list_unsub = _decode_unfold(msg.get("List-Unsubscribe", ""))
    auth_hdr = msg.get("Authentication-Results", "") or ""
    auth = parse_auth_results(auth_hdr)

    # Bodies + attachments
    text_plain, html_raw = "", ""
    attachments: List[str] = []

    if msg.is_multipart():
        for p in msg.walk():
            if p.get_content_maintype() == "multipart":
                continue
            if part_is_attachment(p):
                attachments.append(p.get_filename() or p.get_content_type())
                continue
            ctype = (p.get_content_type() or "").lower()
            try:
                payload = p.get_content()
            except Exception:
                payload = (p.get_payload(decode=True) or b"").decode(
                    "utf-8", errors="ignore"
                )
            if ctype == "text/plain" and not text_plain:
                text_plain = str(payload)
            elif ctype == "text/html" and not html_raw:
                html_raw = str(payload)
    else:
        ctype = (msg.get_content_type() or "").lower()
        try:
            payload = msg.get_content()
        except Exception:
            payload = (msg.get_payload(decode=True) or b"").decode(
                "utf-8", errors="ignore"
            )
        if ctype == "text/plain":
            text_plain = str(payload)
        elif ctype == "text/html":
            html_raw = str(payload)
        else:
            # unknown type: best effort as text
            text_plain = str(payload)

    body_text = text_plain or html_to_text(html_raw)
    body_text = re.sub(r"\s+", " ", body_text or "").strip()

    # URLs & Received IPs
    urls = list({*extract_urls(body_text), *extract_urls(html_raw)})
    received_ips: List[str] = []
    for h in msg.get_all("Received", []) or []:
        # Collect IPv4 and IPv6 candidates
        received_ips += _IPV4_RE.findall(h)
        received_ips += _IPV6_RE.findall(h)
    # de-dup keep order
    seen = set()
    received_ips = [ip for ip in received_ips if not (ip in seen or seen.add(ip))]

    return {
        "subject": subject,
        "sender": sender,
        "to": to,
        "cc": cc,
        "bcc": bcc,
        "reply_to": reply_to,
        "return_path": return_path,
        "message_id": message_id,
        "timestamp": date_hdr,  # keep raw header string; parse later if needed
        "body_text": body_text,
        "html_raw": html_raw,
        "has_text": int(bool(text_plain)),
        "has_html": int(bool(html_raw)),
        "urls": urls,
        "attachments": [a for a in attachments if a],
        "attachments_count": sum(1 for a in attachments if a),
        "list_unsubscribe": list_unsub,
        "spf_result": auth["spf_result"],
        "dkim_result": auth["dkim_result"],
        "dmarc_result": auth["dmarc_result"],
        "received_ips": received_ips,
    }
