from __future__ import annotations

import re
from email import policy
from email.parser import BytesParser
from typing import Any, Dict, List, Optional


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
URL_RE = re.compile(r"https?://[^\s)>\]\"']+", re.IGNORECASE)


def extract_urls(text: str | None) -> List[str]:
    if not text:
        return []
    return [m.group(0).strip(").,>]}\"'") for m in URL_RE.finditer(text)]


# --- Auth & Received parsing ---
AUTH_RE = re.compile(r"(spf|dkim|dmarc)\s*=\s*([a-zA-Z]+)", re.I)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def parse_auth_results(hdr: str | None) -> Dict[str, Optional[str]]:
    res: Dict[str, Optional[str]] = {
        "spf_result": None,
        "dkim_result": None,
        "dmarc_result": None,
    }
    if not hdr:
        return res
    for mech, val in AUTH_RE.findall(hdr):
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
    matching your canonical Email schema fields (strings/lists only).
    """
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)

    # Headers
    subject = msg.get("Subject", "") or ""
    sender = (msg.get("From", "") or "").strip()
    to = msg.get("To", "") or ""
    cc = msg.get("Cc", "") or ""
    bcc = msg.get("Bcc", "") or ""
    reply_to = msg.get("Reply-To", "") or ""
    return_path = msg.get("Return-Path", "") or ""
    message_id = msg.get("Message-ID", "") or ""
    date_hdr = msg.get("Date", "") or ""
    list_unsub = msg.get("List-Unsubscribe", "") or ""
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
            ctype = p.get_content_type()
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
        ctype = msg.get_content_type()
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
            text_plain = str(payload)

    body_text = text_plain or html_to_text(html_raw)

    # URLs & Received IPs
    urls = list({*extract_urls(body_text), *extract_urls(html_raw)})
    received_ips: List[str] = []
    for h in msg.get_all("Received", []) or []:
        received_ips += IP_RE.findall(h)
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
