# services/scan_pipeline.py
from __future__ import annotations

import json
from datetime import datetime
from email.utils import parsedate_to_datetime
from hashlib import sha256
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.models import Email, Scan
from app.services import heuristics
from app.services.parser import parse_rfc822


# --- helpers ---
def _short_id(*vals: str) -> str:
    h = sha256()
    for v in vals:
        h.update((v or "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _parse_ts(date_hdr: str | None) -> Optional[datetime]:
    if not date_hdr:
        return None
    try:
        return parsedate_to_datetime(date_hdr)
    except Exception:
        return None


def _to_reason_string(reasons: Any) -> Optional[str]:
    """
    Normalize explanations into a simple, human-readable string for the legacy `Scan.reasons` column.
    - If list[{"token": "..."}], join top tokens.
    - If list[str], join directly.
    - Else, stringify.
    """
    if reasons is None:
        return None
    try:
        if isinstance(reasons, list):
            # list of dicts with 'token' OR list of strings
            tokens = []
            for r in reasons:
                if isinstance(r, dict) and "token" in r:
                    tokens.append(str(r["token"]))
                elif isinstance(r, str):
                    tokens.append(r)
            if tokens:
                return ", ".join(tokens[:8])  # keep it shortish
        # Fallback: compact string
        return str(reasons)
    except Exception:
        return None


def _maybe_set_json_attr(obj: Any, attr: str, value: Any) -> None:
    """
    Safely set JSON-capable attributes if the column exists on the model.
    (Prevents TypeErrors from kwargs on older schemas.)
    """
    if hasattr(obj, attr):
        setattr(obj, attr, value)


# --- main entry points ---
def create_scan_from_eml_bytes(
    db: Session,
    raw_bytes: bytes,
    *,
    source: str = "upload",
    direction: str = "upload",
    model: Any = None,
) -> Scan:
    """
    Parse raw RFC-822 bytes, persist Email, run heuristics/ML, persist Scan.
    Returns the Scan row.
    """
    parsed = parse_rfc822(raw_bytes)
    return _persist_scan(db, parsed, source=source, direction=direction, model=model)


def create_scan_from_text(
    db: Session,
    body_text: str,
    *,
    subject: str | None = None,
    sender: str | None = None,
    source: str = "upload",
    direction: str = "upload",
    model: Any = None,
) -> Scan:
    """
    Minimal path for pasted text (no headers). Still persists Email.
    """
    parsed = {
        "subject": subject or "",
        "sender": sender or "",
        "to": "",
        "cc": "",
        "bcc": "",
        "reply_to": "",
        "return_path": "",
        "message_id": "",
        "timestamp": "",
        "body_text": body_text or "",
        "html_raw": "",
        "has_text": 1 if body_text else 0,
        "has_html": 0,
        "urls": [],
        "attachments": [],
        "attachments_count": 0,
        "list_unsubscribe": "",
        "spf_result": None,
        "dkim_result": None,
        "dmarc_result": None,
        "received_ips": [],
    }
    return _persist_scan(db, parsed, source=source, direction=direction, model=model)


# --- core persist logic ---
def _persist_scan(
    db: Session,
    parsed: Dict[str, Any],
    *,
    source: str,
    direction: str,
    model: Any,
) -> Scan:
    # Stable id from headers; for text-only path message_id may be empty (that's fine)
    eid = _short_id(
        parsed.get("message_id", ""),
        parsed.get("subject", ""),
        parsed.get("timestamp", ""),
        parsed.get("sender", ""),
    )

    email = db.get(Email, eid)
    if not email:
        email = Email(
            id=eid,
            source=source,
            message_id=parsed.get("message_id"),
            subject=parsed.get("subject"),
            sender=parsed.get("sender"),
            recipients=";".join(
                [parsed.get("to", ""), parsed.get("cc", ""), parsed.get("bcc", "")]
            ).strip(";"),
            reply_to=parsed.get("reply_to"),
            return_path=parsed.get("return_path"),
            timestamp=_parse_ts(parsed.get("timestamp")),
            body_text=parsed.get("body_text"),
            html_raw=parsed.get("html_raw"),
            has_text=int(bool(parsed.get("has_text"))),
            has_html=int(bool(parsed.get("has_html"))),
            urls=parsed.get("urls"),
            attachments=parsed.get("attachments"),
            attachments_count=int(parsed.get("attachments_count") or 0),
            list_unsubscribe=parsed.get("list_unsubscribe"),
            spf_result=parsed.get("spf_result"),
            dkim_result=parsed.get("dkim_result"),
            dmarc_result=parsed.get("dmarc_result"),
            received_ips=parsed.get("received_ips"),
        )
        db.add(email)

    # Heuristics + optional model
    hx = heuristics.analyze(parsed)  # dict of booleans/ints

    if model:
        # Your ModelService signature: (label, prob, reasons)
        label, confidence, reasons = model.predict_with_explanations(
            text=email.body_text or "", meta=parsed
        )
        # Optional: full distribution if available
        try:
            probs_map = model.predict_proba_map(email.body_text or "")
        except Exception:
            probs_map = None
    else:
        # Heuristic-only classification returns similar tuple
        label, confidence, reasons = heuristics.classify_text(email.body_text or "")
        probs_map = None

    # Keep legacy 'reasons' column human-friendly; store full reasons in details_json if available.
    reasons_str = _to_reason_string(reasons)

    scan = Scan(
        email_id=email.id,
        subject=email.subject,
        sender=email.sender,
        body_preview=(email.body_text or "")[:500],
        label=label,
        confidence=float(confidence) if confidence is not None else None,
        reasons=reasons_str,
        header_flags=hx,
        model_version=getattr(model, "version", None),
        direction=direction,
    )
    db.add(scan)

    # If schema supports JSON fields (recent migration), persist richer outputs.
    # Safe no-op if columns are absent.
    _maybe_set_json_attr(scan, "details_json", {"reasons": reasons})
    if probs_map is not None:
        _maybe_set_json_attr(scan, "probs_json", probs_map)

    db.commit()
    db.refresh(scan)
    return scan
