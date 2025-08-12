from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Generator, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db import SessionLocal
from app.models.scan import Scan

logger = get_logger(__name__)
router = APIRouter()


# -----------------------------------------------------------------------------
# DB dependency (defensive)
# -----------------------------------------------------------------------------
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.exception("DB dependency error: %s", e)
        raise
    finally:
        try:
            db.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# GET /scan/{scan_id} — lightweight JSON detail
# -----------------------------------------------------------------------------
@router.get("/scan/{scan_id}")
def scan_detail(scan_id: int, db: Session = Depends(get_db)):
    try:
        scan = db.get(Scan, scan_id)
        if not scan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found"
            )

        payload = {
            "id": scan.id,
            "created_at": scan.created_at.isoformat()
            if getattr(scan, "created_at", None)
            else None,
            "subject": scan.subject,
            "sender": scan.sender,
            "label": scan.label,
            "confidence": float(scan.confidence)
            if scan.confidence is not None
            else None,
            "reasons": (scan.reasons or "").split(", ") if scan.reasons else [],
            "body_preview": scan.body_preview,
        }
        logger.info(
            "served scan detail", extra={"scan_id": scan_id, "label": scan.label}
        )
        return JSONResponse(payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("scan_detail failed for id=%s: %s", scan_id, e)
        return JSONResponse(
            {
                "error": "Internal Server Error",
                "message": "Failed to load scan detail.",
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# -----------------------------------------------------------------------------
# GET /export/csv — optional filters: label, date_from, date_to
# -----------------------------------------------------------------------------
@router.get("/export/csv")
def export_csv(
    db: Session = Depends(get_db),
    label: Optional[str] = Query(default=None, pattern="^(safe|spam|phishing)$"),
    date_from: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
):
    """
    Stream a CSV of scans. Excel-injection guarded.
    Filters:
      - label: safe|spam|phishing
      - date_from/date_to: YYYY-MM-DD (inclusive)
    CSV header:
      id, created_at, subject, sender, label, confidence, reasons, body_preview
    """
    try:
        q = db.query(Scan)

        if label:
            q = q.filter(Scan.label == label)

        start_dt = _parse_date(date_from) if date_from else None
        end_dt = _parse_date(date_to, end_of_day=True) if date_to else None

        if start_dt and hasattr(Scan, "created_at"):
            q = q.filter(Scan.created_at >= start_dt)
        if end_dt and hasattr(Scan, "created_at"):
            q = q.filter(Scan.created_at <= end_dt)

        rows = q.order_by(Scan.id.desc()).all()

        # Build CSV in-memory (fine for small/medium datasets).
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "id",
                "created_at",
                "subject",
                "sender",
                "label",
                "confidence",
                "reasons",
                "body_preview",  # <-- added
            ]
        )

        for s in rows:
            created = s.created_at.isoformat() if getattr(s, "created_at", None) else ""
            writer.writerow(
                [
                    s.id,
                    created,
                    _excel_safe(s.subject or ""),
                    _excel_safe(s.sender or ""),
                    s.label or "",
                    s.confidence if s.confidence is not None else "",
                    s.reasons or "",
                    _excel_safe(s.body_preview or ""),  # <-- guarded
                ]
            )

        buf.seek(0)
        logger.info(
            "csv export complete",
            extra={
                "count": len(rows),
                "label": label,
                "date_from": date_from,
                "date_to": date_to,
            },
        )
        return StreamingResponse(
            iter([buf.read()]),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="scans.csv"'},
        )

    except ValueError as ve:
        # e.g., bad date format
        logger.warning("csv export validation error: %s", ve)
        return PlainTextResponse(str(ve), status_code=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.exception("csv export failed: %s", e)
        return PlainTextResponse(
            "Internal Server Error: failed to export CSV.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _excel_safe(val: str) -> str:
    """
    Guard against Excel/Sheets formula injection by prefixing risky leading chars.
    """
    return f"'{val}" if val[:1] in ("=", "+", "-", "@") else val


def _parse_date(value: str, end_of_day: bool = False) -> datetime:
    """
    Parse YYYY-MM-DD. If end_of_day=True, set time to 23:59:59.999999.
    Raises ValueError on bad formats.
    """
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        return dt
    except Exception:
        raise ValueError(f"Invalid date format '{value}'. Use YYYY-MM-DD.")
