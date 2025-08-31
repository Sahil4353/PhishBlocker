from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any, Dict, Generator, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db import SessionLocal
from app.models.scan import Scan
from app.services.scan_pipeline import create_scan_from_text  # ⬅️ pipeline

logger = get_logger(__name__)
router = APIRouter(tags=["api"])  # groups these under "api" in Swagger


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
# Schemas
# -----------------------------------------------------------------------------
class ScanCreate(BaseModel):
    text: str = Field(min_length=1)
    subject: Optional[str] = None
    sender: Optional[str] = None


def _scan_to_payload(s: Scan) -> Dict[str, Any]:
    return {
        "id": s.id,
        "created_at": s.created_at.isoformat()
        if getattr(s, "created_at", None)
        else None,
        "subject": s.subject,
        "sender": s.sender,
        "label": s.label,
        "confidence": float(s.confidence) if s.confidence is not None else None,
        "reasons": (s.reasons or "").split(", ") if s.reasons else [],
        "body_preview": s.body_preview,
    }


# -----------------------------------------------------------------------------
# NEW: POST /api/scan — create a scan from raw text (JSON)
# -----------------------------------------------------------------------------
@router.post("/api/scan")
def create_scan_api(
    payload: ScanCreate, request: Request, db: Session = Depends(get_db)
):
    try:
        model = getattr(request.app.state, "model", None)
        scan: Scan = create_scan_from_text(
            db=db,
            text=payload.text.strip(),
            subject=payload.subject,
            sender=payload.sender,
            model=model,
        )
        logger.info("api created scan id=%s label=%s", scan.id, scan.label)
        return JSONResponse(_scan_to_payload(scan), status_code=status.HTTP_201_CREATED)
    except Exception as e:
        logger.exception("create_scan_api failed: %s", e)
        return JSONResponse(
            {"error": "Internal Server Error", "message": "Failed to scan text."},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# -----------------------------------------------------------------------------
# NEW: GET /api/scans — paginated list
# -----------------------------------------------------------------------------
@router.get("/api/scans")
def list_scans(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    try:
        total = db.query(Scan).count()
        page = max(1, page)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, pages)

        rows: List[Scan] = (
            db.query(Scan)
            .order_by(Scan.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        items = [_scan_to_payload(s) for s in rows]
        return {
            "items": items,
            "total": total,
            "page": page,
            "pages": pages,
            "page_size": page_size,
        }
    except Exception as e:
        logger.exception("list_scans failed: %s", e)
        return JSONResponse(
            {"error": "Internal Server Error", "message": "Failed to list scans."},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# -----------------------------------------------------------------------------
# GET /scan/{scan_id} — lightweight JSON detail (template version is in pages.py)
# -----------------------------------------------------------------------------
@router.get("/scan/{scan_id}")
def scan_detail(scan_id: int, db: Session = Depends(get_db)):
    try:
        scan = db.get(Scan, scan_id)
        if not scan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found"
            )
        logger.info("served scan detail scan_id=%s label=%s", scan_id, scan.label)
        return JSONResponse(_scan_to_payload(scan))

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
    label: Optional[Literal["safe", "spam", "phishing"]] = Query(default=None),
    date_from: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
):
    """
    Stream a CSV of scans.
    Filters:
      - label: safe|spam|phishing
      - date_from/date_to: YYYY-MM-DD (inclusive)
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

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["id", "created_at", "subject", "sender", "label", "confidence", "reasons"]
        )

        rows = q.order_by(Scan.id.desc()).all()
        for s in rows:
            created = s.created_at.isoformat() if getattr(s, "created_at", None) else ""
            writer.writerow(
                [
                    s.id,
                    created,
                    _excel_safe(s.subject or ""),
                    _excel_safe(s.sender or ""),
                    s.label,
                    s.confidence,
                    s.reasons or "",
                ]
            )

        csv_bytes = buf.getvalue()
        logger.info(
            "csv export complete count=%s label=%s date_from=%s date_to=%s",
            len(rows),
            label,
            date_from,
            date_to,
        )
        return StreamingResponse(
            iter([csv_bytes]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=scans.csv"},
        )

    except ValueError as ve:
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
    """Guard against Excel/Sheets formula injection by prefixing risky leading chars."""
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
