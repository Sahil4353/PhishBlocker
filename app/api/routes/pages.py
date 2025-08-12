# app/api/routes/pages.py
from __future__ import annotations

from math import ceil
from typing import Generator, Optional

from fastapi import APIRouter, Depends, Form, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, load_only

from app.core.logging import get_logger
from app.db import SessionLocal
from app.models.scan import Scan
from app.services.heuristics import classify_text

logger = get_logger(__name__)
templates = Jinja2Templates(directory="app/templates")
MAX_BODY_PREVIEW_LEN = 500

router = APIRouter()


# -----------------------------------------------------------------------------
# DB dependency
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
# Pages
# -----------------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse, name="index")
def index(request: Request):
    """Render the home page with the scan form."""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.exception("Failed to render index: %s", e)
        return HTMLResponse(
            "<h1>Internal Server Error</h1><p>Failed to render index.</p>",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post("/scan", response_class=HTMLResponse, name="scan_email")
def scan_email(
    request: Request,
    raw: str = Form(...),
    subject: Optional[str] = Form(None),
    sender: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Accept pasted/raw email content, classify, and persist."""
    try:
        raw_text = (raw or "").strip()

        label, confidence, reasons = classify_text(raw_text)
        logger.info(
            "classified email", extra={"label": label, "confidence": confidence}
        )

        record = Scan(
            subject=subject,
            sender=sender,
            raw=raw_text,
            body_preview=raw_text[:MAX_BODY_PREVIEW_LEN],
            label=label,
            confidence=confidence,
            reasons=", ".join(reasons),
        )

        try:
            db.add(record)
            db.commit()
            db.refresh(record)
        except Exception as db_err:
            try:
                db.rollback()
            except Exception:
                pass
            logger.exception("DB commit failed while saving Scan record: %s", db_err)
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "label": "error",
                    "confidence": 0.0,
                    "reasons": ["database error while saving scan"],
                },
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Success → include scan_id so template can link to detail
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "label": label,
                "confidence": confidence,
                "reasons": reasons,
                "scan_id": record.id,
            },
        )

    except Exception as e:
        logger.exception("scan_email failed: %s", e)
        return HTMLResponse(
            "<h1>Internal Server Error</h1><p>Scan failed.</p>",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get("/history", response_class=HTMLResponse, name="history")
def history(
    request: Request,
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    """Paginated history list."""
    try:
        total = db.query(Scan).count()
        pages = max(1, ceil(total / page_size))
        page = min(page, pages)

        scans = (
            db.query(Scan)
            .options(
                load_only(
                    Scan.id,
                    Scan.subject,
                    Scan.sender,
                    Scan.label,
                    Scan.confidence,
                    Scan.body_preview,
                    Scan.created_at,
                )
            )
            .order_by(Scan.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        return templates.TemplateResponse(
            "history.html",
            {
                "request": request,
                "scans": scans,
                "total": total,
                "page": page,
                "pages": pages,
                "page_size": page_size,
            },
        )

    except Exception as e:
        logger.exception("history failed: %s", e)
        return templates.TemplateResponse(
            "history.html",
            {
                "request": request,
                "scans": [],
                "total": 0,
                "page": 1,
                "pages": 1,
                "page_size": page_size,
                "error": "Failed to load history.",
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/scan/{scan_id}/view", response_class=HTMLResponse, name="scan_detail_view"
)
def scan_detail_view(scan_id: int, request: Request, db: Session = Depends(get_db)):
    """
    Render an HTML detail page for a scan.
    Keeps JSON detail in scans.py; this is the page version.
    """
    try:
        scan = db.get(Scan, scan_id)
        if not scan:
            return HTMLResponse(
                "<h1>Not Found</h1><p>Scan not found.</p>",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        payload = {
            "id": scan.id,
            "created_at": (
                scan.created_at.isoformat()
                if getattr(scan, "created_at", None)
                else None
            ),
            "subject": scan.subject,
            "sender": scan.sender,
            "label": scan.label,
            "confidence": float(scan.confidence)
            if scan.confidence is not None
            else None,
            "reasons": (scan.reasons or "").split(", ") if scan.reasons else [],
            "body_preview": scan.body_preview,
        }

        logger.info("rendering scan detail view", extra={"scan_id": scan_id})
        return templates.TemplateResponse(
            "scan_detail.html",
            {"request": request, "scan": payload},
        )

    except Exception as e:
        logger.exception("scan_detail_view failed for id=%s: %s", scan_id, e)
        return HTMLResponse(
            "<h1>Internal Server Error</h1><p>Failed to render scan detail.</p>",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
