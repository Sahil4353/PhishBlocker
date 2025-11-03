# app/api/routes/pages.py
from __future__ import annotations

from math import ceil
from datetime import datetime as _dt
from typing import Generator, Optional

from fastapi import (
    APIRouter,
    Depends,
    Form,
    Query,
    Request,
    status,
    HTTPException,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, load_only
from sqlalchemy import func

from app.core.logging import get_logger
from app.db import SessionLocal
from app.models.scan import Scan
from app.models.feedback import Feedback, VALID_LABELS
from app.services.scan_pipeline import create_scan_from_text  # ⬅️ pipeline

logger = get_logger(__name__)
templates = Jinja2Templates(directory="app/templates")
MAX_BODY_PREVIEW_LEN = 500

router = APIRouter(tags=["pages"])  # purely for Swagger grouping


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


# Optional nicety: any GET /scan becomes a redirect to "/"
@router.get("/scan", include_in_schema=False, name="scan_get_redirect")
def scan_get_redirect(request: Request):
    return RedirectResponse(url=request.url_for("index"), status_code=303)


# -----------------------------------------------------------------------------
# Scan email route (POST)
# -----------------------------------------------------------------------------
@router.post("/scan", response_class=HTMLResponse, name="scan_email")
def scan_email(
    request: Request,
    raw: str = Form(...),
    subject: Optional[str] = Form(None),
    sender: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Accept pasted/raw email content, classify via pipeline, persist Email+Scan."""
    try:
        raw_text = (raw or "").strip()
        if not raw_text:
            return HTMLResponse(
                "<h1>Bad Request</h1><p>Empty message body.</p>",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        model = getattr(request.app.state, "model", None)

        scan_obj: Scan = create_scan_from_text(
            db=db,
            body_text=raw_text,
            subject=(subject or "").strip() or None,
            sender=(sender or "").strip() or None,
            model=model,
        )

        logger.info(
            "pipeline created scan id=%s label=%s conf=%.3f",
            scan_obj.id,
            scan_obj.label,
            float(scan_obj.confidence) if scan_obj.confidence is not None else -1.0,
        )

        detail_url = request.url_for(
            "scan_detail_view", scan_id=scan_obj.id
        ).include_query_params(ok="1")
        return RedirectResponse(url=str(detail_url), status_code=303)

    except Exception as e:
        logger.exception("scan_email failed: %s", e)
        return HTMLResponse(
            "<h1>Internal Server Error</h1><p>Scan failed.</p>",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# -----------------------------------------------------------------------------
# History page with filters
# -----------------------------------------------------------------------------
@router.get("/history", response_class=HTMLResponse, name="history")
def history(
    request: Request,
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    label: Optional[str] = Query(default=None),
    sender: Optional[str] = Query(default=None),
    domain: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
):
    """Paginated history list with optional filters."""
    try:
        q = db.query(Scan)

        if label in ("safe", "spam", "phishing"):
            q = q.filter(Scan.label == label)
        if sender:
            q = q.filter(Scan.sender == sender)
        if domain:
            q = q.filter(func.substr(Scan.sender, func.instr(Scan.sender, "@") + 1) == domain)
        if date_from:
            start_dt = _parse_date(date_from)
            if hasattr(Scan, "created_at"):
                q = q.filter(Scan.created_at >= start_dt)
        if date_to:
            end_dt = _parse_date(date_to, end_of_day=True)
            if hasattr(Scan, "created_at"):
                q = q.filter(Scan.created_at <= end_dt)

        total = q.count()
        pages = max(1, ceil(total / page_size))
        page = min(max(1, page), pages)

        scans = (
            q.options(
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
                "label": label or "",
                "sender": sender or "",
                "domain": domain or "",
                "date_from": date_from or "",
                "date_to": date_to or "",
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


# -----------------------------------------------------------------------------
# Feedback form (UI) — POST
# -----------------------------------------------------------------------------
@router.post("/scan/{scan_id}/feedback", name="add_feedback_page")
def add_feedback_page(
    scan_id: int,
    request: Request,
    user_label: str = Form(...),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Accept user feedback (Safe/Spam/Phishing) from the UI form."""
    try:
        if user_label not in VALID_LABELS:
            raise HTTPException(status_code=400, detail="Invalid label")

        scan_obj = db.get(Scan, scan_id)
        if not scan_obj:
            raise HTTPException(status_code=404, detail="Scan not found")

        fb = Feedback(scan_id=scan_id, user_label=user_label, notes=notes, source="ui")
        db.add(fb)
        db.commit()

        detail_url = request.url_for("scan_detail_view", scan_id=scan_id).include_query_params(ok="1")
        return RedirectResponse(url=str(detail_url), status_code=303)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("add_feedback_page failed: %s", e)
        return HTMLResponse(
            "<h1>Internal Server Error</h1><p>Failed to save feedback.</p>",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# -----------------------------------------------------------------------------
# Scan detail view
# -----------------------------------------------------------------------------
@router.get("/scan/{scan_id}/view", response_class=HTMLResponse, name="scan_detail_view")
def scan_detail_view(scan_id: int, request: Request, db: Session = Depends(get_db)):
    """Render HTML detail page for a scan."""
    try:
        scan = db.get(Scan, scan_id)
        if not scan:
            return HTMLResponse(
                "<h1>Not Found</h1><p>Scan not found.</p>",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        # Prefer legacy reasons string, else try structured details.reasons
        reasons_list = (scan.reasons or "").split(", ") if scan.reasons else []
        if not reasons_list and hasattr(scan, "details") and scan.details:
            try:
                r = scan.details.get("reasons")
                if isinstance(r, list):
                    tokens = []
                    for it in r:
                        if isinstance(it, dict) and "token" in it:
                            tokens.append(str(it["token"]))
                        elif isinstance(it, str):
                            tokens.append(it)
                    if tokens:
                        reasons_list = tokens[:8]
            except Exception:
                pass

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
            "reasons": reasons_list,
            "body_preview": scan.body_preview,
        }

        logger.info("rendering scan detail view scan_id=%s", scan_id)
        return templates.TemplateResponse(
            "scan_detail.html",
            {
                "request": request,
                "scan": payload,
                "ok": request.query_params.get("ok") == "1",
            },
        )

    except Exception as e:
        logger.exception("scan_detail_view failed for id=%s: %s", scan_id, e)
        return HTMLResponse(
            "<h1>Internal Server Error</h1><p>Failed to render scan detail.</p>",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# -----------------------------------------------------------------------------
# Helper: parse YYYY-MM-DD
# -----------------------------------------------------------------------------
def _parse_date(value: str, end_of_day: bool = False) -> _dt:
    """Parse date from string; adjust to 23:59:59 if end_of_day=True."""
    try:
        dt = _dt.strptime(value, "%Y-%m-%d")
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        return dt
    except Exception:
        raise ValueError(f"Invalid date format '{value}'. Use YYYY-MM-DD.")
