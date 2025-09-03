# app/api/routes/pages.py
from __future__ import annotations

from math import ceil
from typing import Generator, Optional

from fastapi import APIRouter, Depends, Form, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, load_only

from app.core.logging import get_logger
from app.db import SessionLocal
from app.models.scan import Scan
from app.services.scan_pipeline import create_scan_from_text  # ⬅️ use pipeline

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


# PRG: on success we redirect to /scan/{id}/view instead of rendering directly
@router.post("/scan", response_class=HTMLResponse, name="scan_email")
def scan_email(
    request: Request,
    raw: str = Form(...),
    subject: Optional[str] = Form(None),
    sender: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Accept pasted/raw email content, classify via pipeline, persist Email+Scan,
    then redirect (PRG) to the detail page.
    """
    try:
        raw_text = (raw or "").strip()
        if not raw_text:
            return HTMLResponse(
                "<h1>Bad Request</h1><p>Empty message body.</p>",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # Pull model from app.state if available; pipeline should gracefully
        # fallback to heuristics when model is None.
        model = getattr(request.app.state, "model", None)

        # Pipeline persists Email + Scan and returns the Scan row
        scan: Scan = create_scan_from_text(
            db=db,
            body_text=raw_text,
            subject=(subject or "").strip() or None,
            sender=(sender or "").strip() or None,
            model=model,
        )

        logger.info(
            "pipeline created scan id=%s label=%s conf=%.3f",
            scan.id,
            scan.label,
            float(scan.confidence) if scan.confidence is not None else -1.0,
        )

        # PRG: 303 See Other -> always follow with GET to the detail page
        detail_url = request.url_for(
            "scan_detail_view", scan_id=scan.id
        ).include_query_params(ok="1")
        return RedirectResponse(url=str(detail_url), status_code=303)

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

        # Prefer legacy reasons string, else try details_json.reasons (if present)
        reasons_list = (scan.reasons or "").split(", ") if scan.reasons else []
        if not reasons_list and hasattr(scan, "details_json") and scan.details_json:
            try:
                r = scan.details_json.get("reasons")
                if isinstance(r, list):
                    # list of {"token": "..."} or list[str]
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
                "ok": request.query_params.get("ok") == "1",  # for optional banner
            },
        )

    except Exception as e:
        logger.exception("scan_detail_view failed for id=%s: %s", scan_id, e)
        return HTMLResponse(
            "<h1>Internal Server Error</h1><p>Failed to render scan detail.</p>",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
