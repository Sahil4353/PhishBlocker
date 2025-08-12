import logging
import re
from contextlib import asynccontextmanager
from math import ceil

from fastapi import Depends, FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from sqlalchemy.orm import Session, load_only

from app.db import SessionLocal, engine
from app.models.base import Base
from app.models.scan import Scan  # noqa: F401 (import ensures model is registered)

# -----------------------------------------------------------------------------
# App setup (lifespan replaces on_event("startup") in newer FastAPI style)
# -----------------------------------------------------------------------------

logger = logging.getLogger("phishblocker")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
MAX_BODY_PREVIEW_LEN = 500


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dev convenience: create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    yield  # (place shutdown cleanup here if needed)


app = FastAPI(title="PhishBlocker", version="0.1.0", lifespan=lifespan)
templates = Jinja2Templates(directory="app/templates")


# -----------------------------------------------------------------------------
# DB dependency
# -----------------------------------------------------------------------------


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------------------------------------------------------
# Health & debug
# -----------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/_debug/scan_count")
def scan_count(db: Session = Depends(get_db)):
    return {"count": db.query(Scan).count()}


@app.get("/_debug/cwd")
def dbg_cwd():
    import os

    return {"cwd": os.getcwd()}


# -----------------------------------------------------------------------------
# Tiny heuristic (improved): precompiled regex, multi-hit scoring
# -----------------------------------------------------------------------------

PHISH_PATTERNS = [
    r"\bverify your (?:account|password)\b",
    r"\burgent action required\b",
    r"\bclick (?:here|the link)\b",
    r"\bconfirm your (?:details|identity)\b",
]
SPAM_PATTERNS = [
    r"\bfree\s+money\b",
    r"\bwinner\b",
    r"\bwork from home\b",
    r"\bweight loss\b",
]

PHISH_REGEX = [re.compile(p, re.I) for p in PHISH_PATTERNS]
SPAM_REGEX = [re.compile(p, re.I) for p in SPAM_PATTERNS]


def classify_text(text: str) -> tuple[str, float, list[str]]:
    """
    Returns: (label, confidence, reasons[])
    - Prefer 'phishing' over 'spam' if both match.
    - Confidence scales with number of hits.
    """
    t = text or ""
    hits_phish = [r.pattern for r in PHISH_REGEX if r.search(t)]
    hits_spam = [r.pattern for r in SPAM_REGEX if r.search(t)]

    if hits_phish:
        conf = min(0.70 + 0.10 * len(hits_phish), 0.99)
        reasons = [f"matched: {p}" for p in hits_phish]
        return "phishing", conf, reasons

    if hits_spam:
        conf = min(0.60 + 0.10 * len(hits_spam), 0.95)
        reasons = [f"matched: {p}" for p in hits_spam]
        return "spam", conf, reasons

    return "safe", 0.50, ["no suspicious patterns found"]


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/scan", response_class=HTMLResponse)
def scan_email(
    request: Request,
    raw: str = Form(...),
    subject: str | None = Form(None),
    sender: str | None = Form(None),
    db: Session = Depends(get_db),
):
    # Minimal normalization
    raw_text = (raw or "").strip()

    label, confidence, reasons = classify_text(raw_text)
    logger.info("classified email", extra={"label": label, "confidence": confidence})

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
    except Exception:
        db.rollback()
        logger.exception("DB commit failed while saving Scan record")
        # You could render an error page instead; keeping it simple:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "label": "error",
                "confidence": 0.0,
                "reasons": ["database error while saving scan"],
            },
            status_code=500,
        )

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "label": label,
            "confidence": confidence,
            "reasons": reasons,
        },
    )


@app.get("/history", response_class=HTMLResponse)
def history(
    request: Request,
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    total = db.query(Scan).count()
    pages = max(1, ceil(total / page_size))
    page = min(page, pages)

    # Lighter query: avoid loading large 'raw' payload if not needed in the table
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
                (
                    Scan.created_at if hasattr(Scan, "created_at") else Scan.id
                ),  # keep sort info if exists
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
