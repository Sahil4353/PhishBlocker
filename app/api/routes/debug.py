from __future__ import annotations

import os
from typing import Generator

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db import SessionLocal
from app.models.scan import Scan

logger = get_logger(__name__)
router = APIRouter()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.exception("DB dependency error (debug): %s", e)
        raise
    finally:
        try:
            db.close()
        except Exception:
            pass


@router.get("/health")
def health():
    try:
        return {"status": "ok"}
    except Exception as e:
        logger.exception("health endpoint failed: %s", e)
        return JSONResponse(
            {"status": "error", "message": "Health check failed"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get("/_debug/scan_count")
def scan_count(db: Session = Depends(get_db)):
    try:
        count = db.query(Scan).count()
        return {"count": count}
    except Exception as e:
        logger.exception("_debug/scan_count failed: %s", e)
        return JSONResponse(
            {"error": "Failed to get scan count"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get("/_debug/cwd")
def dbg_cwd():
    try:
        return {"cwd": os.getcwd()}
    except Exception as e:
        logger.exception("_debug/cwd failed: %s", e)
        return JSONResponse(
            {"error": "Failed to read working directory"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
