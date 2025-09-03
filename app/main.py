# app/main.py
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.routes.debug import router as debug_router
from app.api.routes.pages import router as pages_router
from app.api.routes.scans import router as scans_router
from app.core.config import settings
from app.core.logging import (
    attach_filters_to,  # helper to add our filters to uvicorn loggers
    get_logger,
    setup_logging,
)
from app.core.middleware import RequestIdMiddleware
from app.db import engine
from app.models.base import Base  # ensures metadata is available

# ⬇️ ML model service
try:
    from app.services.model import ModelService  # your loader
except Exception:  # pragma: no cover
    ModelService = None  # type: ignore

# ---- logging: init once, and propagate filters to uvicorn loggers
setup_logging()
for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    attach_filters_to(logging.getLogger(name))

logger = get_logger(__name__)


def _resolve_model_path() -> Path:
    """
    Best-effort resolution of model artifact path from settings, with a sensible default.
    Supports any of: MODEL_PATH, MODEL_ARTIFACT, MODEL_FILE in settings.
    """
    candidates = []
    for attr in ("MODEL_PATH", "MODEL_ARTIFACT", "MODEL_FILE"):
        if hasattr(settings, attr):
            val = getattr(settings, attr)
            if val:
                candidates.append(Path(str(val)))
    # default fallback
    candidates.append(Path("models/tfidf_lr_small_l2.joblib"))
    for p in candidates:
        if p and Path(p).exists():
            return Path(p)
    # Return the first candidate even if it doesn't exist; loader will warn.
    return candidates[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Dev convenience: create tables if they don't exist (skip in prod)
        if settings.ENV.lower() != "prod":
            Base.metadata.create_all(bind=engine)
            logger.info("DB tables ensured/created (env=%s)", settings.ENV)
        else:
            logger.info("DB migration-managed (env=prod); skipping create_all()")

        # ---- Load ML model (optional, heuristics still work if it fails)
        app.state.model = None
        model_path = _resolve_model_path()
        if ModelService is None:
            logger.warning(
                "ModelService not importable; ML disabled (heuristics only)."
            )
        else:
            try:
                app.state.model = ModelService(str(model_path))
                # Optional: attach a version string for scans
                setattr(
                    app.state.model,
                    "version",
                    getattr(app.state.model, "_artifact_path", None),
                )
                logger.info("ML model loaded: %s", model_path)
            except Exception as e:
                logger.warning(
                    "Failed to load ML model at %s; continuing without ML. err=%s",
                    model_path,
                    e,
                )

        yield
    except Exception as e:
        logger.exception("App lifespan error: %s", e)
        raise
    finally:
        logger.info("App shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(title="PhishBlocker", version="0.1.0", lifespan=lifespan)

    # Middleware
    app.add_middleware(RequestIdMiddleware)

    # Routers
    app.include_router(pages_router)
    app.include_router(scans_router)
    app.include_router(debug_router)

    logger.info("App ready | env=%s | routers=pages,scans,debug", settings.ENV)
    return app


# ASGI entrypoint for uvicorn
app = create_app()
