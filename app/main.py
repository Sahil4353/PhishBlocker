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
    attach_filters_to,
    get_logger,
    setup_logging,
)
from app.core.middleware import RequestIdMiddleware
from app.db import engine
from app.models.base import Base

# ML model service (optional fallback to heuristics if import/load fails)
try:
    from app.services.model import ModelService
except Exception:  # pragma: no cover
    ModelService = None  # type: ignore

# ---- logging: init once, and propagate filters to uvicorn loggers
setup_logging()
for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    attach_filters_to(logging.getLogger(name))

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # --- DB init (dev-only convenience)
        if settings.ENV.lower() != "prod":
            Base.metadata.create_all(bind=engine)
            logger.info("DB tables ensured/created (env=%s)", settings.ENV)
        else:
            logger.info("DB migration-managed (env=prod); skipping create_all()")

        # --- ML model load
        app.state.model = None
        if ModelService is None:
            logger.warning("ModelService not importable; ML disabled (heuristics only).")
        else:
            model_path = Path(settings.MODEL_PATH)
            if not model_path.exists():
                logger.warning(
                    "Configured MODEL_PATH does not exist: %s ; "
                    "continuing without ML (heuristics only).",
                    model_path,
                )
            else:
                try:
                    app.state.model = ModelService(
                        artifact_path=str(model_path),
                        version=(settings.MODEL_VERSION or None),
                    )
                    logger.info(
                        "ML model loaded",
                        extra={
                            "artifact": str(model_path),
                            "model_version": getattr(app.state.model, "version", None),
                            "env": settings.ENV,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to load ML model at %s; "
                        "continuing without ML. err=%s",
                        model_path,
                        e,
                    )

        # hand control back to FastAPI
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
