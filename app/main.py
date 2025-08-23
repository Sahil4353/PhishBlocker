# app/main.py
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

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

# ---- logging: init once, and propagate filters to uvicorn loggers
setup_logging()
for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    attach_filters_to(logging.getLogger(name))

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Dev convenience: create tables if they don't exist (skip in prod)
        if settings.ENV.lower() != "prod":
            Base.metadata.create_all(bind=engine)
            logger.info("DB tables ensured/created (env=%s)", settings.ENV)
        else:
            logger.info("DB migration-managed (env=prod); skipping create_all()")
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
