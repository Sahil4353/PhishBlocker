# app/main.py
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.debug import router as debug_router

# Routers
from app.api.routes.pages import router as pages_router
from app.api.routes.scans import router as scans_router
from app.core.logging import get_logger
from app.core.middleware import RequestIdMiddleware
from app.db import engine
from app.models.base import Base  # ensures metadata is available

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Dev convenience: create tables if they don't exist
        Base.metadata.create_all(bind=engine)
        logger.info("DB tables ensured/created")
        yield
    except Exception as e:
        logger.exception("App lifespan error: %s", e)
        # Nothing special to return here; FastAPI will surface startup error.
        raise
    finally:
        # Place shutdown cleanup here if needed later (e.g., closing pools)
        logger.info("App shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(title="PhishBlocker", version="0.1.0", lifespan=lifespan)

    app.add_middleware(RequestIdMiddleware)

    # Mount routers
    app.include_router(pages_router)
    app.include_router(scans_router)
    app.include_router(debug_router)

    logger.info("Routers registered: pages, scans, debug")
    return app


# ASGI entrypoint for uvicorn
app = create_app()
