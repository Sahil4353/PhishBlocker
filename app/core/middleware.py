# app/core/middleware.py
from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.logging import clear_request_id, get_logger, set_request_id

logger = get_logger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Adds/propagates a request ID for each incoming request.

    - Reads X-Request-ID if provided, else generates uuid4 hex
    - Exposes it to logs via logging filter (see core/logging.py)
    - Writes X-Request-ID on the response
    - Logs method, path, status, duration (ms)
    """

    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        rid = request.headers.get(self.header_name) or uuid.uuid4().hex
        set_request_id(rid)

        start = time.perf_counter()
        response: Optional[Response] = None
        try:
            logger.debug(
                "request start",
                extra={"method": request.method, "path": request.url.path},
            )
            response = await call_next(request)
            return response
        except Exception as e:
            # Make sure exceptions are logged with the same rid
            logger.exception(
                "unhandled error during request: %s %s -> %s",
                request.method,
                request.url.path,
                e,
            )
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            status_code = response.status_code if response is not None else "ERR"
            logger.info(
                "request complete",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                },
            )
            # Always set the header even on errors (if a response exists)
            if response is not None:
                response.headers[self.header_name] = rid
            clear_request_id()
