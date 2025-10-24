import hmac
import uuid
from typing import List

from fastapi import Request
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import settings
from app.core.constants import APISettings
from app.utils.logger import logger, trace_id_var


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware for checking X-API-Key in request headers."""

    async def dispatch(self, request: Request, call_next):

        # Only protect /api/v1 routes
        if request.url.path.startswith(APISettings.API_V1_STR):
            api_key = request.headers.get(APISettings.X_API_KEY_HEADER)
            if not api_key or not hmac.compare_digest(api_key, settings.API.X_API_KEY):
                return JSONResponse(
                    {"detail": "Unauthorized - invalid or missing API key"},
                    status_code=401,
                )

        return await call_next(request)


class LogRequestMiddleware(BaseHTTPMiddleware):
    """Custom middleware for logging requests and adding a Trace ID."""

    async def dispatch(self, request: Request, call_next):
        trace_id = str(uuid.uuid4())
        trace_id_var.set(trace_id)

        if request.url.path != "/api/health":
            logger.info(
                f"Incoming request {request.method} {request.url} (TraceID: {trace_id})"
            )

        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id  # Include Trace ID in the response

        if request.url.path != "/api/health":
            logger.info(
                f"Completed request {request.method} {request.url} with status "
                f"{response.status_code} (TraceID: {trace_id})"
            )

        return response


def get_middlewares() -> List[Middleware]:
    """
    Create and configure middleware for the application.
    """
    middlewares = [
        Middleware(LogRequestMiddleware),
        # CORS middleware that should only be called from other services
        Middleware(
            CORSMiddleware,
            allow_origins=[],  # Empty list = reject all origins
            allow_credentials=False,  # Disable credentials
            allow_methods=[],  # Empty list = reject all methods
            allow_headers=[],  # Empty list = reject all headers
        ),
        Middleware(AuthMiddleware),
    ]
    return middlewares
