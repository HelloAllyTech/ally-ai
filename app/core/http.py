# app/core/http.py

import httpx
from fastapi import FastAPI

from app.core.config import settings

ally_core_http_client: httpx.AsyncClient | None = None


def register_http_clients(app: FastAPI) -> None:
    @app.on_event("startup")
    async def startup() -> None:
        global ally_core_http_client
        ally_core_http_client = httpx.AsyncClient(
            base_url=settings.ALLY_CORE.ENDPOINT,
            timeout=httpx.Timeout(
                connect=3.0,
                read=5.0,
                write=5.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )

    @app.on_event("shutdown")
    async def shutdown() -> None:
        if ally_core_http_client:
            await ally_core_http_client.aclose()
