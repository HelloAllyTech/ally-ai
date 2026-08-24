import asyncio

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.vector_db.weaviate_client import WeaviateClient
from app.schemas.health import HealthCheckResponse, ReadinessCheckResponse
from app.utils.logger import get_logger

router = APIRouter()
health_logger = get_logger(__name__)

# Readiness is polled, so keep the dependency check itself cheap: a stuck
# Weaviate must resolve this endpoint quickly, not hang the poller.
WEAVIATE_READY_TIMEOUT_SECONDS = 2.0


@router.get("/health", tags=["health"], response_model=HealthCheckResponse)
async def health_check():
    """
    Liveness probe: is the process up and able to serve requests at all.

    Deliberately checks nothing downstream. An orchestrator polls this to decide
    whether to kill and restart the container — a slow or unreachable Weaviate
    must not get a perfectly healthy process killed for an outage elsewhere.
    Dependency health lives at `/health/ready`.
    """
    health_logger.debug("Health check called")

    return HealthCheckResponse(status="ok")


@router.get(
    "/health/ready",
    tags=["health"],
    response_model=ReadinessCheckResponse,
)
async def readiness_check():
    """
    Readiness probe: can this instance actually serve retrieval/generation traffic.

    Checks Weaviate connectivity (retrieval's hard dependency) with a short
    timeout so the check stays cheap under polling. Returns 503 with a
    "degraded" body when Weaviate is unreachable, so an orchestrator or load
    balancer can take the instance out of rotation instead of routing it
    traffic it cannot serve — see `/health` for the separate liveness probe
    that must stay unconditional.

    Does not call an LLM provider: there is no cheap, side-effect-free call
    that verifies one without spending a real generation request on every poll.
    """
    weaviate_ready = False
    try:
        client = WeaviateClient.get_client()
        weaviate_ready = bool(
            await asyncio.wait_for(
                client.is_ready(), timeout=WEAVIATE_READY_TIMEOUT_SECONDS
            )
        )
    except Exception as e:
        health_logger.warning(f"Readiness check: Weaviate is not ready: {e}")

    payload = ReadinessCheckResponse(
        status="ok" if weaviate_ready else "degraded",
        dependencies={"weaviate": "ok" if weaviate_ready else "unavailable"},
    )
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if weaviate_ready
            else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content=payload.model_dump(),
    )
