from fastapi import APIRouter

from app.schemas.health import HealthCheckResponse
from app.utils.logger import get_logger

router = APIRouter()
health_logger = get_logger(__name__)

@router.get("", tags=["health"], response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify if the service is running.
    """
    health_logger.debug("Health check endpoint was called.")

    return HealthCheckResponse(status="ok")