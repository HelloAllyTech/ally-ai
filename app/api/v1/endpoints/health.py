from fastapi import APIRouter

from app.schemas.health import HealthCheckResponse

router = APIRouter()


@router.get("", tags=["health"], response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify if the service is running.
    """
    return HealthCheckResponse(status="ok")
