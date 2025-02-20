from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Indicates the health status of the service.")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok"
            }
        }
