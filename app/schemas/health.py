from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Indicates the health status of the service.")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "ok"
            }
        }
