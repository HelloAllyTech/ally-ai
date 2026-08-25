from typing import Dict

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Indicates the health status of the service.")

    class ConfigDict:
        json_schema_extra = {"example": {"status": "ok"}}


class ReadinessCheckResponse(BaseModel):
    """Dependency health, as opposed to `/health`'s bare liveness check.

    `status` is "ok" only when every checked dependency is. Kept a plain string
    rather than a bool so a future partial-degradation case (e.g. one of several
    providers down) has somewhere to go without a breaking response shape change.
    """

    status: str = Field(
        ...,
        description=(
            "'ok' when every dependency below is reachable, else 'degraded'."
        ),
    )
    dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-dependency status ('ok' or 'unavailable'), e.g. {'weaviate': 'ok'}."
        ),
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {"status": "ok", "dependencies": {"weaviate": "ok"}}
        }
