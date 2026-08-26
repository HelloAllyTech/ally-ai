"""
Tests for app/schemas/health.py OpenAPI example generation.
"""

from app.schemas.health import HealthCheckResponse, ReadinessCheckResponse


class TestHealthSchemaExamples:
    """The nested `class ConfigDict` in these models is a Pydantic v2 no-op;
    only `model_config = ConfigDict(...)` is recognized, so the intended
    `example` must actually surface in the generated JSON schema."""

    def test_health_check_response_schema_has_example(self):
        schema = HealthCheckResponse.model_json_schema()

        assert schema.get("example") == {"status": "ok"}

    def test_readiness_check_response_schema_has_example(self):
        schema = ReadinessCheckResponse.model_json_schema()

        assert schema.get("example") == {
            "status": "ok",
            "dependencies": {"weaviate": "ok"},
        }
