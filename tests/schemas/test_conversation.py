"""
Tests for app/schemas/conversation.py OpenAPI example generation.
"""

from app.schemas.conversation import (
    AnalyzeRequest,
    AnalyzeResponse,
    IdentifyRequest,
    IdentifyResponse,
    Nudge,
)


class TestConversationSchemaExamples:
    """The nested `class ConfigDict` in these models is a Pydantic v2 no-op;
    only `model_config = ConfigDict(...)` is recognized, so the intended
    `example` must actually surface in the generated JSON schema."""

    def test_nudge_schema_has_example(self):
        schema = Nudge.model_json_schema()

        assert schema.get("example") == {
            "nudge": "### Be empathetic and ask open-ended questions"
        }

    def test_analyze_request_schema_has_example(self):
        schema = AnalyzeRequest.model_json_schema()

        assert "example" in schema

    def test_analyze_response_schema_has_example(self):
        schema = AnalyzeResponse.model_json_schema()

        assert schema.get("example") == {
            "nudge": "### Be empathetic and ask open-ended questions",
            "stage": "Rapport Building",
        }

    def test_identify_request_schema_has_example(self):
        schema = IdentifyRequest.model_json_schema()

        assert "example" in schema

    def test_identify_response_schema_has_example(self):
        schema = IdentifyResponse.model_json_schema()

        assert schema.get("example") == {"speaker0": "client", "speaker1": "counselor"}
