"""
Tests for app/core/text_generations/structured_output_models.py OpenAPI example
generation.
"""

from app.core.text_generations.structured_output_models import (
    StructuredDiarization,
    StructuredDiarizedMessage,
    StructuredSummaryNote,
)


class TestStructuredOutputModelSchemaExamples:
    """The nested `class ConfigDict` in these models is a Pydantic v2 no-op;
    only `model_config = ConfigDict(...)` is recognized, so the intended
    `example` must actually surface in the generated JSON schema."""

    def test_structured_summary_note_schema_has_example(self):
        schema = StructuredSummaryNote.model_json_schema()

        assert "example" in schema

    def test_structured_diarized_message_schema_has_example(self):
        schema = StructuredDiarizedMessage.model_json_schema()

        assert "example" in schema

    def test_structured_diarization_schema_has_example(self):
        schema = StructuredDiarization.model_json_schema()

        assert "example" in schema
