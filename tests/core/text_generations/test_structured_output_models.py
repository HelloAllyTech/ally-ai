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


class TestStructuredSummaryNoteValidators:
    """The StructuredSummaryNote model should clamp out-of-range values."""

    def test_call_quality_clamps_above_100(self):
        """A call_quality > 100 should be clamped to 100."""
        note = StructuredSummaryNote(
            tags=[{"tag": "test", "positivity_rating": 3}], call_quality=150
        )
        assert note.call_quality == 100

    def test_call_quality_clamps_below_0(self):
        """A call_quality < 0 should be clamped to 0."""
        note = StructuredSummaryNote(
            tags=[{"tag": "test", "positivity_rating": 3}], call_quality=-10
        )
        assert note.call_quality == 0

    def test_call_quality_allows_in_range(self):
        """A call_quality between 0 and 100 should not be changed."""
        note = StructuredSummaryNote(
            tags=[{"tag": "test", "positivity_rating": 3}], call_quality=50
        )
        assert note.call_quality == 50
