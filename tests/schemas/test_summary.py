"""
Tests for app/schemas/summary.py OpenAPI example generation.
"""

from app.schemas.summary import (
    ContentEnhance,
    ContentEnhanceRequest,
    ContentEnhanceResponse,
    DynamicSummaryNoteResponse,
    SummaryNoteAndTagsRequest,
    SummaryNoteAndTagsResponse,
    Tag,
    TagPositivityRatingRequest,
    TagPositivityRatingResponse,
)


class TestSummarySchemaExamples:
    """The nested `class ConfigDict` in these models is a Pydantic v2 no-op;
    only `model_config = ConfigDict(...)` is recognized, so the intended
    `example` must actually surface in the generated JSON schema."""

    def test_summary_note_and_tags_request_schema_has_example(self):
        schema = SummaryNoteAndTagsRequest.model_json_schema()

        assert "example" in schema

    def test_tag_schema_has_example(self):
        schema = Tag.model_json_schema()

        assert schema.get("example") == {"tag": "Stress", "positivity_rating": 2}

    def test_summary_note_and_tags_response_schema_has_example(self):
        schema = SummaryNoteAndTagsResponse.model_json_schema()

        assert "example" in schema

    def test_dynamic_summary_note_response_schema_has_example(self):
        schema = DynamicSummaryNoteResponse.model_json_schema()

        assert "example" in schema

    def test_content_enhance_request_schema_has_example(self):
        schema = ContentEnhanceRequest.model_json_schema()

        assert schema.get("example") == {
            "content": "Exam stress - pressure from parents."
        }

    def test_content_enhance_response_schema_has_example(self):
        schema = ContentEnhanceResponse.model_json_schema()

        assert "example" in schema

    def test_content_enhance_schema_has_example(self):
        schema = ContentEnhance.model_json_schema()

        assert "example" in schema

    def test_tag_positivity_rating_request_schema_has_example(self):
        schema = TagPositivityRatingRequest.model_json_schema()

        assert schema.get("example") == {
            "tags": ["Stress", "Anxiety", "Work-life balance"]
        }

    def test_tag_positivity_rating_response_schema_has_example(self):
        schema = TagPositivityRatingResponse.model_json_schema()

        assert "example" in schema
