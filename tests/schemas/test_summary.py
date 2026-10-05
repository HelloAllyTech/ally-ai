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


class TestScenarioEvaluationUsageTask:
    """The closed set of llm_usage labels a /scenario/evaluate call may use."""

    def test_every_allowed_label_is_a_known_task(self):
        """Each value must exist in LLMTask under the SAME string (and, by the
        same contract, in ally-be's LlmTask), or the row lands unlabelled."""
        from typing import get_args

        from app.core.llm_usage.tasks import LLMTask
        from app.schemas.summary import ScenarioEvaluationUsageTask

        allowed = set(get_args(ScenarioEvaluationUsageTask))
        assert allowed == {"scenario_evaluation", "scenario_evaluation_language"}
        assert allowed <= {t.value for t in LLMTask}

    def test_language_label_matches_the_enum_exactly(self):
        from app.core.llm_usage.tasks import LLMTask

        assert (
            LLMTask.SCENARIO_EVALUATION_LANGUAGE.value == "scenario_evaluation_language"
        )

    def test_defaults_and_null_resolve_to_scenario_evaluation(self):
        from app.schemas.summary import ScenarioEvaluationRequest

        assert (
            ScenarioEvaluationRequest(chat_history=[]).usage_task
            == "scenario_evaluation"
        )
        assert (
            ScenarioEvaluationRequest(chat_history=[], usage_task=None).usage_task
            == "scenario_evaluation"
        )

    def test_rejects_any_other_label(self):
        import pytest
        from pydantic import ValidationError

        from app.schemas.summary import ScenarioEvaluationRequest

        with pytest.raises(ValidationError):
            ScenarioEvaluationRequest(chat_history=[], usage_task="nudge")
