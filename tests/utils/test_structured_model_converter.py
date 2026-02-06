"""
Unit tests for structured model converter utility.
"""

import pytest

from app.core.text_generations.structured_output_models import (
    StructuredSummaryNote,
    StructuredTag,
)
from app.schemas.summary import Tag
from app.utils.structured_model_converter import (
    fetch_levels_of_feeling_from_dominant_feelings,
    format_dominant_feeling,
    format_list_to_bullet_points,
    structured_output_model_to_rest,
)


class TestFetchLevelsOfFeelingFromDominantFeelings:
    """Test cases for fetch_levels_of_feeling_from_dominant_feelings function."""

    def test_valid_dominant_feeling(self):
        """Test with valid dominant feeling string."""
        result = fetch_levels_of_feeling_from_dominant_feelings("Hurt:Depressed")
        assert result == ("Sad", "Depressed", "Hurt")

    def test_valid_dominant_feeling_with_spaces(self):
        """Test with dominant feeling string containing spaces."""
        result = fetch_levels_of_feeling_from_dominant_feelings(
            " Let Down : Frustrated "
        )
        assert result == ("Angry", "Frustrated", "Let Down")

    def test_none_input(self):
        """Test with None input."""
        result = fetch_levels_of_feeling_from_dominant_feelings(None)
        assert result == (None, None, None)

    def test_empty_string(self):
        """Test with empty string."""
        # Empty string is falsy, so it returns (None, None, None) like None input
        result = fetch_levels_of_feeling_from_dominant_feelings("")
        assert result == (None, None, None)

    def test_invalid_format_no_colon(self):
        """Test with string that doesn't contain colon."""
        with pytest.raises(
            ValueError, match="Input string must be in the format 'level3:level2'"
        ):
            fetch_levels_of_feeling_from_dominant_feelings("InvalidFormat")

    def test_invalid_format_multiple_colons(self):
        """Test with string containing multiple colons."""
        # Multiple colons will be split on first colon, so "Level2:Level3"
        # which is not recognized, so it raises a different error
        with pytest.raises(
            ValueError,
            match="Level2 value 'Level2:Level3' is not recognized",
        ):
            fetch_levels_of_feeling_from_dominant_feelings("Level1:Level2:Level3")

    def test_unknown_level2(self):
        """Test with unknown level2 feeling."""
        with pytest.raises(
            ValueError, match="Level2 value 'Unknown' is not recognized"
        ):
            fetch_levels_of_feeling_from_dominant_feelings("Inner:Unknown")


class TestFormatDominantFeeling:
    """Test cases for format_dominant_feeling function."""

    def test_valid_levels(self):
        """Test with valid levels."""
        result = format_dominant_feeling("Sad", "Depressed", "Hurt")
        assert result == "Sad > Depressed > Hurt"

    def test_none_level1(self):
        """Test with None level1."""
        result = format_dominant_feeling(None, "Depressed", "Hurt")
        assert result is None

    def test_none_level2(self):
        """Test with None level2."""
        result = format_dominant_feeling("Sad", None, "Hurt")
        assert result is None

    def test_none_level3(self):
        """Test with None level3."""
        result = format_dominant_feeling("Sad", "Depressed", None)
        assert result is None

    def test_all_none(self):
        """Test with all None levels."""
        result = format_dominant_feeling(None, None, None)
        assert result is None


class TestFormatListToBulletPoints:
    """Test cases for format_list_to_bullet_points function."""

    def test_empty_list(self):
        """Test with empty list."""
        result = format_list_to_bullet_points([])
        assert result == ""

    def test_single_item(self):
        """Test with single item."""
        result = format_list_to_bullet_points(["Item 1"])
        assert result == "- Item 1"

    def test_multiple_items(self):
        """Test with multiple items."""
        result = format_list_to_bullet_points(["Item 1", "Item 2", "Item 3"])
        expected = "- Item 1\n- Item 2\n- Item 3"
        assert result == expected

    def test_items_with_special_characters(self):
        """Test with items containing special characters."""
        result = format_list_to_bullet_points(["Item-1", "Item_2", "Item.3"])
        expected = "- Item-1\n- Item_2\n- Item.3"
        assert result == expected


class TestStructuredOutputModelToRest:
    """Test cases for structured_output_model_to_rest function."""

    def test_none_input(self):
        """Test with None input."""
        result = structured_output_model_to_rest(None)
        assert result is None

    def test_structured_tag_conversion(self):
        """Test conversion of StructuredTag to Tag."""
        structured_tag = StructuredTag(tag="Test Tag", positivity_rating=5)
        result = structured_output_model_to_rest(structured_tag)

        assert isinstance(result, Tag)
        assert result.tag == "Test Tag"
        assert result.positivity_rating == 5

    def test_unsupported_type(self):
        """Test with unsupported type."""
        with pytest.raises(
            NotImplementedError,
            match="Conversion for type <class 'str'> is not implemented",
        ):
            structured_output_model_to_rest("unsupported_type")

    def test_structured_summary_note_full_conversion(self):
        """Ensure full StructuredSummaryNote converts correctly to SummaryNoteAndTagsResponse."""  # noqa: E501
        sop = StructuredSummaryNote(
            call_id="CALL123",
            call_duration=1800,
            date_of_session="2025-10-04",
            call_time="10:00 AM",
            client_id="CL123",
            counsellor="Alex",
            call_type="Follow-up",
            age="25-34",
            gender="Female",
            profession="Engineer",
            relationship_status="Single",
            location="Mumbai",
            code_of_concern="Work-life Concerns",
            session_summary=["Summary..."],
            counseling_process_flow=["Phase A", "Phase B"],
            key_concerns=["Concern 1", "Concern 2"],
            subjective_observations=["Obs 1"],
            objective_observations=["Obj 1"],
            assessment="Assessment...",
            dominant_feelings=["Embarrassed:Hurt"],  # level2=Hurt -> level1=Sad
            issues_worked_on=["Issue 1"],
            key_therapeutic_techniques=["Technique 1"],
            referrals_provided=["Referral 1"],
            homework=["Task 1"],
            plan_for_next_call=["Plan 1"],
            tags=[StructuredTag(tag="Stress", positivity_rating=3)],
            emotional_lift="Lift",
            call_quality=85,
        )

        resp = structured_output_model_to_rest(sop)

        # Basic fields
        assert resp.call_id == "CALL123"
        assert resp.call_duration == 1800
        assert resp.call_date == "2025-10-04"
        assert resp.call_time == "10:00 AM"
        assert resp.client_id == "CL123"
        assert resp.counsellor == "Alex"
        assert resp.call_type == "Follow-up"
        assert resp.age == "25-34"
        assert resp.gender == "Female"
        assert resp.profession == "Engineer"
        assert resp.relationship_status == "Single"
        assert resp.location == "Mumbai"
        assert resp.code_of_concern == "Work-life Concerns"
        assert resp.session_summary == "- Summary..."

        # Bullet list formatting
        assert resp.counseling_process_flow == "- Phase A\n- Phase B"
        assert resp.key_concerns == "- Concern 1\n- Concern 2"
        assert resp.subjective_observations == "- Obs 1"
        assert resp.objective_observations == "- Obj 1"
        assert resp.issues_worked_on == "- Issue 1"
        assert resp.key_therapeutic_techniques == "- Technique 1"
        assert resp.referrals_provided == "- Referral 1"
        assert resp.homework == "- Task 1"
        assert resp.plan_for_next_call == "- Plan 1"

        # Dominant feelings formatting: Sad > Hurt > Embarrassed
        assert resp.dominant_feelings == "- Sad > Hurt > Embarrassed"

        # Tags conversion
        assert len(resp.tags) == 1
        assert isinstance(resp.tags[0], Tag)
        assert resp.tags[0].tag == "Stress"
        assert resp.tags[0].positivity_rating == 3

        # Other fields
        assert resp.emotional_lift == "Lift"
        assert resp.call_quality == 85
        assert resp.listening_share is None  # fixed to None in converter

    def test_structured_summary_note_handles_none_and_empty(self):
        """Lists that are None should yield None, dominant_feelings None -> empty string."""  # noqa: E501
        sop = StructuredSummaryNote(
            call_id=None,
            call_duration=None,
            date_of_session=None,
            call_time=None,
            client_id=None,
            counsellor=None,
            call_type=None,
            age=None,
            gender=None,
            profession=None,
            relationship_status=None,
            location=None,
            code_of_concern=None,
            session_summary=None,
            counseling_process_flow=None,
            key_concerns=None,
            subjective_observations=None,
            objective_observations=None,
            assessment=None,
            dominant_feelings=None,  # leads to empty string after formatting
            issues_worked_on=None,
            key_therapeutic_techniques=None,
            referrals_provided=None,
            homework=None,
            plan_for_next_call=None,
            tags=[StructuredTag(tag="T", positivity_rating=1)],
            emotional_lift=None,
            call_quality=50,
        )

        resp = structured_output_model_to_rest(sop)

        # Fields that are conditionally formatted should be None
        assert resp.counseling_process_flow is None
        assert resp.key_concerns is None
        assert resp.subjective_observations is None
        assert resp.objective_observations is None
        assert resp.issues_worked_on is None
        assert resp.key_therapeutic_techniques is None
        assert resp.referrals_provided is None
        assert resp.homework is None
        assert resp.plan_for_next_call is None

        # Dominant feelings becomes empty string when source is None
        assert resp.dominant_feelings == ""

    @pytest.mark.parametrize("input_quality,expected", [(150, 100), (-10, 0), (42, 42)])
    def test_call_quality_clamped(self, input_quality, expected):
        sop = StructuredSummaryNote(
            tags=[StructuredTag(tag="X", positivity_rating=2)],
            call_quality=input_quality,
        )
        resp = structured_output_model_to_rest(sop)
        assert resp.call_quality == expected
