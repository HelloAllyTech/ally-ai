"""
Unit tests for structured model converter utility.
"""

import pytest

from app.core.text_generations.structured_output_models import StructuredTag
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
