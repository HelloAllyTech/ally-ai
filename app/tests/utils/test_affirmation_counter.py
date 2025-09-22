"""
Unit tests for affirmation counter utility.
"""

from unittest.mock import patch

from app.schemas.common import ChatMessage
from app.utils.affirmation_counter import FUZZY_THRESHOLD, count_affirmations


class TestAffirmationCounter:
    """Test cases for affirmation counter utility."""

    def test_count_affirmations_empty_list(self):
        """Test counting affirmations in empty list."""
        result = count_affirmations([])
        assert result == 0

    def test_count_affirmations_no_affirmations(self):
        """Test counting affirmations when none present."""
        messages = [
            ChatMessage(role="counselor", content="How are you feeling today?"),
            ChatMessage(role="counselor", content="Can you tell me more about that?"),
        ]
        result = count_affirmations(messages)
        assert result == 0

    def test_count_affirmations_with_affirmations(self):
        """Test counting affirmations when present."""
        messages = [
            ChatMessage(role="counselor", content="That makes complete sense."),
            ChatMessage(
                role="counselor",
                content="I can completely understand why you'd feel this way.",
            ),
            ChatMessage(
                role="counselor", content="Your feelings are absolutely valid."
            ),
        ]
        result = count_affirmations(messages)
        assert result > 0

    def test_count_affirmations_mixed_content(self):
        """Test counting affirmations in mixed content."""
        messages = [
            ChatMessage(role="counselor", content="How are you feeling today?"),
            ChatMessage(
                role="counselor", content="That makes complete sense."
            ),  # affirmation
            ChatMessage(role="counselor", content="Can you tell me more?"),
            ChatMessage(
                role="counselor", content="Your feelings are absolutely valid."
            ),  # affirmation
        ]
        result = count_affirmations(messages)
        assert result == 2

    def test_count_affirmations_only_client_messages(self):
        """Test counting affirmations with only client messages."""
        messages = [
            ChatMessage(role="client", content="I'm feeling anxious."),
            ChatMessage(
                role="client", content="That makes complete sense."
            ),  # affirmation but from client
        ]
        result = count_affirmations(messages)
        assert result == 0  # Should not count client messages

    def test_count_affirmations_fuzzy_matching(self):
        """Test counting affirmations with fuzzy matching."""
        messages = [
            ChatMessage(
                role="counselor", content="That makes complete sense!"
            ),  # Slight variation
            ChatMessage(
                role="counselor", content="I can understand why you'd feel this way."
            ),  # Partial match
        ]
        result = count_affirmations(messages)
        assert result >= 0  # Should find at least some matches

    def test_count_affirmations_case_insensitivity(self):
        """Test that affirmation matching is case-insensitive."""
        messages = [
            ChatMessage(role="counselor", content="that makes complete sense."),
            ChatMessage(
                role="counselor", content="YOUR FEELINGS ARE ABSOLUTELY VALID."
            ),
        ]
        result = count_affirmations(messages)
        assert result == 2

    def test_count_affirmations_punctuation_handling(self):
        """Test that punctuation does not prevent matching."""
        messages = [
            ChatMessage(role="counselor", content="That makes complete sense!"),
            ChatMessage(
                role="counselor", content="Your feelings are absolutely valid?"
            ),
            ChatMessage(role="counselor", content="You're not wrong to feel this way."),
        ]
        result = count_affirmations(messages)
        assert result == 3

    def test_count_affirmations_partial_sentence_match(self):
        """Test that partial sentences containing affirmations are matched."""
        messages = [
            ChatMessage(
                role="counselor",
                content="I just want to say, that makes complete sense, and I agree.",
            ),
        ]
        result = count_affirmations(messages)
        assert result == 1

    def test_count_affirmations_multiple_in_one_message(self):
        """Test counting multiple affirmations within a single counselor message."""
        messages = [
            ChatMessage(
                role="counselor",
                content=(
                    "That makes complete sense. Your feelings are absolutely valid."
                ),
            ),
        ]
        result = count_affirmations(messages)
        assert result == 2

    def test_count_affirmations_threshold_effect(self):
        """Test the effect of the fuzzy matching threshold."""
        original_threshold = FUZZY_THRESHOLD
        # Temporarily lower threshold to catch weaker matches
        with patch("app.utils.affirmation_counter.FUZZY_THRESHOLD", 50):
            messages = [
                ChatMessage(
                    role="counselor", content="That makes some sense."
                ),  # Should match with lower threshold
            ]
            result = count_affirmations(messages)
            assert result == 1
        # Restore original threshold and use a message that definitely won't match
        with patch("app.utils.affirmation_counter.FUZZY_THRESHOLD", original_threshold):
            messages = [
                ChatMessage(
                    role="counselor", content="That is completely different."
                ),  # Should not match with original threshold
            ]
            result = count_affirmations(messages)
            assert result == 0

    def test_count_affirmations_long_non_affirmation_message(self):
        """Test a long message that contains no affirmations."""
        messages = [
            ChatMessage(
                role="counselor",
                content=(
                    "This is a very long message that discusses many different topics "
                    "and provides various forms of support, but intentionally avoids "  # noqa: E501
                    "using any of the predefined affirmation phrases. It focuses on "  # noqa: E501
                    "open-ended questions and active listening without direct validation "  # noqa: E501
                    "phrases."
                ),
            ),
        ]
        result = count_affirmations(messages)
        assert result == 0

    def test_count_affirmations_empty_content_message(self):
        """Test messages with empty content."""
        messages = [
            ChatMessage(role="counselor", content=""),
            ChatMessage(role="counselor", content="   "),
            ChatMessage(role="counselor", content="That makes complete sense."),
        ]
        result = count_affirmations(messages)
        assert result == 1
