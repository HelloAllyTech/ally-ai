"""
Unit tests for silence calculator utility.
"""

from app.schemas.common import ChatMessage
from app.utils.silence_calculator import calculate_silence_by_counselor


class TestSilenceCalculator:
    """Test cases for silence calculator utility."""

    def test_calculate_silence_by_counselor_empty_list(self):
        """Test calculating silence with empty message list."""
        result = calculate_silence_by_counselor([])
        assert result == 0

    def test_calculate_silence_by_counselor_single_message(self):
        """Test calculating silence with only one message."""
        messages = [
            ChatMessage(role="client", content="Hello", start_time=0.0, end_time=2.0),
        ]
        result = calculate_silence_by_counselor(messages)
        assert result == 0

    def test_calculate_silence_by_counselor_no_timing_data(self):
        """Test calculating silence with messages that have no timing data."""
        messages = [
            ChatMessage(role="client", content="Hello"),
            ChatMessage(role="counselor", content="Hi there"),
        ]
        result = calculate_silence_by_counselor(messages)
        assert result == 0

    def test_calculate_silence_by_counselor_client_to_counselor_silence(self):
        """Test calculating silence between client and counselor messages."""
        messages = [
            ChatMessage(
                role="client",
                content="I'm feeling anxious",
                start_time=0.0,
                end_time=3.0,
            ),
            ChatMessage(
                role="counselor", content="I understand", start_time=8.0, end_time=10.0
            ),  # 5s silence
        ]
        result = calculate_silence_by_counselor(messages)
        assert result == 1

    def test_calculate_silence_by_counselor_client_to_client_silence(self):
        """Test calculating silence between two client messages."""
        messages = [
            ChatMessage(
                role="client", content="First message", start_time=0.0, end_time=2.0
            ),
            ChatMessage(
                role="client", content="Second message", start_time=7.0, end_time=9.0
            ),  # 5s silence
        ]
        result = calculate_silence_by_counselor(messages)
        assert result == 1

    def test_calculate_silence_by_counselor_short_silence_ignored(self):
        """Test that short silences (less than MIN_SILENCE_DURATION) are ignored."""
        messages = [
            ChatMessage(role="client", content="First", start_time=0.0, end_time=2.0),
            ChatMessage(
                role="counselor", content="Second", start_time=4.0, end_time=6.0
            ),  # 2s silence (too short)
        ]
        result = calculate_silence_by_counselor(messages)
        assert result == 0

    def test_calculate_silence_by_counselor_exact_minimum_silence(self):
        """Test silence calculation with exactly MIN_SILENCE_DURATION."""
        messages = [
            ChatMessage(role="client", content="First", start_time=0.0, end_time=2.0),
            ChatMessage(
                role="counselor", content="Second", start_time=5.0, end_time=7.0
            ),  # 3s silence (exactly minimum)
        ]
        result = calculate_silence_by_counselor(messages)
        assert result == 1

    def test_calculate_silence_by_counselor_multiple_silences(self):
        """Test calculating multiple silence periods."""
        messages = [
            ChatMessage(role="client", content="First", start_time=0.0, end_time=2.0),
            ChatMessage(
                role="counselor", content="Second", start_time=6.0, end_time=8.0
            ),  # 4s silence
            ChatMessage(role="client", content="Third", start_time=8.0, end_time=10.0),
            ChatMessage(
                role="client", content="Fourth", start_time=15.0, end_time=17.0
            ),  # 5s silence
        ]
        result = calculate_silence_by_counselor(messages)
        assert result == 2

    def test_calculate_silence_by_counselor_mixed_roles_no_silence(self):
        """Test with mixed roles but no qualifying silence periods."""
        messages = [
            ChatMessage(
                role="counselor", content="First", start_time=0.0, end_time=2.0
            ),
            ChatMessage(
                role="client", content="Second", start_time=2.5, end_time=4.5
            ),  # 0.5s silence (too short)
            ChatMessage(
                role="counselor", content="Third", start_time=5.0, end_time=7.0
            ),  # 0.5s silence (too short)
        ]
        result = calculate_silence_by_counselor(messages)
        assert result == 0

    def test_calculate_silence_by_counselor_counselor_to_client_no_silence(self):
        """Test that counselor-to-client transitions don't count as silence."""
        messages = [
            ChatMessage(
                role="counselor", content="First", start_time=0.0, end_time=2.0
            ),
            ChatMessage(
                role="client", content="Second", start_time=8.0, end_time=10.0
            ),  # 6s gap but counselor->client
        ]
        result = calculate_silence_by_counselor(messages)
        assert result == 0

    def test_calculate_silence_by_counselor_unsorted_messages(self):
        """Test that messages are properly sorted by start_time."""
        messages = [
            ChatMessage(role="client", content="Second", start_time=5.0, end_time=7.0),
            ChatMessage(role="client", content="First", start_time=0.0, end_time=2.0),
        ]
        result = calculate_silence_by_counselor(messages)
        # 3-second silence between client messages (5.0 - 2.0 = 3.0 >= MIN_SILENCE)
        assert result == 1

    def test_calculate_silence_by_counselor_missing_timing_fields(self):
        """Test with messages that have some timing fields missing."""
        messages = [
            ChatMessage(role="client", content="Valid", start_time=0.0, end_time=2.0),
            ChatMessage(
                role="client", content="No start", start_time=None, end_time=5.0
            ),
            ChatMessage(
                role="counselor", content="Valid", start_time=8.0, end_time=10.0
            ),
        ]
        result = calculate_silence_by_counselor(messages)
        # Should only process messages with both start_time and end_time
        assert result == 1

    def test_calculate_silence_by_counselor_edge_case_timing(self):
        """Test edge cases with timing boundaries."""
        messages = [
            ChatMessage(role="client", content="First", start_time=0.0, end_time=2.0),
            ChatMessage(
                role="counselor", content="Second", start_time=2.0, end_time=4.0
            ),  # No silence (end_time = start_time)
        ]
        result = calculate_silence_by_counselor(messages)
        assert result == 0
