"""
Unit tests for utterance duration calculator utility.
"""

from app.schemas.common import ChatMessage
from app.utils.utterance_duration_calculator import (
    calculate_avg_client_utterance_duration,
)


class TestUtteranceDurationCalculator:
    """Test cases for utterance duration calculator utility."""

    def test_calculate_avg_client_utterance_duration_empty_list(self):
        """Test calculating average duration with empty message list."""
        result = calculate_avg_client_utterance_duration([])
        assert result is None

    def test_calculate_avg_client_utterance_duration_no_client_messages(self):
        """Test calculating average duration with no client messages."""
        messages = [
            ChatMessage(
                role="counselor", content="How are you?", start_time=0.0, end_time=2.0
            ),
            ChatMessage(
                role="system", content="Session started", start_time=0.0, end_time=1.0
            ),
        ]
        result = calculate_avg_client_utterance_duration(messages)
        assert result is None

    def test_calculate_avg_client_utterance_duration_no_timing_data(self):
        """Test calculating average duration with client messages but no timing data."""
        messages = [
            ChatMessage(role="client", content="I'm feeling anxious."),
            ChatMessage(role="client", content="Work is stressful."),
        ]
        result = calculate_avg_client_utterance_duration(messages)
        assert result is None

    def test_calculate_avg_client_utterance_duration_single_client_message(self):
        """Test calculating average duration with a single client message."""
        messages = [
            ChatMessage(role="client", content="Hello", start_time=0.0, end_time=3.5),
        ]
        result = calculate_avg_client_utterance_duration(messages)
        assert result == 3.5

    def test_calculate_avg_client_utterance_duration_multiple_client_messages(self):
        """Test calculating average duration with multiple client messages."""
        messages = [
            ChatMessage(role="client", content="Hello", start_time=0.0, end_time=2.0),
            ChatMessage(
                role="client", content="How are you?", start_time=5.0, end_time=8.0
            ),
            ChatMessage(
                role="client", content="I'm fine", start_time=10.0, end_time=12.0
            ),
        ]
        result = calculate_avg_client_utterance_duration(messages)
        # Average of 2.0, 3.0, 2.0 = 7.0/3 = 2.33... rounded to 2.33
        assert result == 2.33

    def test_calculate_avg_client_utterance_duration_mixed_roles(self):
        """Test calculating average duration with mixed roles."""
        messages = [
            ChatMessage(
                role="counselor", content="Hello", start_time=0.0, end_time=1.0
            ),
            ChatMessage(
                role="client", content="Hi there", start_time=2.0, end_time=5.0
            ),
            ChatMessage(
                role="counselor", content="How are you?", start_time=6.0, end_time=8.0
            ),
            ChatMessage(
                role="client", content="I'm good", start_time=9.0, end_time=11.0
            ),
        ]
        result = calculate_avg_client_utterance_duration(messages)
        # Average of 3.0, 2.0 = 5.0/2 = 2.5
        assert result == 2.5

    def test_calculate_avg_client_utterance_duration_invalid_timing(self):
        """Test calculating average duration with invalid timing data."""
        messages = [
            ChatMessage(role="client", content="Valid", start_time=0.0, end_time=2.0),
            ChatMessage(
                role="client", content="Invalid", start_time=5.0, end_time=3.0
            ),  # end < start
            ChatMessage(role="client", content="Valid", start_time=6.0, end_time=8.0),
        ]
        result = calculate_avg_client_utterance_duration(messages)
        # Should only count valid messages: 2.0, 2.0 = 4.0/2 = 2.0
        assert result == 2.0

    def test_calculate_avg_client_utterance_duration_missing_timing(self):
        """Test calculating average duration with some messages missing timing data."""
        messages = [
            ChatMessage(
                role="client", content="With timing", start_time=0.0, end_time=2.0
            ),
            ChatMessage(role="client", content="No timing"),
            ChatMessage(
                role="client", content="With timing", start_time=5.0, end_time=7.0
            ),
        ]
        result = calculate_avg_client_utterance_duration(messages)
        # Should only count messages with timing: 2.0, 2.0 = 4.0/2 = 2.0
        assert result == 2.0

    def test_calculate_avg_client_utterance_duration_zero_duration(self):
        """Test calculating average duration with zero duration messages."""
        messages = [
            ChatMessage(
                role="client", content="Zero duration", start_time=0.0, end_time=0.0
            ),
            ChatMessage(
                role="client", content="Normal duration", start_time=1.0, end_time=3.0
            ),
        ]
        result = calculate_avg_client_utterance_duration(messages)
        # Should only count messages with end_time > start_time: 2.0
        assert result == 2.0

    def test_calculate_avg_client_utterance_duration_precision(self):
        """Test that the result is rounded to 2 decimal places."""
        messages = [
            ChatMessage(
                role="client", content="Message 1", start_time=0.0, end_time=1.0
            ),
            ChatMessage(
                role="client", content="Message 2", start_time=2.0, end_time=4.0
            ),
            ChatMessage(
                role="client", content="Message 3", start_time=5.0, end_time=8.0
            ),
        ]
        result = calculate_avg_client_utterance_duration(messages)
        # Average of 1.0, 2.0, 3.0 = 6.0/3 = 2.0
        assert result == 2.0
        assert isinstance(result, float)
