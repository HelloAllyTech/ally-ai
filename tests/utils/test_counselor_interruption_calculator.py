"""
Unit tests for counselor interruption calculator utility.
"""

from app.schemas.common import ChatMessage
from app.utils.counselor_interruption_calculator import (
    calculate_counselor_interruptions,
)


class TestCounselorInterruptionCalculator:
    """Test cases for counselor interruption calculator utility."""

    def test_calculate_counselor_interruptions_empty_list(self):
        """Test calculating interruptions with empty message list."""
        result = calculate_counselor_interruptions([])
        assert result == 0

    def test_calculate_counselor_interruptions_single_message(self):
        """Test calculating interruptions with only one message."""
        messages = [
            ChatMessage(role="client", content="Hello", start_time=0.0, end_time=2.0),
        ]
        result = calculate_counselor_interruptions(messages)
        assert result == 0

    def test_calculate_counselor_interruptions_no_timing_data(self):
        """Test calculating interruptions with messages that have no timing data."""
        messages = [
            ChatMessage(role="client", content="Hello"),
            ChatMessage(role="counselor", content="Hi there"),
        ]
        result = calculate_counselor_interruptions(messages)
        assert result == 0

    def test_calculate_counselor_interruptions_no_interruptions(self):
        """Test calculating interruptions when there are no interruptions."""
        messages = [
            ChatMessage(role="client", content="First", start_time=0.0, end_time=3.0),
            ChatMessage(
                role="counselor", content="Second", start_time=4.0, end_time=6.0
            ),  # No overlap
        ]
        result = calculate_counselor_interruptions(messages)
        assert result == 0

    def test_calculate_counselor_interruptions_single_interruption(self):
        """Test calculating interruptions with one interruption."""
        messages = [
            ChatMessage(role="client", content="First", start_time=0.0, end_time=5.0),
            ChatMessage(
                role="counselor", content="Second", start_time=2.0, end_time=4.0
            ),  # Interrupts client
        ]
        result = calculate_counselor_interruptions(messages)
        assert result == 1

    def test_calculate_counselor_interruptions_multiple_interruptions(self):
        """Test calculating interruptions with multiple interruptions."""
        messages = [
            ChatMessage(role="client", content="First", start_time=0.0, end_time=10.0),
            ChatMessage(
                role="counselor", content="Second", start_time=2.0, end_time=4.0
            ),  # Interrupts
            ChatMessage(
                role="counselor", content="Third", start_time=6.0, end_time=8.0
            ),  # Interrupts
        ]
        result = calculate_counselor_interruptions(messages)
        assert result == 2

    def test_calculate_counselor_interruptions_multiple_client_messages(self):
        """Test calculating interruptions with multiple client messages."""
        messages = [
            ChatMessage(role="client", content="First", start_time=0.0, end_time=3.0),
            ChatMessage(role="client", content="Second", start_time=5.0, end_time=8.0),
            ChatMessage(
                role="counselor", content="Third", start_time=1.0, end_time=2.0
            ),  # Interrupts first client
            ChatMessage(
                role="counselor", content="Fourth", start_time=6.0, end_time=7.0
            ),  # Interrupts second client
        ]
        result = calculate_counselor_interruptions(messages)
        assert result == 2

    def test_calculate_counselor_interruptions_edge_case_start_time(self):
        """Test interruption detection at exact start time boundary."""
        messages = [
            ChatMessage(role="client", content="First", start_time=0.0, end_time=5.0),
            ChatMessage(
                role="counselor", content="Second", start_time=0.0, end_time=2.0
            ),  # Starts at same time
        ]
        result = calculate_counselor_interruptions(messages)
        assert result == 0  # No interruption if start times are equal

    def test_calculate_counselor_interruptions_edge_case_end_time(self):
        """Test interruption detection at exact end time boundary."""
        messages = [
            ChatMessage(role="client", content="First", start_time=0.0, end_time=5.0),
            ChatMessage(
                role="counselor", content="Second", start_time=5.0, end_time=7.0
            ),  # Starts at end time
        ]
        result = calculate_counselor_interruptions(messages)
        assert (
            result == 0
        )  # No interruption if counselor starts exactly when client ends

    def test_calculate_counselor_interruptions_unsorted_messages(self):
        """Test that messages are properly sorted by start_time."""
        messages = [
            ChatMessage(
                role="counselor", content="Second", start_time=2.0, end_time=4.0
            ),
            ChatMessage(role="client", content="First", start_time=0.0, end_time=5.0),
        ]
        result = calculate_counselor_interruptions(messages)
        assert result == 1  # Should still detect interruption despite unsorted input

    def test_calculate_counselor_interruptions_tie_breaking_client_first(self):
        """Test that client messages are prioritized in case of start_time ties."""
        messages = [
            ChatMessage(
                role="counselor", content="Counselor", start_time=0.0, end_time=2.0
            ),
            ChatMessage(
                role="client", content="Client", start_time=0.0, end_time=3.0
            ),  # Same start time
        ]
        result = calculate_counselor_interruptions(messages)
        assert result == 0  # Client should be processed first, so no interruption

    def test_calculate_counselor_interruptions_mixed_roles_no_interruptions(self):
        """Test with mixed roles but no interruptions."""
        messages = [
            ChatMessage(role="client", content="First", start_time=0.0, end_time=2.0),
            ChatMessage(
                role="counselor", content="Second", start_time=3.0, end_time=5.0
            ),
            ChatMessage(role="client", content="Third", start_time=6.0, end_time=8.0),
        ]
        result = calculate_counselor_interruptions(messages)
        assert result == 0

    def test_calculate_counselor_interruptions_missing_timing_fields(self):
        """Test with messages that have some timing fields missing."""
        messages = [
            ChatMessage(role="client", content="Valid", start_time=0.0, end_time=5.0),
            ChatMessage(
                role="client", content="No start", start_time=None, end_time=3.0
            ),
            ChatMessage(
                role="counselor", content="Valid", start_time=2.0, end_time=4.0
            ),
        ]
        result = calculate_counselor_interruptions(messages)
        # Should only process messages with both start_time and end_time
        assert result == 1

    def test_calculate_counselor_interruptions_counselor_to_counselor(self):
        """Test that counselor-to-counselor messages don't cause interruptions."""
        messages = [
            ChatMessage(role="client", content="First", start_time=0.0, end_time=5.0),
            ChatMessage(
                role="counselor", content="Second", start_time=2.0, end_time=3.0
            ),  # Interrupts
            ChatMessage(
                role="counselor", content="Third", start_time=3.5, end_time=4.5
            ),  # Also interrupts client
        ]
        result = calculate_counselor_interruptions(messages)
        assert result == 2  # Both counselor messages interrupt the client

    def test_calculate_counselor_interruptions_complex_scenario(self):
        """Test a complex scenario with overlapping messages."""
        messages = [
            ChatMessage(
                role="client", content="Client1", start_time=0.0, end_time=10.0
            ),
            ChatMessage(
                role="client", content="Client2", start_time=5.0, end_time=15.0
            ),
            ChatMessage(
                role="counselor", content="Counselor1", start_time=2.0, end_time=4.0
            ),  # Interrupts Client1
            ChatMessage(
                role="counselor", content="Counselor2", start_time=7.0, end_time=9.0
            ),  # Interrupts both clients
        ]
        result = calculate_counselor_interruptions(messages)
        assert result == 2  # Both counselor messages cause interruptions
