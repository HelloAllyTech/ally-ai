"""
Unit tests for client positivity lift calculator utility.
"""

from app.schemas.common import ChatMessage
from app.utils.client_positivity_lift_calculator import calculate_client_positivity_lift


class TestClientPositivityLiftCalculator:
    """Test cases for client positivity lift calculator utility."""

    def test_calculate_positivity_lift_empty_data(self):
        """Test calculating positivity lift with empty data."""
        result = calculate_client_positivity_lift([])
        assert result is None

    def test_calculate_positivity_lift_insufficient_messages(self):
        """Test calculating positivity lift with insufficient client messages."""
        messages = [
            ChatMessage(role="client", content="I feel great today!"),
            ChatMessage(role="counselor", content="That's wonderful!"),
        ]
        result = calculate_client_positivity_lift(messages)
        assert result is None  # Need at least 10 client messages

    def test_calculate_positivity_lift_sufficient_messages(self):
        """Test calculating positivity lift with sufficient client messages."""
        # Create 10+ client messages with varying sentiment
        messages = []
        for i in range(12):
            if i < 6:
                content = f"I feel terrible about situation {i}."
            else:
                content = f"I'm feeling much better now about situation {i}."
            messages.append(ChatMessage(role="client", content=content))

        result = calculate_client_positivity_lift(messages)
        assert result is not None
        assert isinstance(result, float)

    def test_calculate_positivity_lift_mixed_roles(self):
        """Test calculating positivity lift with mixed roles."""
        messages = []
        for i in range(15):
            if i % 3 == 0:
                messages.append(
                    ChatMessage(
                        role="counselor", content=f"How are you feeling about {i}?"
                    )
                )
            else:
                content = (
                    f"I feel {'terrible' if i < 8 else 'better'} about situation {i}."
                )
                messages.append(ChatMessage(role="client", content=content))

        result = calculate_client_positivity_lift(messages)
        assert result is not None
        assert isinstance(result, float)

    def test_calculate_positivity_lift_positive_trend(self):
        """Test calculating positivity lift with clear positive trend."""
        messages = []
        for i in range(10):
            if i < 5:
                content = "I feel terrible about this situation."
            else:
                content = "I'm feeling much better and more positive about this."
            messages.append(ChatMessage(role="client", content=content))

        result = calculate_client_positivity_lift(messages)
        assert result is not None
        assert isinstance(result, float)
        # Should be positive due to improving sentiment
        assert result > 0

    def test_calculate_positivity_lift_all_positive(self):
        """Test with all client messages being positive."""
        messages = [ChatMessage(role="client", content="I am very happy!")] * 10
        result = calculate_client_positivity_lift(messages)
        assert result is not None
        assert result == 0.0  # No change in sentiment

    def test_calculate_positivity_lift_all_negative(self):
        """Test with all client messages being negative."""
        messages = [ChatMessage(role="client", content="I am very sad!")] * 10
        result = calculate_client_positivity_lift(messages)
        assert result is not None
        assert result == 0.0  # No change in sentiment

    def test_calculate_positivity_lift_negative_to_positive_trend(self):
        """Test a clear trend from negative to positive sentiment."""
        messages = [ChatMessage(role="client", content="I feel terrible.")] * 5 + [
            ChatMessage(role="client", content="I feel great!")
        ] * 5
        result = calculate_client_positivity_lift(messages)
        assert result is not None
        assert result > 0  # Should show a positive lift

    def test_calculate_positivity_lift_positive_to_negative_trend(self):
        """Test a clear trend from positive to negative sentiment."""
        messages = [ChatMessage(role="client", content="I feel great!")] * 5 + [
            ChatMessage(role="client", content="I feel terrible.")
        ] * 5
        result = calculate_client_positivity_lift(messages)
        assert result is not None
        assert result < 0  # Should show a negative lift

    def test_calculate_positivity_lift_fluctuating_sentiment(self):
        """Test with fluctuating sentiment."""
        messages = [
            ChatMessage(role="client", content="Good"),
            ChatMessage(role="client", content="Bad"),
            ChatMessage(role="client", content="Good"),
            ChatMessage(role="client", content="Bad"),
            ChatMessage(role="client", content="Good"),
            ChatMessage(role="client", content="Bad"),
            ChatMessage(role="client", content="Good"),
            ChatMessage(role="client", content="Bad"),
            ChatMessage(role="client", content="Good"),
            ChatMessage(role="client", content="Bad"),
        ]
        result = calculate_client_positivity_lift(messages)
        assert result is not None
        # Expecting a value close to 0 or slightly fluctuating
        assert -50 < result < 50

    def test_calculate_positivity_lift_single_segment_change(self):
        """Test with only two segments having different sentiment."""
        messages = [ChatMessage(role="client", content="Very negative.")] * 9 + [
            ChatMessage(role="client", content="Very positive.")
        ]
        result = calculate_client_positivity_lift(messages)
        assert result is not None
        assert result > 0

    def test_calculate_positivity_lift_exact_ten_messages(self):
        """Test with exactly 10 client messages."""
        messages = [ChatMessage(role="client", content="Neutral.")] * 10
        result = calculate_client_positivity_lift(messages)
        assert result is not None
        assert result == 0.0

    def test_calculate_positivity_lift_non_client_messages_ignored(self):
        """Test that non-client messages are ignored."""
        messages = [
            ChatMessage(role="counselor", content="Hello"),
            ChatMessage(role="client", content="1"),
            ChatMessage(role="counselor", content="How are you?"),
            ChatMessage(role="client", content="2"),
            ChatMessage(role="client", content="3"),
            ChatMessage(role="counselor", content="Okay"),
            ChatMessage(role="client", content="4"),
            ChatMessage(role="client", content="5"),
            ChatMessage(role="client", content="6"),
            ChatMessage(role="counselor", content="Right"),
            ChatMessage(role="client", content="7"),
            ChatMessage(role="client", content="8"),
            ChatMessage(role="client", content="9"),
            ChatMessage(role="client", content="10"),
            ChatMessage(role="client", content="11"),
            ChatMessage(role="client", content="12"),
        ]
        result = calculate_client_positivity_lift(messages)
        assert result is not None
        assert isinstance(result, float)
