"""
Unit tests for common utility functions.
"""

from app.schemas.common import ChatMessage
from app.utils.common import convert_chat_messages_to_string


class TestCommonUtils:
    """Test cases for common utility functions."""

    def test_convert_chat_messages_to_string_empty_list(self):
        """Test converting an empty list of chat messages."""
        result = convert_chat_messages_to_string([])
        assert result == ""

    def test_convert_chat_messages_to_string_single_message(self):
        """Test converting a single chat message."""
        messages = [ChatMessage(role="client", content="Hello, how are you?")]
        result = convert_chat_messages_to_string(messages)
        assert result == "client: Hello, how are you?"

    def test_convert_chat_messages_to_string_multiple_messages(self):
        """Test converting multiple chat messages."""
        messages = [
            ChatMessage(role="client", content="I'm feeling anxious today."),
            ChatMessage(
                role="counselor", content="I understand. Can you tell me more?"
            ),
            ChatMessage(role="client", content="Work has been very stressful."),
        ]
        result = convert_chat_messages_to_string(messages)
        expected = (
            "client: I'm feeling anxious today.\n"
            "counselor: I understand. Can you tell me more?\n"
            "client: Work has been very stressful."
        )
        assert result == expected

    def test_convert_chat_messages_to_string_with_empty_content(self):
        """Test converting messages with empty content."""
        messages = [
            ChatMessage(role="client", content=""),
            ChatMessage(role="counselor", content="How are you feeling?"),
        ]
        result = convert_chat_messages_to_string(messages)
        expected = "client: \ncounselor: How are you feeling?"
        assert result == expected

    def test_convert_chat_messages_to_string_with_special_characters(self):
        """Test converting messages with special characters and newlines."""
        messages = [
            ChatMessage(
                role="client", content="I have mixed feelings...\nIt's complicated."
            ),
            ChatMessage(
                role="counselor", content="That sounds challenging. Let's explore this."
            ),
        ]
        result = convert_chat_messages_to_string(messages)
        expected = (
            "client: I have mixed feelings...\nIt's complicated.\n"
            "counselor: That sounds challenging. Let's explore this."
        )
        assert result == expected

    def test_convert_chat_messages_to_string_different_roles(self):
        """Test converting messages with different role types."""
        messages = [
            ChatMessage(role="CLIENT", content="Hello"),
            ChatMessage(role="COUNSELOR", content="Hi there"),
            ChatMessage(role="system", content="Session started"),
        ]
        result = convert_chat_messages_to_string(messages)
        expected = "CLIENT: Hello\nCOUNSELOR: Hi there\nsystem: Session started"
        assert result == expected
