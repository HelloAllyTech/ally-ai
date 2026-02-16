"""
Unit tests for common utility functions.
"""

from types import SimpleNamespace

from app.schemas.common import ChatMessage
from app.utils.common import (
    convert_chat_messages_to_string,
    filter_emotional_movement,
    filter_message_tags,
)


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



def _tag(label: str, category_value: str):
    """Helper to build a tag-like object with .label and .category.value."""
    cat = SimpleNamespace(value=category_value)
    return SimpleNamespace(label=label, category=cat)


def _tag_item(id_: str, tags):
    """Helper to build a MessageTagItemOutput-like object."""
    return SimpleNamespace(id=id_, tags=tags)


class TestFilterMessageTags:
    """Test cases for filter_message_tags utility."""

    def test_keeps_only_valid_ids(self):
        """Entries with IDs not in the valid set are dropped."""
        items = [
            _tag_item("msg-1", [_tag("Pacing", "POSITIVE")]),
            _tag_item("msg-2", [_tag("Paraphrases", "POSITIVE")]),
            _tag_item("msg-3", [_tag("Pacing", "POSITIVE")]),
        ]
        result = filter_message_tags(items, {"msg-1", "msg-3"})

        assert len(result) == 2
        assert result[0]["id"] == "msg-1"
        assert result[1]["id"] == "msg-3"

    def test_serialises_tags_correctly(self):
        """Tags are serialised to dicts with label and category string."""
        items = [
            _tag_item(
                "msg-1",
                [
                    _tag("Pacing", "POSITIVE"),
                    _tag("Avoid Advice Giving", "NEGATIVE"),
                ],
            ),
        ]
        result = filter_message_tags(items, {"msg-1"})

        assert result == [
            {
                "id": "msg-1",
                "tags": [
                    {"label": "Pacing", "category": "POSITIVE"},
                    {"label": "Avoid Advice Giving", "category": "NEGATIVE"},
                ],
            }
        ]

    def test_empty_tags_list(self):
        """Message with no tags produces an empty tags array."""
        items = [_tag_item("msg-1", [])]
        result = filter_message_tags(items, {"msg-1"})

        assert result == [{"id": "msg-1", "tags": []}]

    def test_empty_input(self):
        """No items in, no items out."""
        assert filter_message_tags([], {"msg-1"}) == []

    def test_all_filtered_out(self):
        """All items filtered when none match valid set."""
        items = [_tag_item("msg-99", [_tag("Pacing", "POSITIVE")])]
        assert filter_message_tags(items, {"msg-1"}) == []


def _emotional_item(message_id: str, level: int):
    """Helper to build an EmotionalMovementItemOutput-like object."""
    return SimpleNamespace(message_id=message_id, level=level)


class TestFilterEmotionalMovement:
    """Test cases for filter_emotional_movement utility."""

    def test_keeps_only_valid_ids(self):
        """Entries with IDs not in the valid list are dropped."""
        items = [
            _emotional_item("msg-1", -2),
            _emotional_item("msg-2", 0),
            _emotional_item("msg-3", 3),
        ]
        result = filter_emotional_movement(items, ["msg-1", "msg-3"])

        assert len(result) == 2
        assert result[0]["message_id"] == "msg-1"
        assert result[0]["level"] == -2
        assert result[1]["message_id"] == "msg-3"
        assert result[1]["level"] == 3

    def test_serialises_correctly(self):
        """Items are serialised to dicts with message_id and level."""
        items = [
            _emotional_item("msg-2", -5),
            _emotional_item("msg-4", 5),
        ]
        result = filter_emotional_movement(items, ["msg-2", "msg-4"])

        assert result == [
            {"message_id": "msg-2", "level": -5},
            {"message_id": "msg-4", "level": 5},
        ]

    def test_empty_input(self):
        """No LLM items — every client message back-filled with default 0."""
        result = filter_emotional_movement([], ["msg-1", "msg-2"])

        assert result == [
            {"message_id": "msg-1", "level": 0},
            {"message_id": "msg-2", "level": 0},
        ]

    def test_all_filtered_out(self):
        """LLM rated wrong messages — client messages back-filled with default."""
        items = [_emotional_item("msg-99", 2)]
        result = filter_emotional_movement(items, ["msg-1"])

        assert result == [{"message_id": "msg-1", "level": 0}]

    def test_preserves_conversation_order(self):
        """Output order matches conversation order, not LLM response order."""
        items = [
            _emotional_item("msg-8", 0),
            _emotional_item("msg-2", -3),
            _emotional_item("msg-5", 1),
        ]
        result = filter_emotional_movement(items, ["msg-2", "msg-5", "msg-8"])

        assert [r["message_id"] for r in result] == ["msg-2", "msg-5", "msg-8"]
        assert [r["level"] for r in result] == [-3, 1, 0]

    def test_backfills_missing_messages(self):
        """Client messages the LLM skipped are back-filled with default level."""
        items = [
            _emotional_item("msg-2", -3),
            # msg-4 missing from LLM response
        ]
        result = filter_emotional_movement(items, ["msg-2", "msg-4", "msg-6"])

        assert result == [
            {"message_id": "msg-2", "level": -3},
            {"message_id": "msg-4", "level": 0},
            {"message_id": "msg-6", "level": 0},
        ]

    def test_custom_default_level(self):
        """Custom default level is used for missing messages."""
        result = filter_emotional_movement([], ["msg-1"], default_level=-1)

        assert result == [{"message_id": "msg-1", "level": -1}]
