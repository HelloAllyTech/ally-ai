"""
Unit tests for common utility functions.
"""

from types import SimpleNamespace

from app.core.text_generations.structured_output_models import MessageTagLabelEnum
from app.schemas.common import ChatMessage
from app.utils.common import (
    build_id_mapping,
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


class TestBuildIdMapping:
    """Test cases for build_id_mapping utility."""

    def test_basic_mapping(self):
        """UUIDs are mapped to sequential keys."""
        messages = [
            ChatMessage(id="aaa-111", role="counselor", content="Hi"),
            ChatMessage(id="bbb-222", role="client", content="Hello"),
            ChatMessage(id="ccc-333", role="counselor", content="How are you?"),
        ]
        uuid_to_key, key_to_uuid = build_id_mapping(messages)

        assert uuid_to_key == {"aaa-111": "m1", "bbb-222": "m2", "ccc-333": "m3"}
        assert key_to_uuid == {"m1": "aaa-111", "m2": "bbb-222", "m3": "ccc-333"}

    def test_empty_messages(self):
        """Empty input returns empty dicts."""
        uuid_to_key, key_to_uuid = build_id_mapping([])
        assert uuid_to_key == {}
        assert key_to_uuid == {}

    def test_duplicate_ids_mapped_once(self):
        """If the same ID appears twice, it is only mapped once (first occurrence)."""
        messages = [
            ChatMessage(id="aaa-111", role="counselor", content="Hi"),
            ChatMessage(id="aaa-111", role="counselor", content="Hi again"),
            ChatMessage(id="bbb-222", role="client", content="Hello"),
        ]
        uuid_to_key, key_to_uuid = build_id_mapping(messages)

        assert uuid_to_key == {"aaa-111": "m1", "bbb-222": "m3"}
        assert key_to_uuid == {"m1": "aaa-111", "m3": "bbb-222"}

    def test_bidirectional_consistency(self):
        """Every mapping in uuid_to_key has a reverse in key_to_uuid."""
        messages = [
            ChatMessage(id="x-1", role="counselor", content="A"),
            ChatMessage(id="x-2", role="client", content="B"),
        ]
        uuid_to_key, key_to_uuid = build_id_mapping(messages)

        for uuid_val, key_val in uuid_to_key.items():
            assert key_to_uuid[key_val] == uuid_val


def _tag(label: MessageTagLabelEnum):
    """
    Helper to build a tag-like object with .label.

    Category is derived from label.
    """
    return SimpleNamespace(label=label)


def _tag_item(id_: str, tags):
    """Helper to build a MessageTagItemOutput-like object."""
    return SimpleNamespace(id=id_, tags=tags)


class TestFilterMessageTags:
    """Test cases for filter_message_tags utility."""

    def test_keeps_only_valid_ids(self):
        """Entries with IDs not in the valid set are dropped."""
        key_to_uuid = {"m1": "uuid-1", "m2": "uuid-2", "m3": "uuid-3"}
        items = [
            _tag_item("m1", [_tag(MessageTagLabelEnum.STEADY_PACING)]),
            _tag_item("m2", [_tag(MessageTagLabelEnum.PARAPHRASING)]),
            _tag_item("m3", [_tag(MessageTagLabelEnum.STEADY_PACING)]),
        ]
        result = filter_message_tags(items, {"m1", "m3"}, key_to_uuid)

        assert len(result) == 2
        assert result[0]["id"] == "uuid-1"
        assert result[1]["id"] == "uuid-3"

    def test_serialises_tags_correctly(self):
        """
        Tags are serialised to dicts with label and category.

        Category is derived from label.
        """
        key_to_uuid = {"m1": "uuid-1"}
        items = [
            _tag_item(
                "m1",
                [
                    _tag(MessageTagLabelEnum.STEADY_PACING),
                    _tag(MessageTagLabelEnum.REINFORCE_AUTONOMY),
                ],
            ),
        ]
        result = filter_message_tags(items, {"m1"}, key_to_uuid)

        assert result == [
            {
                "id": "uuid-1",
                "tags": [
                    {"label": "Steady pacing", "category": "POSITIVE"},
                    {"label": "Reinforce autonomy", "category": "NEGATIVE"},
                ],
            }
        ]

    def test_empty_tags_list(self):
        """Message with no tags produces an empty tags array."""
        key_to_uuid = {"m1": "uuid-1"}
        items = [_tag_item("m1", [])]
        result = filter_message_tags(items, {"m1"}, key_to_uuid)

        assert result == [{"id": "uuid-1", "tags": []}]

    def test_empty_input(self):
        """No items in, no items out."""
        assert filter_message_tags([], {"m1"}, {"m1": "uuid-1"}) == []

    def test_all_filtered_out(self):
        """All items filtered when none match valid set."""
        key_to_uuid = {"m1": "uuid-1"}
        items = [_tag_item("m99", [_tag(MessageTagLabelEnum.STEADY_PACING)])]
        assert filter_message_tags(items, {"m1"}, key_to_uuid) == []

    def test_remaps_keys_to_uuids(self):
        """Output IDs are remapped from keys to original UUIDs."""
        items = [
            _tag_item("m1", [_tag(MessageTagLabelEnum.STEADY_PACING)]),
            _tag_item("m3", [_tag(MessageTagLabelEnum.PARAPHRASING)]),
        ]
        key_to_uuid = {"m1": "aaa-111", "m2": "bbb-222", "m3": "ccc-333"}

        result = filter_message_tags(items, {"m1", "m3"}, key_to_uuid)

        assert len(result) == 2
        assert result[0]["id"] == "aaa-111"
        assert result[1]["id"] == "ccc-333"

    def test_hallucinated_key_filtered(self):
        """Keys not in the valid set are filtered out (hallucination guard)."""
        items = [
            _tag_item("m1", [_tag(MessageTagLabelEnum.STEADY_PACING)]),
            _tag_item("m99", [_tag(MessageTagLabelEnum.STEADY_PACING)]),
        ]
        key_to_uuid = {"m1": "aaa-111"}

        result = filter_message_tags(items, {"m1"}, key_to_uuid)

        assert len(result) == 1
        assert result[0]["id"] == "aaa-111"

    def test_filters_invalid_tag_labels(self):
        """Tags with labels not in MessageTagLabelEnum are filtered out."""
        from types import SimpleNamespace

        # Create a mock tag with an invalid label (not in the enum)
        class FakeLabel:
            value = "Invalid Tag Label"

        key_to_uuid = {"m1": "uuid-1"}
        items = [
            _tag_item(
                "m1",
                [
                    _tag(MessageTagLabelEnum.STEADY_PACING),  # Valid
                    SimpleNamespace(label=FakeLabel()),  # Invalid - not in enum
                    _tag(MessageTagLabelEnum.PARAPHRASING),  # Valid
                ],
            ),
        ]
        result = filter_message_tags(items, {"m1"}, key_to_uuid)

        # Only the two valid tags should remain
        assert len(result) == 1
        assert result[0]["id"] == "uuid-1"
        assert len(result[0]["tags"]) == 2
        assert result[0]["tags"][0]["label"] == "Steady pacing"
        assert result[0]["tags"][1]["label"] == "Paraphrasing"


def _emotional_item(message_id: str, level: int):
    """Helper to build an EmotionalMovementItemOutput-like object."""
    return SimpleNamespace(message_id=message_id, level=level)


class TestFilterEmotionalMovement:
    """Test cases for filter_emotional_movement utility."""

    def test_keeps_only_valid_ids(self):
        """Entries with IDs not in the valid list are dropped."""
        key_to_uuid = {"m1": "uuid-1", "m2": "uuid-2", "m3": "uuid-3"}
        items = [
            _emotional_item("m1", -2),
            _emotional_item("m2", 0),
            _emotional_item("m3", 3),
        ]
        result = filter_emotional_movement(items, ["m1", "m3"], key_to_uuid)

        assert len(result) == 2
        assert result[0]["message_id"] == "uuid-1"
        assert result[0]["level"] == -2
        assert result[1]["message_id"] == "uuid-3"
        assert result[1]["level"] == 3

    def test_serialises_correctly(self):
        """Items are serialised to dicts with message_id and level."""
        key_to_uuid = {"m2": "uuid-2", "m4": "uuid-4"}
        items = [
            _emotional_item("m2", -5),
            _emotional_item("m4", 5),
        ]
        result = filter_emotional_movement(items, ["m2", "m4"], key_to_uuid)

        assert result == [
            {"message_id": "uuid-2", "level": -5},
            {"message_id": "uuid-4", "level": 5},
        ]

    def test_empty_input(self):
        """No LLM items — every client message back-filled with default 0."""
        key_to_uuid = {"m1": "uuid-1", "m2": "uuid-2"}
        result = filter_emotional_movement([], ["m1", "m2"], key_to_uuid)

        assert result == [
            {"message_id": "uuid-1", "level": 0},
            {"message_id": "uuid-2", "level": 0},
        ]

    def test_all_filtered_out(self):
        """LLM rated wrong messages — client messages back-filled with default."""
        key_to_uuid = {"m1": "uuid-1"}
        items = [_emotional_item("m99", 2)]
        result = filter_emotional_movement(items, ["m1"], key_to_uuid)

        assert result == [{"message_id": "uuid-1", "level": 0}]

    def test_preserves_conversation_order(self):
        """Output order matches conversation order, not LLM response order."""
        key_to_uuid = {"m2": "uuid-2", "m5": "uuid-5", "m8": "uuid-8"}
        items = [
            _emotional_item("m8", 0),
            _emotional_item("m2", -3),
            _emotional_item("m5", 1),
        ]
        result = filter_emotional_movement(items, ["m2", "m5", "m8"], key_to_uuid)

        assert [r["message_id"] for r in result] == ["uuid-2", "uuid-5", "uuid-8"]
        assert [r["level"] for r in result] == [-3, 1, 0]

    def test_backfills_missing_messages(self):
        """Client messages the LLM skipped are back-filled with 0."""
        key_to_uuid = {"m2": "uuid-2", "m4": "uuid-4", "m6": "uuid-6"}
        items = [
            _emotional_item("m2", -3),
        ]
        result = filter_emotional_movement(items, ["m2", "m4", "m6"], key_to_uuid)

        assert result == [
            {"message_id": "uuid-2", "level": -3},
            {"message_id": "uuid-4", "level": 0},
            {"message_id": "uuid-6", "level": 0},
        ]

    def test_remaps_keys_to_uuids(self):
        """Output message_ids are remapped from keys to original UUIDs."""
        items = [
            _emotional_item("m2", -3),
            _emotional_item("m4", 1),
        ]
        key_to_uuid = {"m2": "bbb-222", "m4": "ddd-444"}

        result = filter_emotional_movement(items, ["m2", "m4"], key_to_uuid)

        assert result == [
            {"message_id": "bbb-222", "level": -3},
            {"message_id": "ddd-444", "level": 1},
        ]

    def test_backfill_with_remap(self):
        """Missing messages are back-filled and remapped to UUIDs."""
        items = [_emotional_item("m2", -3)]
        key_to_uuid = {"m2": "bbb-222", "m4": "ddd-444"}

        result = filter_emotional_movement(items, ["m2", "m4"], key_to_uuid)

        assert result == [
            {"message_id": "bbb-222", "level": -3},
            {"message_id": "ddd-444", "level": 0},
        ]

    def test_hallucinated_key_filtered_in_emotional(self):
        """Keys not in the valid list are filtered out (hallucination guard)."""
        items = [
            _emotional_item("m2", -3),
            _emotional_item("m99", 5),
        ]
        key_to_uuid = {"m2": "bbb-222"}

        result = filter_emotional_movement(items, ["m2"], key_to_uuid)

        assert result == [{"message_id": "bbb-222", "level": -3}]
