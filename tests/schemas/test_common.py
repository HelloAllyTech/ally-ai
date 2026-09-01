"""
Tests for app/schemas/common.py OpenAPI example generation.
"""

from app.schemas.common import ChatMessage


class TestCommonSchemaExamples:
    """The nested `class ConfigDict` in these models is a Pydantic v2 no-op;
    only `model_config = ConfigDict(...)` is recognized, so the intended
    `example` must actually surface in the generated JSON schema."""

    def test_chat_message_schema_has_example(self):
        schema = ChatMessage.model_json_schema()

        assert schema.get("example") == {
            "id": "msg-1",
            "role": "counselor",
            "content": "Hello, how are you?",
            "start_time": 0.5,
            "end_time": 5.7,
        }
