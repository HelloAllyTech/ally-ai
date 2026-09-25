"""Unit tests for the llm_usage emitter: the room_id / scenario_session_id
attribution fields on the payload, and backward compatibility when a caller
omits them (an old ally-be, or a task with no session to attribute to).
"""

import json
from unittest.mock import MagicMock, patch

from app.core.llm_usage import emitter as emitter_module
from app.core.llm_usage.emitter import _build_body, emit_llm_usage_blocking


class TestBuildBody:
    """_build_body is the pure function that shapes the wire payload."""

    def test_includes_scenario_session_id_when_given(self):
        body = json.loads(
            _build_body(
                "llm",
                "openai",
                "gpt-4o",
                "scenario_evaluation",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                room_id="room-abc",
                scenario_session_id="sess-123",
            )
        )
        usage = body["data"]["llm_usage"]
        assert body["room_id"] == "room-abc"
        assert usage["scenario_session_id"] == "sess-123"

    def test_scenario_session_id_defaults_to_none(self):
        """A caller that never mentions scenario_session_id (every existing
        call site, before this change) gets a payload byte-for-byte
        equivalent except for the new null field."""
        body = json.loads(
            _build_body(
                "llm",
                "openai",
                "gpt-4o",
                "scenario_evaluation",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )
        )
        usage = body["data"]["llm_usage"]
        assert body["room_id"] is None
        assert usage["scenario_session_id"] is None
        # Unrelated fields are unaffected.
        assert usage["prompt_tokens"] == 10
        assert usage["task"] == "scenario_evaluation"


def _mock_settings():
    settings = MagicMock()
    settings.LLM_USAGE.ENABLED = True
    settings.LLM_USAGE.QUEUE_URL = "https://sqs.example/test-queue"
    settings.ENV.ENV = "test"
    return settings


class TestEmitLlmUsageBlockingAttribution:
    """End-to-end through emit_llm_usage_blocking (the sync helper judges
    and other non-async call sites use), verifying room_id /
    scenario_session_id reach the message actually sent.
    """

    def test_room_id_and_scenario_session_id_reach_the_wire(self):
        with (
            patch.object(emitter_module, "settings", _mock_settings()),
            patch.object(emitter_module, "_send_blocking") as send,
        ):
            emit_llm_usage_blocking(
                "openai",
                "gpt-4o",
                "scenario_evaluation",
                (10, 5, 15),
                room_id="room-abc",
                scenario_session_id="sess-123",
            )

        send.assert_called_once()
        body = json.loads(send.call_args[0][0])
        assert body["room_id"] == "room-abc"
        assert body["data"]["llm_usage"]["scenario_session_id"] == "sess-123"

    def test_omitting_both_fields_still_emits_successfully(self):
        """Backward compatibility: an old caller passing neither field must
        emit exactly as it did before this change."""
        with (
            patch.object(emitter_module, "settings", _mock_settings()),
            patch.object(emitter_module, "_send_blocking") as send,
        ):
            emit_llm_usage_blocking(
                "openai",
                "gpt-4o",
                "scenario_evaluation",
                (10, 5, 15),
            )

        send.assert_called_once()
        body = json.loads(send.call_args[0][0])
        assert body["room_id"] is None
        assert body["data"]["llm_usage"]["scenario_session_id"] is None
        assert body["data"]["llm_usage"]["prompt_tokens"] == 10

    def test_no_usage_still_no_ops(self):
        """Unrelated to the new fields: emitting with no usage tuple must
        stay a no-op, exactly as before."""
        with (
            patch.object(emitter_module, "settings", _mock_settings()),
            patch.object(emitter_module, "_send_blocking") as send,
        ):
            emit_llm_usage_blocking(
                "openai",
                "gpt-4o",
                "scenario_evaluation",
                None,
                room_id="room-abc",
                scenario_session_id="sess-123",
            )

        send.assert_not_called()
