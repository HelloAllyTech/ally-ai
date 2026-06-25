"""Tests for TranscriptionHandler."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

import pytest

from app.core.constants import PipelineStage
from app.core.queue.message_models import MessageType
from app.core.queue.transcription_request_handler import (
    PipelineStageError,
    TranscriptionRequestHandler,
)


class TestTranscriptionHandler:
    """Class-based tests for TranscriptionHandler."""

    @pytest.fixture
    def mock_ally_core_service(self):
        return AsyncMock()

    @pytest.fixture
    def mock_text_generation_service(self):
        return AsyncMock()

    @pytest.fixture
    def mock_transcription_service(self):
        return Mock()

    @pytest.fixture
    def handler(
        self, mock_ally_core_service, mock_text_generation_service, mock_transcription_service
    ):
        return TranscriptionRequestHandler(
            ally_core_service=mock_ally_core_service,
            text_generation_service=mock_text_generation_service,
            transcription_service=mock_transcription_service
        )

    # ---------------- process_request ----------------
    @pytest.mark.asyncio
    async def test_process_transcription_request(self, handler, mock_ally_core_service):
        # Arrange
        message_data = {
            "message_type": MessageType.TRANSCRIBE_AND_SUMMARIZE_REQUEST,
            "audio_url": "http://example.com",
            "chat_id": 123,
            "sample_rate": 8000,
            "timestamp": 1
        }
        handler.transcription_service.transcribe_audio_from_url = AsyncMock(
            return_value=(None, "transcribed text")
        )
        handler._process_transcription_result = AsyncMock(return_value=True)
        handler._send_error_response = AsyncMock()

        # Act
        await handler.process_transcription_request(message_data)

        # Assert
        handler.transcription_service.transcribe_audio_from_url.assert_awaited_once()
        handler._process_transcription_result.assert_awaited_once()
        handler._send_error_response.assert_not_awaited()
        mock_ally_core_service.process_transcript.assert_not_awaited()  # no direct sends here

    @pytest.mark.asyncio
    async def test_process_transcription_request_audio_transcribe_failure_sends_error(self, handler):
        # Arrange
        message_data = {
            "message_type": MessageType.TRANSCRIBE_AND_SUMMARIZE_REQUEST,
            "audio_url": "http://example.com",
            "chat_id": 123,
            "sample_rate": 8000,
            "timestamp": 1
        }
        handler.transcription_service.transcribe_audio_from_url = AsyncMock(
            side_effect=Exception("Audio Transcribe Failure")
        )

        handler._process_transcription_result = AsyncMock(return_value=True)
        handler._send_error_response = AsyncMock()

        # Act
        await handler.process_transcription_request(message_data)

        # Assert: a transcription failure is reported with the TRANSCRIBE stage
        # and the real upstream reason (not the old generic "Processing failed").
        handler._process_transcription_result.assert_not_awaited()
        handler._send_error_response.assert_awaited_once()
        call = handler._send_error_response.await_args
        assert call.args[0] == 123
        assert "Audio Transcribe Failure" in call.args[1]
        assert call.kwargs["stage"] == PipelineStage.TRANSCRIBE

    @pytest.mark.asyncio
    async def test_process_transcription_request_delivery_failure_no_error(self, handler):
        # A False from _process_transcription_result means the result could not
        # be *delivered* (not a processing failure). We must NOT post an error
        # (that could clobber a result that landed); the message is left for
        # redrive instead, and the handler reports it as unhandled (False).
        message_data = {
            "message_type": MessageType.TRANSCRIBE_AND_SUMMARIZE_REQUEST,
            "audio_url": "http://example.com",
            "chat_id": 123,
            "sample_rate": 8000,
            "timestamp": 1
        }
        handler.transcription_service.transcribe_audio_from_url = AsyncMock(
            return_value=(None, "transcribed text")
        )

        handler._process_transcription_result = AsyncMock(return_value=False)
        handler._send_error_response = AsyncMock()

        # Act
        handled = await handler.process_transcription_request(message_data)

        # Assert
        assert handled is False
        handler._send_error_response.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_transcription_request_exception_sends_error(self, handler):
        # Arrange
        message_data = {
            "message_type": MessageType.TRANSCRIBE_AND_SUMMARIZE_REQUEST,
            "audio_url": "http://example.com",
            "chat_id": 123,
            "sample_rate": 8000,
            "timestamp": 1
        }
        handler.transcription_service.transcribe_audio_from_url = AsyncMock(
            return_value=(None, "transcribed text")
        )

        handler._process_transcription_result = AsyncMock(side_effect=RuntimeError("boom"))
        handler._send_error_response = AsyncMock()

        # Act
        await handler.process_transcription_request(message_data)

        # Assert: unexpected error is reported with the real reason.
        handler._send_error_response.assert_awaited_once()
        call = handler._send_error_response.await_args
        assert call.args[0] == 123
        assert "boom" in call.args[1]

    # ---------------- _process_transcription_result ----------------
    @pytest.mark.asyncio
    async def test__process_transcription_result_happy_path(
        self,
        handler,
        mock_text_generation_service: AsyncMock,
        mock_ally_core_service: AsyncMock,
    ):
        # Arrange diarization result with lowercase roles to verify uppercasing
        diarized_messages = [
            SimpleNamespace(role="client", content="hi", start_time=0, end_time=1),
            SimpleNamespace(
                role="counselor", content="hello", start_time=1, end_time=2
            ),
        ]
        mock_text_generation_service.diarize_from_transcription.return_value = (
            SimpleNamespace(messages=diarized_messages)
        )

        fake_summary = {"summary": "ok"}
        handler._generate_summary = AsyncMock(return_value=fake_summary)
        handler.send_combined_result_to_ally_core = AsyncMock(return_value=True)

        request = SimpleNamespace(
            chat_id=111, segments_text="00:00-00:01 hi\n00:01-00:02 hello"
        )

        # Act
        ok = await handler._process_transcription_result(request)

        # Assert: returns the delivery result (True)
        assert ok is True
        mock_text_generation_service.diarize_from_transcription.assert_awaited_once()
        handler._generate_summary.assert_awaited_once()
        # Validate roles uppercased in messages passed to _generate_summary
        passed_messages = (
            handler._generate_summary.await_args.kwargs.get("messages")
            or handler._generate_summary.await_args.args[0]
        )
        assert passed_messages[0].role == "CLIENT"
        assert passed_messages[1].role == "COUNSELOR"
        # Two-phase delivery: phase 1 sends the transcript on its own, phase 2
        # sends the transcript + summary.
        calls = handler.send_combined_result_to_ally_core.await_args_list
        assert len(calls) == 2
        transcript_payload = [m.model_dump() for m in passed_messages]
        # Phase 1: transcript only (summary is None)
        assert calls[0].args == (111, transcript_payload, None)
        # Phase 2: transcript + summary
        assert calls[1].args == (111, transcript_payload, fake_summary)
        assert calls[1].kwargs.get("summary_error") is None

    @pytest.mark.asyncio
    async def test__process_transcription_raises_stage_error_on_diarize_failure(
        self, handler, mock_text_generation_service: AsyncMock
    ):
        # A diarization failure is a genuine processing failure: it must raise a
        # stage-tagged error (DIARIZE) so the single error path can report it.
        mock_text_generation_service.diarize_from_transcription.side_effect = (
            RuntimeError("oops")
        )
        request = SimpleNamespace(chat_id=222, segments_text="...")

        # Act / Assert
        with pytest.raises(PipelineStageError) as exc:
            await handler._process_transcription_result(request)
        assert exc.value.stage == PipelineStage.DIARIZE

    @pytest.mark.asyncio
    async def test__process_transcription_delivers_transcript_when_summary_fails(
        self, handler, mock_text_generation_service: AsyncMock
    ):
        # A summary failure must NOT discard the transcript: the transcript is
        # delivered with a summary_error so the backend can mark it retryable.
        diarized = [
            SimpleNamespace(role="client", content="hi", start_time=0, end_time=1),
        ]
        mock_text_generation_service.diarize_from_transcription.return_value = (
            SimpleNamespace(messages=diarized)
        )
        handler._generate_summary = AsyncMock(
            side_effect=PipelineStageError(PipelineStage.SUMMARIZE, "llm down")
        )
        handler.send_combined_result_to_ally_core = AsyncMock(return_value=True)

        request = SimpleNamespace(chat_id=321, segments_text="00:00-00:01 hi")

        ok = await handler._process_transcription_result(request)

        assert ok is True
        call = handler.send_combined_result_to_ally_core.await_args
        assert call.args[0] == 321
        assert len(call.args[1]) == 1  # transcript delivered
        assert call.args[2] is None  # no summary
        assert call.kwargs["summary_error"] == "llm down"

    # # ---------------- _generate_summary ----------------
    @pytest.mark.asyncio
    async def test__generate_summary_success(
        self, handler, mock_text_generation_service: AsyncMock
    ):
        # Arrange
        messages = []
        mock_text_generation_service.generate_summary_notes.return_value = {"x": 1}

        # Act
        result = await handler._generate_summary(
            messages, chat_id=333, session_mode="DICTATION"
        )

        # Assert
        assert result == {"x": 1}
        mock_text_generation_service.generate_summary_notes.assert_awaited_once_with(
            chat_history=messages,
            keys=None,
            session_mode="DICTATION",
        )

    @pytest.mark.asyncio
    async def test__generate_summary_default_session_mode(
        self, handler, mock_text_generation_service: AsyncMock
    ):
        messages = []
        mock_text_generation_service.generate_summary_notes.return_value = {"x": 1}

        await handler._generate_summary(messages, chat_id=334)

        mock_text_generation_service.generate_summary_notes.assert_awaited_once_with(
            chat_history=messages,
            keys=None,
            session_mode=None,
        )

    @pytest.mark.asyncio
    async def test__generate_summary_failure_raises_stage_error(
        self, handler, mock_text_generation_service: AsyncMock
    ):
        # A summary failure now raises a stage-tagged error and does NOT post an
        # error itself (the single error path does that), avoiding the old
        # duplicate error callbacks.
        mock_text_generation_service.generate_summary_notes.side_effect = RuntimeError(
            "fail"
        )
        handler._send_error_response = AsyncMock()

        # Act / Assert
        with pytest.raises(PipelineStageError) as exc:
            await handler._generate_summary([], chat_id=444)
        assert exc.value.stage == PipelineStage.SUMMARIZE
        handler._send_error_response.assert_not_awaited()

    # # ---------------- send_combined_result_to_ally_core ----------------
    @pytest.mark.asyncio
    async def test_send_combined_result_to_ally_core_success(
        self,
        handler,
        mock_ally_core_service: AsyncMock,
    ):
        # Arrange
        chat_id = 555
        transcription = [{"role": "CLIENT", "content": "hi"}]
        summary = {"summary": "great"}

        # Act
        await handler.send_combined_result_to_ally_core(chat_id, transcription, summary)

        # Assert queue send with proper body
        assert mock_ally_core_service.process_transcript.await_count == 1
        called_kwargs = mock_ally_core_service.process_transcript.await_args.kwargs
        assert called_kwargs["chat_id"] == chat_id
        assert called_kwargs["transcription"] == transcription
        assert called_kwargs["summary"] == summary

    # # ---------------- _send_error_response ----------------
    @pytest.mark.asyncio
    async def test_send_error_response_pushes_message(
        self, handler, mock_ally_core_service: AsyncMock
    ):
        # Act
        await handler._send_error_response(chat_id=777, error_message="bad news")

        # Assert
        assert mock_ally_core_service.process_transcript.await_count == 1
        kwargs = mock_ally_core_service.process_transcript.await_args.kwargs
        assert kwargs["chat_id"] == 777
        assert kwargs["error"] == "bad news"
