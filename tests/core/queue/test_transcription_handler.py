"""Tests for TranscriptionHandler."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.queue.message_models import MessageType
from app.core.queue.transcription_handler import TranscriptionHandler


class TestTranscriptionHandler:
    """Class-based tests for TranscriptionHandler."""

    @pytest.fixture
    def mock_ally_core_service(self):
        return AsyncMock()

    @pytest.fixture
    def mock_storage_service(self):
        return AsyncMock()

    @pytest.fixture
    def mock_text_generation_service(self):
        return AsyncMock()

    @pytest.fixture
    def handler(
        self, mock_ally_core_service, mock_storage_service, mock_text_generation_service
    ):
        return TranscriptionHandler(
            ally_core_service=mock_ally_core_service,
            request_queue_url="http://localhost:4566/test-queue",
            result_queue_url="http://localhost:4566/response-queue",
            text_generation_service=mock_text_generation_service,
            storage_service=mock_storage_service,
            bucket_name="test-bucket",
        )

    # ---------------- process_request ----------------
    @pytest.mark.asyncio
    async def test_process_request_success(self, handler, mock_ally_core_service):
        # Arrange
        message_data = {
            "message_type": MessageType.TRANSCRIPTION_RESULT,
            "timestamp": 1,
            "chat_id": 123,
            "segments_text": "00:00-00:02: Hello",
        }
        handler._process_transcription = AsyncMock(return_value=True)
        handler._send_error_response = AsyncMock()

        # Act
        await handler.process_request(message_data)

        # Assert
        handler._process_transcription.assert_awaited_once()
        handler._send_error_response.assert_not_awaited()
        mock_ally_core_service.process_transcript.assert_not_awaited()  # no direct sends here

    @pytest.mark.asyncio
    async def test_process_request_failure_sends_error(self, handler):
        # Arrange
        message_data = {
            "message_type": MessageType.TRANSCRIPTION_RESULT,
            "timestamp": 1,
            "chat_id": 456,
            "segments_text": "...",
        }
        handler._process_transcription = AsyncMock(return_value=False)
        handler._send_error_response = AsyncMock()

        # Act
        await handler.process_request(message_data)

        # Assert
        handler._send_error_response.assert_awaited_once_with(456, "Processing failed")

    @pytest.mark.asyncio
    async def test_process_request_exception_sends_error(self, handler):
        # Arrange
        message_data = {
            "message_type": MessageType.TRANSCRIPTION_RESULT,
            "timestamp": 1,
            "chat_id": 789,
            "segments_text": "...",
        }
        handler._process_transcription = AsyncMock(side_effect=RuntimeError("boom"))
        handler._send_error_response = AsyncMock()

        # Act
        await handler.process_request(message_data)

        # Assert
        handler._send_error_response.assert_awaited_once_with(789, "Processing failed")

    # ---------------- _process_transcription ----------------
    @pytest.mark.asyncio
    async def test__process_transcription_happy_path(
        self,
        handler,
        mock_text_generation_service: AsyncMock,
        mock_storage_service: AsyncMock,
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
        handler.send_combined_result_to_ally_core = AsyncMock()

        request = SimpleNamespace(
            chat_id=111, segments_text="00:00-00:01 hi\n00:01-00:02 hello"
        )

        # Act
        ok = await handler._process_transcription(request)

        # Assert
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
        handler.send_combined_result_to_ally_core.assert_awaited_once_with(
            111, [m.model_dump() for m in passed_messages], fake_summary
        )

    @pytest.mark.asyncio
    async def test__process_transcription_handles_exception(
        self, handler, mock_text_generation_service: AsyncMock
    ):
        # Arrange
        mock_text_generation_service.diarize_from_transcription.side_effect = (
            RuntimeError("oops")
        )
        request = SimpleNamespace(chat_id=222, segments_text="...")

        # Act
        ok = await handler._process_transcription(request)

        # Assert
        assert ok is False

    # ---------------- _generate_summary ----------------
    @pytest.mark.asyncio
    async def test__generate_summary_success(
        self, handler, mock_text_generation_service: AsyncMock
    ):
        # Arrange
        messages = []
        mock_text_generation_service.generate_summary_notes.return_value = {"x": 1}

        # Act
        result = await handler._generate_summary(messages, chat_id=333)

        # Assert
        assert result == {"x": 1}
        mock_text_generation_service.generate_summary_notes.assert_awaited_once()

    @pytest.mark.asyncio
    async def test__generate_summary_failure_sends_error_and_raises(
        self, handler, mock_text_generation_service: AsyncMock
    ):
        # Arrange
        mock_text_generation_service.generate_summary_notes.side_effect = RuntimeError(
            "fail"
        )
        handler._send_error_response = AsyncMock()

        # Act
        with pytest.raises(Exception):
            await handler._generate_summary([], chat_id=444)

        # Assert
        handler._send_error_response.assert_awaited_once_with(444, "Processing failed")

    # ---------------- send_combined_result_to_ally_core ----------------
    @pytest.mark.asyncio
    async def test_send_combined_result_to_ally_core_success(
        self,
        handler,
        mock_storage_service: AsyncMock,
        mock_ally_core_service: AsyncMock,
    ):
        # Arrange
        chat_id = 555
        transcription = [{"role": "CLIENT", "content": "hi"}]
        summary = {"summary": "great"}

        mock_storage_service.generate_presigned_download_url.return_value = (
            "https://dl/url"
        )
        mock_storage_service.generate_presigned_delete_url.return_value = (
            "https://del/url"
        )

        # Act
        await handler.send_combined_result_to_ally_core(chat_id, transcription, summary)

        # Assert upload and presigned generation
        mock_storage_service.upload_to_s3.assert_awaited_once()
        mock_storage_service.generate_presigned_download_url.assert_awaited_once()
        mock_storage_service.generate_presigned_delete_url.assert_awaited_once()

        # Assert queue send with proper body
        assert mock_ally_core_service.process_transcript.await_count == 1
        called_kwargs = mock_ally_core_service.process_transcript.await_args.kwargs
        assert called_kwargs["chat_id"] == chat_id
        assert called_kwargs["download_presigned_url"].startswith("https://dl/")
        assert called_kwargs["delete_presigned_url"].startswith("https://del/")

    @pytest.mark.asyncio
    async def test_send_combined_result_to_ally_core_missing_presigned_triggers_error(
        self, handler, mock_storage_service: AsyncMock
    ):
        # Arrange
        mock_storage_service.generate_presigned_download_url.return_value = None
        mock_storage_service.generate_presigned_delete_url.return_value = (
            "https://del/url"
        )
        handler._send_error_response = AsyncMock()

        # Act
        await handler.send_combined_result_to_ally_core(666, [], {})

        # Assert: error path sends error response
        handler._send_error_response.assert_awaited_once_with(666, "Processing failed")

    # ---------------- _send_error_response ----------------
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
