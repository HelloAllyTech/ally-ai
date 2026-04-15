"""Tests for queue message models."""

from datetime import datetime

import pytest

from app.core.queue.message_models import (
    BaseQueueMessage,
    MessageType,
    TranscribeAndSummarizeRequestMessage,
    TranscribeAndSummarizeResponseMessage,
    TranscriptionResultMessage,
)


class TestMessageType:
    """Test cases for MessageType enum."""

    def test_message_type_values(self):
        """Test MessageType enum values."""
        assert MessageType.TRANSCRIPTION_RESULT == "transcription_result"
        assert (
            MessageType.TRANSCRIBE_AND_SUMMARIZE_RESPONSE
            == "transcribe_and_summarize_response"
        )

    def test_message_type_membership(self):
        """Test MessageType enum membership."""
        assert "transcription_result" in MessageType.__members__.values()
        assert "transcribe_and_summarize_response" in MessageType.__members__.values()


class TestBaseQueueMessage:
    """Test cases for BaseQueueMessage."""

    def test_base_queue_message_creation(self):
        """Test BaseQueueMessage creation."""
        timestamp = int(datetime.now().timestamp() * 1000)
        message = BaseQueueMessage(
            message_type=MessageType.TRANSCRIPTION_RESULT, timestamp=timestamp
        )

        assert message.message_type == MessageType.TRANSCRIPTION_RESULT
        assert message.timestamp == timestamp

    def test_base_queue_message_validation(self):
        """Test BaseQueueMessage validation."""
        timestamp = int(datetime.now().timestamp() * 1000)

        # Valid message
        message = BaseQueueMessage(
            message_type=MessageType.TRANSCRIPTION_RESULT, timestamp=timestamp
        )
        assert message.message_type == MessageType.TRANSCRIPTION_RESULT
        assert message.timestamp == timestamp

    def test_base_queue_message_invalid_type(self):
        """Test BaseQueueMessage with invalid message type."""
        timestamp = int(datetime.now().timestamp() * 1000)

        with pytest.raises(ValueError):
            BaseQueueMessage(message_type="invalid_type", timestamp=timestamp)


class TestTranscriptionResultMessage:
    """Test cases for TranscriptionResultMessage."""

    def test_transcription_result_message_creation(self):
        """Test TranscriptionResultMessage creation."""
        timestamp = int(datetime.now().timestamp() * 1000)
        message = TranscriptionResultMessage(
            chat_id=123, segments_text="Hello world", timestamp=timestamp
        )

        assert message.message_type == MessageType.TRANSCRIPTION_RESULT
        assert message.chat_id == 123
        assert message.segments_text == "Hello world"
        assert message.timestamp == timestamp

    def test_transcription_result_message_default_type(self):
        """Test TranscriptionResultMessage with default message type."""
        timestamp = int(datetime.now().timestamp() * 1000)
        message = TranscriptionResultMessage(
            chat_id=456, segments_text="Test transcription", timestamp=timestamp
        )

        assert message.message_type == MessageType.TRANSCRIPTION_RESULT

    def test_transcription_result_message_override_type(self):
        """Test TranscriptionResultMessage with overridden message type."""
        timestamp = int(datetime.now().timestamp() * 1000)
        message = TranscriptionResultMessage(
            message_type=MessageType.TRANSCRIBE_AND_SUMMARIZE_RESPONSE,
            chat_id=789,
            segments_text="Override test",
            timestamp=timestamp,
        )

        assert message.message_type == MessageType.TRANSCRIBE_AND_SUMMARIZE_RESPONSE
        assert message.chat_id == 789
        assert message.segments_text == "Override test"

    def test_transcription_result_message_validation(self):
        """Test TranscriptionResultMessage validation."""
        timestamp = int(datetime.now().timestamp() * 1000)

        # Valid message
        message = TranscriptionResultMessage(
            chat_id=123, segments_text="Valid transcription", timestamp=timestamp
        )
        assert message.chat_id == 123
        assert message.segments_text == "Valid transcription"

    def test_transcription_result_message_empty_text(self):
        """Test TranscriptionResultMessage with empty text."""
        timestamp = int(datetime.now().timestamp() * 1000)
        message = TranscriptionResultMessage(
            chat_id=123, segments_text="", timestamp=timestamp
        )

        assert message.segments_text == ""

    def test_transcription_result_message_long_text(self):
        """Test TranscriptionResultMessage with long text."""
        timestamp = int(datetime.now().timestamp() * 1000)
        long_text = "This is a very long transcription text. " * 100
        message = TranscriptionResultMessage(
            chat_id=123, segments_text=long_text, timestamp=timestamp
        )

        assert message.segments_text == long_text


class TestTranscribeAndSummarizeRequestMessage:
    """Test cases for TranscribeAndSummarizeRequestMessage."""

    def test_request_includes_optional_mode(self):
        timestamp = int(datetime.now().timestamp() * 1000)
        msg = TranscribeAndSummarizeRequestMessage(
            chat_id=1,
            audio_url="https://example.com/a.wav",
            timestamp=timestamp,
            mode="DICTATION",
        )
        assert msg.mode == "DICTATION"
        assert msg.sample_rate == 8000

    def test_request_mode_defaults_to_none(self):
        timestamp = int(datetime.now().timestamp() * 1000)
        msg = TranscribeAndSummarizeRequestMessage(
            chat_id=1,
            audio_url="https://example.com/a.wav",
            timestamp=timestamp,
        )
        assert msg.mode is None


class TestTranscribeAndSummarizeResponseMessage:
    """Test cases for TranscribeAndSummarizeResponseMessage."""

    def test_transcribe_and_summarize_response_message_creation(self):
        """Test TranscribeAndSummarizeResponseMessage creation."""
        timestamp = int(datetime.now().timestamp() * 1000)
        message = TranscribeAndSummarizeResponseMessage(
            chat_id=123,
            download_presigned_url="https://example.com/download",
            delete_presigned_url="https://example.com/delete",
            timestamp=timestamp,
        )

        assert message.message_type == MessageType.TRANSCRIBE_AND_SUMMARIZE_RESPONSE
        assert message.chat_id == 123
        assert message.download_presigned_url == "https://example.com/download"
        assert message.delete_presigned_url == "https://example.com/delete"
        assert message.error is None
        assert message.timestamp == timestamp

    def test_transcribe_and_summarize_response_message_with_error(self):
        """Test TranscribeAndSummarizeResponseMessage with error."""
        timestamp = int(datetime.now().timestamp() * 1000)
        message = TranscribeAndSummarizeResponseMessage(
            chat_id=456, error="Processing failed", timestamp=timestamp
        )

        assert message.message_type == MessageType.TRANSCRIBE_AND_SUMMARIZE_RESPONSE
        assert message.chat_id == 456
        assert message.download_presigned_url is None
        assert message.delete_presigned_url is None
        assert message.error == "Processing failed"

    def test_transcribe_and_summarize_response_message_default_type(self):
        """Test TranscribeAndSummarizeResponseMessage with default message type."""
        timestamp = int(datetime.now().timestamp() * 1000)
        message = TranscribeAndSummarizeResponseMessage(
            chat_id=789, timestamp=timestamp
        )

        assert message.message_type == MessageType.TRANSCRIBE_AND_SUMMARIZE_RESPONSE

    def test_transcribe_and_summarize_response_message_override_type(self):
        """Test TranscribeAndSummarizeResponseMessage with overridden message type."""
        timestamp = int(datetime.now().timestamp() * 1000)
        message = TranscribeAndSummarizeResponseMessage(
            message_type=MessageType.TRANSCRIPTION_RESULT,
            chat_id=101,
            timestamp=timestamp,
        )

        assert message.message_type == MessageType.TRANSCRIPTION_RESULT
        assert message.chat_id == 101

    def test_transcribe_and_summarize_response_message_validation(self):
        """Test TranscribeAndSummarizeResponseMessage validation."""
        timestamp = int(datetime.now().timestamp() * 1000)

        # Valid message with URLs
        message = TranscribeAndSummarizeResponseMessage(
            chat_id=123,
            download_presigned_url="https://example.com/download",
            delete_presigned_url="https://example.com/delete",
            timestamp=timestamp,
        )
        assert message.chat_id == 123
        assert message.download_presigned_url == "https://example.com/download"
        assert message.delete_presigned_url == "https://example.com/delete"

        # Valid message with error
        message = TranscribeAndSummarizeResponseMessage(
            chat_id=456, error="Error message", timestamp=timestamp
        )
        assert message.chat_id == 456
        assert message.error == "Error message"

    def test_transcribe_and_summarize_response_message_minimal(self):
        """Test TranscribeAndSummarizeResponseMessage with minimal fields."""
        timestamp = int(datetime.now().timestamp() * 1000)
        message = TranscribeAndSummarizeResponseMessage(
            chat_id=123, timestamp=timestamp
        )

        assert message.chat_id == 123
        assert message.download_presigned_url is None
        assert message.delete_presigned_url is None
        assert message.error is None

    def test_transcribe_and_summarize_response_message_all_fields(self):
        """Test TranscribeAndSummarizeResponseMessage with all fields."""
        timestamp = int(datetime.now().timestamp() * 1000)
        message = TranscribeAndSummarizeResponseMessage(
            chat_id=123,
            download_presigned_url="https://example.com/download",
            delete_presigned_url="https://example.com/delete",
            error="Warning: Some processing issues",
            timestamp=timestamp,
        )

        assert message.chat_id == 123
        assert message.download_presigned_url == "https://example.com/download"
        assert message.delete_presigned_url == "https://example.com/delete"
        assert message.error == "Warning: Some processing issues"


class TestMessageInheritance:
    """Test cases for message inheritance."""

    def test_message_inheritance(self):
        """Test that message classes properly inherit from BaseQueueMessage."""
        timestamp = int(datetime.now().timestamp() * 1000)

        # Test TranscriptionResultMessage inheritance
        transcription_msg = TranscriptionResultMessage(
            chat_id=123, segments_text="test", timestamp=timestamp
        )
        assert isinstance(transcription_msg, BaseQueueMessage)

        # Test TranscribeAndSummarizeResponseMessage inheritance
        response_msg = TranscribeAndSummarizeResponseMessage(
            chat_id=456, timestamp=timestamp
        )
        assert isinstance(response_msg, BaseQueueMessage)

    def test_message_serialization(self):
        """Test message serialization to dict."""
        timestamp = int(datetime.now().timestamp() * 1000)

        # Test TranscriptionResultMessage serialization
        transcription_msg = TranscriptionResultMessage(
            chat_id=123, segments_text="test transcription", timestamp=timestamp
        )
        data = transcription_msg.model_dump()
        assert data["message_type"] == "transcription_result"
        assert data["chat_id"] == 123
        assert data["segments_text"] == "test transcription"
        assert data["timestamp"] == timestamp

        # Test TranscribeAndSummarizeResponseMessage serialization
        response_msg = TranscribeAndSummarizeResponseMessage(
            chat_id=456,
            download_presigned_url="https://example.com/download",
            timestamp=timestamp,
        )
        data = response_msg.model_dump()
        assert data["message_type"] == "transcribe_and_summarize_response"
        assert data["chat_id"] == 456
        assert data["download_presigned_url"] == "https://example.com/download"
        assert data["delete_presigned_url"] is None
        assert data["error"] is None
