"""
Unit tests for logger utility.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from app.utils.logger import (
    SlackSdkHandler,
    TraceIdFilter,
    get_logger,
    get_trace_id,
    trace_id_var,
)


class TestGetTraceId:
    """Test cases for get_trace_id function."""

    def test_get_trace_id_default(self):
        """Test getting trace ID with default value."""
        # Reset context variable to default
        trace_id_var.set("N/A")
        result = get_trace_id()
        assert result == "N/A"

    def test_get_trace_id_custom(self):
        """Test getting trace ID with custom value."""
        trace_id_var.set("test-trace-123")
        result = get_trace_id()
        assert result == "test-trace-123"

    def test_get_trace_id_empty_string(self):
        """Test getting trace ID with empty string."""
        trace_id_var.set("")
        result = get_trace_id()
        assert result == ""

    def test_get_trace_id_none(self):
        """Test getting trace ID with None value."""
        trace_id_var.set(None)
        result = get_trace_id()
        assert result is None


class TestTraceIdFilter:
    """Test cases for TraceIdFilter class."""

    def test_filter_adds_trace_id(self):
        """Test that filter adds trace_id to log record."""
        filter_instance = TraceIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Set a trace ID
        trace_id_var.set("test-trace-456")

        result = filter_instance.filter(record)

        assert result is True
        assert hasattr(record, "trace_id")
        assert record.trace_id == "test-trace-456"

    def test_filter_with_default_trace_id(self):
        """Test filter with default trace ID."""
        filter_instance = TraceIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Reset to default
        trace_id_var.set("N/A")

        result = filter_instance.filter(record)

        assert result is True
        assert record.trace_id == "N/A"

    def test_filter_always_returns_true(self):
        """Test that filter always returns True."""
        filter_instance = TraceIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = filter_instance.filter(record)
        assert result is True


class TestSlackSdkHandler:
    """Test cases for SlackSdkHandler class."""

    def test_init(self):
        """Test SlackSdkHandler initialization."""
        handler = SlackSdkHandler("test-token", "test-channel")

        assert handler.channel_id == "test-channel"
        assert handler.client is not None

    @patch("app.utils.logger.WebClient")
    def test_emit_success(self, mock_web_client):
        """Test successful log emission to Slack."""
        # Mock the WebClient and its methods
        mock_client = MagicMock()
        mock_web_client.return_value = mock_client

        handler = SlackSdkHandler("test-token", "test-channel")
        handler.client = mock_client

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Test error message",
            args=(),
            exc_info=None,
        )

        # Call emit
        handler.emit(record)

        # Verify that chat_postMessage was called
        mock_client.chat_postMessage.assert_called_once()
        call_args = mock_client.chat_postMessage.call_args
        assert call_args[1]["channel"] == "test-channel"
        assert "ERROR" in call_args[1]["text"]
        assert "Test error message" in call_args[1]["text"]

    @patch("app.utils.logger.WebClient")
    @patch("app.utils.logger._slack_alerts_logger")
    def test_emit_slack_api_error(self, mock_slack_logger, mock_web_client):
        """Test handling of SlackApiError during emission."""
        # Mock the WebClient to raise SlackApiError
        mock_client = MagicMock()
        mock_web_client.return_value = mock_client

        # Create a mock SlackApiError with proper structure
        from slack_sdk.errors import SlackApiError

        mock_error = SlackApiError("channel_not_found", {"error": "channel_not_found"})
        mock_client.chat_postMessage.side_effect = mock_error

        handler = SlackSdkHandler("test-token", "test-channel")
        handler.client = mock_client

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Test error message",
            args=(),
            exc_info=None,
        )

        # Call emit - should not raise exception
        handler.emit(record)

        # Verify that the error was logged
        mock_slack_logger.error.assert_called_once()
        error_call = mock_slack_logger.error.call_args[0][0]
        assert "Failed to send log to Slack" in error_call
        assert "channel_not_found" in error_call

    @patch("app.utils.logger.WebClient")
    def test_emit_general_exception(self, mock_web_client):
        """Test handling of general exceptions during emission."""
        # Mock the WebClient to raise a general exception
        mock_client = MagicMock()
        mock_web_client.return_value = mock_client
        mock_client.chat_postMessage.side_effect = Exception("General error")

        handler = SlackSdkHandler("test-token", "test-channel")
        handler.client = mock_client

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Test error message",
            args=(),
            exc_info=None,
        )

        # Call emit - should raise exception since it only catches SlackApiError
        with pytest.raises(Exception, match="General error"):
            handler.emit(record)

        # Verify that chat_postMessage was called
        mock_client.chat_postMessage.assert_called_once()


class TestGetLogger:
    """Test cases for get_logger function."""

    def test_get_logger_with_name(self):
        """Test getting logger with specific name."""
        logger = get_logger("test-logger")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test-logger"

    def test_get_logger_with_existing_name(self):
        """Test getting logger with existing name."""
        logger1 = get_logger("existing-logger")
        logger2 = get_logger("existing-logger")

        # Should return the same logger instance
        assert logger1 is logger2

    def test_get_logger_with_empty_name(self):
        """Test getting logger with empty name."""
        logger = get_logger("")

        assert isinstance(logger, logging.Logger)
        # Empty string logger name becomes "root" in Python logging
        assert logger.name == "root"

    def test_get_logger_with_none_name(self):
        """Test getting logger with None name."""
        logger = get_logger(None)

        assert isinstance(logger, logging.Logger)
        # None logger name becomes "root" in Python logging
        assert logger.name == "root"

    def test_get_logger_with_special_characters(self):
        """Test getting logger with special characters in name."""
        logger = get_logger("test-logger-123_!@#")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test-logger-123_!@#"


class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def test_trace_id_integration(self):
        """Test that trace ID is properly integrated with logging."""
        # Set a trace ID
        trace_id_var.set("integration-test-789")

        # Get a logger
        logger = get_logger("integration-test")

        # Create a custom handler that captures records
        class TestHandler(logging.Handler):
            def __init__(self):
                super().__init__()
                self.records = []

            def emit(self, record):
                self.records.append(record)

        handler = TestHandler()
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Add the trace ID filter
        trace_filter = TraceIdFilter()
        handler.addFilter(trace_filter)

        # Log a message
        logger.info("Integration test message")

        # Check that the handler received the record with trace_id
        assert len(handler.records) == 1
        record = handler.records[0]
        assert hasattr(record, "trace_id")
        assert record.trace_id == "integration-test-789"

        # Clean up
        logger.removeHandler(handler)

    def test_logger_hierarchy(self):
        """Test logger hierarchy and propagation."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")

        # Both should be valid loggers
        assert isinstance(parent_logger, logging.Logger)
        assert isinstance(child_logger, logging.Logger)

        # Child logger name should include parent
        assert child_logger.name == "parent.child"
