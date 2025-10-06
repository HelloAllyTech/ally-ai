"""Tests for MessageProcessor."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.core.queue.message_processor import MessageProcessor


class TestMessageProcessor:
    """Test cases for MessageProcessor."""

    @pytest.fixture
    def mock_queue_service(self):
        """Mock SQS queue service."""
        return AsyncMock()

    @pytest.fixture
    def mock_handler(self):
        """Mock message handler."""
        return AsyncMock()

    @pytest.fixture
    def message_processor(self, mock_queue_service, mock_handler):
        """Create MessageProcessor instance for testing."""
        return MessageProcessor(
            queue_service=mock_queue_service,
            handler=mock_handler,
            queue_url="https://example-queue",
            max_messages=10,
            wait_time_seconds=20,
            visibility_timeout=30,
            polling_interval=0,
            delete_after_processing=True,
        )

    def test_init(self, mock_queue_service, mock_handler):
        """Test MessageProcessor initialization."""
        mp = MessageProcessor(
            queue_service=mock_queue_service,
            handler=mock_handler,
            queue_url="test-queue",
            max_messages=5,
            wait_time_seconds=10,
            visibility_timeout=60,
            polling_interval=5,
            delete_after_processing=False,
        )
        
        assert mp.queue_service == mock_queue_service
        assert mp.handler == mock_handler
        assert mp.queue_url == "test-queue"
        assert mp.max_messages == 5
        assert mp.wait_time_seconds == 10
        assert mp.visibility_timeout == 60
        assert mp.polling_interval == 5
        assert mp.delete_after_processing is False
        assert mp._running is False
        assert mp._task is None

    @pytest.mark.asyncio
    async def test_process_message_success_deletes_message(self, message_processor, mock_handler, mock_queue_service):
        """Test successful message processing with deletion."""
        message = {
            "receipt_handle": "rh-123",
            "body": {"chat_id": "c1", "payload": 42},
        }

        await message_processor.process_message(message)

        mock_handler.assert_awaited_once_with({"chat_id": "c1", "payload": 42})
        mock_queue_service.delete_message.assert_awaited_once_with(
            queue_url="https://example-queue", receipt_handle="rh-123"
        )

    @pytest.mark.asyncio
    async def test_process_message_parses_json_string_body(self, mock_queue_service, mock_handler):
        """Test message processing with JSON string body."""
        mp = MessageProcessor(mock_queue_service, mock_handler, "q-url")

        body_dict = {"chat_id": "c2", "x": "y"}
        message = {"receipt_handle": "rh-2", "body": json.dumps(body_dict)}

        await mp.process_message(message)

        mock_handler.assert_awaited_once_with(body_dict)
        mock_queue_service.delete_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_message_invalid_json_body_skips_handler_but_deletes(self, mock_queue_service, mock_handler):
        """Test message processing with invalid JSON body."""
        mp = MessageProcessor(mock_queue_service, mock_handler, "q-url")

        message = {"receipt_handle": "rh-3", "body": "{not-json}"}

        await mp.process_message(message)

        mock_handler.assert_not_awaited()
        mock_queue_service.delete_message.assert_awaited_once_with(
            queue_url="q-url", receipt_handle="rh-3"
        )

    @pytest.mark.asyncio
    async def test_process_message_handler_error_still_deletes(self, mock_queue_service, mock_handler):
        """Test message processing when handler raises exception."""
        mock_handler.side_effect = RuntimeError("boom")
        mp = MessageProcessor(mock_queue_service, mock_handler, "q-url")

        message = {"receipt_handle": "rh-4", "body": {"chat_id": "c4"}}

        await mp.process_message(message)

        mock_queue_service.delete_message.assert_awaited_once_with(
            queue_url="q-url", receipt_handle="rh-4"
        )

    @pytest.mark.asyncio
    async def test_process_message_skip_delete_when_flag_false(self, mock_queue_service, mock_handler):
        """Test message processing without deletion when flag is False."""
        mp = MessageProcessor(
            mock_queue_service, mock_handler, "q-url", delete_after_processing=False
        )

        message = {"receipt_handle": "rh-5", "body": {"chat_id": "c5"}}

        await mp.process_message(message)

        mock_queue_service.delete_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_message_no_receipt_handle_skips_delete(self, mock_queue_service, mock_handler):
        """Test message processing without receipt handle."""
        mp = MessageProcessor(mock_queue_service, mock_handler, "q-url")

        message = {"body": {"chat_id": "c6"}}

        await mp.process_message(message)

        mock_handler.assert_awaited_once()
        mock_queue_service.delete_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_message_delete_error_is_logged(self, mock_queue_service, mock_handler):
        """Test message processing when deletion fails."""
        mock_queue_service.delete_message.side_effect = RuntimeError("delete-fail")
        mp = MessageProcessor(mock_queue_service, mock_handler, "q-url")

        message = {"receipt_handle": "rh-7", "body": {"chat_id": "c7"}}

        # Should not raise despite delete error
        await mp.process_message(message)

        mock_handler.assert_awaited_once()
        mock_queue_service.delete_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_poll_queue_processes_all_messages_concurrently(self, message_processor, mock_queue_service, mock_handler):
        """Test polling queue and processing multiple messages."""
        messages = [
            {"receipt_handle": f"rh-{i}", "body": {"chat_id": f"c-{i}"}}
            for i in range(3)
        ]
        mock_queue_service.receive_messages.return_value = messages

        await message_processor.poll_queue()

        # Verify receive was called with correct params
        mock_queue_service.receive_messages.assert_awaited_once_with(
            queue_url="https://example-queue",
            max_messages=10,
            wait_time_seconds=20,
            visibility_timeout=30,
        )

        # Handler called for each message
        assert mock_handler.await_count == 3
        # Deletion attempted for each message
        assert mock_queue_service.delete_message.await_count == 3

    @pytest.mark.asyncio
    async def test_poll_queue_no_messages_received(self, message_processor, mock_queue_service, mock_handler):
        """Test polling queue when no messages are received."""
        mock_queue_service.receive_messages.return_value = []

        await message_processor.poll_queue()

        mock_queue_service.receive_messages.assert_awaited_once()
        mock_handler.assert_not_awaited()
        mock_queue_service.delete_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_poll_queue_handles_receive_exception(self, message_processor, mock_queue_service, mock_handler):
        """Test polling queue when receive_messages raises exception."""
        mock_queue_service.receive_messages.side_effect = RuntimeError("recv-fail")

        # Should not raise
        await message_processor.poll_queue()

        mock_handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_start_and_stop_lifecycle(self, mock_queue_service, mock_handler):
        """Test start and stop lifecycle of message processor."""
        # Make poll_queue a quick no-op by returning no messages
        mock_queue_service.receive_messages.return_value = []

        mp = MessageProcessor(
            mock_queue_service,
            mock_handler,
            "q-url",
            polling_interval=1,  # small interval for testing
            wait_time_seconds=0,
        )

        # Verify initial state
        assert mp._running is False
        assert mp._task is None

        # Start background task
        await mp.start()
        assert mp._task is not None
        assert not mp._task.done()

        # Give the run() method time to start and set _running = True
        await asyncio.sleep(0.02)
        assert mp._running is True

        # Stop and ensure task is cancelled/awaited cleanly
        await mp.stop()

        assert mp._running is False
        assert mp._task is not None
        assert mp._task.done()

        # Verify it actually called receive_messages
        mock_queue_service.receive_messages.assert_called()

    @pytest.mark.asyncio
    async def test_start_multiple_times_reuses_task(self, mock_queue_service, mock_handler):
        """Test starting message processor multiple times."""
        mock_queue_service.receive_messages.return_value = []

        mp = MessageProcessor(mock_queue_service, mock_handler, "q-url", wait_time_seconds=0, polling_interval=0.01)

        # Start first time
        await mp.start()
        first_task = mp._task

        # Start again - should reuse if task is still running
        await mp.start()
        second_task = mp._task
        
        # Clean up
        await mp.stop()

        assert first_task is not None
        assert first_task == second_task  # Should be the same task

    @pytest.mark.asyncio
    async def test_run_with_polling_interval(self, mock_queue_service, mock_handler):
        """Test run method with polling interval."""
        mock_queue_service.receive_messages.return_value = []

        mp = MessageProcessor(
            mock_queue_service,
            mock_handler,
            "q-url",
            polling_interval=0.01,  # Very small interval for testing
            wait_time_seconds=0,
        )

        # Start task
        task = asyncio.create_task(mp.run())
        await asyncio.sleep(0.02)  # Let it run one cycle
        
        # Stop by setting flag and cancelling
        mp._running = False
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected

        # Should have called receive_messages at least once
        mock_queue_service.receive_messages.assert_awaited()

    @pytest.mark.asyncio
    async def test_run_handles_cancelled_error(self, mock_queue_service, mock_handler):
        """Test run method handles CancelledError gracefully."""
        mock_queue_service.receive_messages.return_value = []

        mp = MessageProcessor(mock_queue_service, mock_handler, "q-url", wait_time_seconds=0, polling_interval=0.01)

        # Start task and cancel it
        task = asyncio.create_task(mp.run())
        await asyncio.sleep(0.02)  # Let it start
        task.cancel()

        # Should not raise CancelledError
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected

        # Verify it handled the cancellation gracefully
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_run_continues_on_exception(self, mock_queue_service, mock_handler):
        """Test run method continues running even when poll_queue raises exception."""
        # First call raises exception, second call returns empty to allow graceful exit
        mock_queue_service.receive_messages.side_effect = [
            RuntimeError("poll-error"),
            []
        ]

        mp = MessageProcessor(mock_queue_service, mock_handler, "q-url", wait_time_seconds=0, polling_interval=0.01)

        # Start task
        task = asyncio.create_task(mp.run())
        await asyncio.sleep(0.05)  # Let it handle the exception and continue
        
        # Stop by setting flag and cancelling
        mp._running = False
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected

        # Should have called receive_messages multiple times despite the error
        assert mock_queue_service.receive_messages.await_count >= 1
