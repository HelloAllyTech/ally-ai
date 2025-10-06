"""Tests for SQSQueueService."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from app.core.queue.sqs_queue_service import SQSQueueService


class TestSQSQueueService:
    """Test cases for SQSQueueService."""

    @pytest.fixture
    def mock_client(self):
        """Mock SQS client."""
        return MagicMock()

    @pytest.fixture
    def sqs_service(self, mock_client):
        """Create SQSQueueService instance for testing."""
        return SQSQueueService(mock_client, max_workers=5)

    def test_init(self, mock_client):
        """Test SQSQueueService initialization."""
        service = SQSQueueService(mock_client, max_workers=10)
        assert service.client == mock_client
        assert isinstance(service._executor, ThreadPoolExecutor)

    def test_init_default_max_workers(self, mock_client):
        """Test SQSQueueService initialization with default max_workers."""
        service = SQSQueueService(mock_client)
        assert service.client == mock_client
        assert isinstance(service._executor, ThreadPoolExecutor)

    @pytest.mark.asyncio
    async def test_run_in_executor(self, sqs_service):
        """Test _run_in_executor method."""
        # Setup mocks
        mock_func = MagicMock(return_value="test_result")

        # Execute
        result = await sqs_service._run_in_executor(
            mock_func, "arg1", "arg2", key="value"
        )

        # Assert
        assert result == "test_result"
        mock_func.assert_called_once_with("arg1", "arg2", key="value")

    @pytest.mark.asyncio
    async def test_send_message_success(self, sqs_service):
        """Test successful message sending."""
        # Setup mocks
        queue_url = "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
        message_body = "test message"
        mock_response = {"MessageId": "test-message-id"}
        sqs_service.client.send_message.return_value = mock_response

        # Execute
        result = await sqs_service.send_message(queue_url, message_body)

        # Assert
        assert result == "test-message-id"
        sqs_service.client.send_message.assert_called_once_with(
            QueueUrl=queue_url, MessageBody=message_body, DelaySeconds=0
        )

    @pytest.mark.asyncio
    async def test_send_message_with_delay(self, sqs_service):
        """Test message sending with delay."""
        # Setup mocks
        queue_url = "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
        message_body = "test message"
        delay_seconds = 30
        mock_response = {"MessageId": "test-message-id"}
        sqs_service.client.send_message.return_value = mock_response

        # Execute
        result = await sqs_service.send_message(queue_url, message_body, delay_seconds)

        # Assert
        assert result == "test-message-id"
        sqs_service.client.send_message.assert_called_once_with(
            QueueUrl=queue_url, MessageBody=message_body, DelaySeconds=delay_seconds
        )

    @pytest.mark.asyncio
    async def test_send_message_with_string_attributes(self, sqs_service):
        """Test message sending with string attributes."""
        # Setup mocks
        queue_url = "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
        message_body = "test message"
        message_attributes = {
            "string_attr": "string_value",
            "number_attr": 42,
            "float_attr": 3.14,
        }
        mock_response = {"MessageId": "test-message-id"}
        sqs_service.client.send_message.return_value = mock_response

        # Execute
        result = await sqs_service.send_message(
            queue_url, message_body, message_attributes=message_attributes
        )

        # Assert
        assert result == "test-message-id"
        call_args = sqs_service.client.send_message.call_args
        assert call_args[1]["QueueUrl"] == queue_url
        assert call_args[1]["MessageBody"] == message_body
        assert call_args[1]["DelaySeconds"] == 0
        assert "MessageAttributes" in call_args[1]

        # Check attribute formatting
        attrs = call_args[1]["MessageAttributes"]
        assert attrs["string_attr"]["DataType"] == "String"
        assert attrs["string_attr"]["StringValue"] == "string_value"
        assert attrs["number_attr"]["DataType"] == "Number"
        assert attrs["number_attr"]["StringValue"] == "42"
        assert attrs["float_attr"]["DataType"] == "Number"
        assert attrs["float_attr"]["StringValue"] == "3.14"

    @pytest.mark.asyncio
    async def test_send_message_with_binary_attributes(self, sqs_service):
        """Test message sending with binary attributes."""
        # Setup mocks
        queue_url = "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
        message_body = "test message"
        message_attributes = {"binary_attr": b"binary_data"}
        mock_response = {"MessageId": "test-message-id"}
        sqs_service.client.send_message.return_value = mock_response

        # Execute
        result = await sqs_service.send_message(
            queue_url, message_body, message_attributes=message_attributes
        )

        # Assert
        assert result == "test-message-id"
        call_args = sqs_service.client.send_message.call_args
        attrs = call_args[1]["MessageAttributes"]
        assert attrs["binary_attr"]["DataType"] == "Binary"
        assert attrs["binary_attr"]["BinaryValue"] == b"binary_data"

    @pytest.mark.asyncio
    async def test_send_message_failure(self, sqs_service):
        """Test message sending failure."""
        # Setup mocks
        queue_url = "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
        message_body = "test message"
        sqs_service.client.send_message.side_effect = Exception("Send failed")

        # Execute and assert
        with pytest.raises(Exception, match="Send failed"):
            await sqs_service.send_message(queue_url, message_body)

    @pytest.mark.asyncio
    async def test_receive_messages_success(self, sqs_service):
        """Test successful message receiving."""
        # Setup mocks
        queue_url = "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
        mock_messages = [
            {
                "MessageId": "msg1",
                "ReceiptHandle": "handle1",
                "Body": "message1",
                "Attributes": {"attr1": "value1"},
                "MessageAttributes": {"msg_attr1": "msg_value1"},
            },
            {
                "MessageId": "msg2",
                "ReceiptHandle": "handle2",
                "Body": "message2",
                "Attributes": {},
                "MessageAttributes": {},
            },
        ]
        mock_response = {"Messages": mock_messages}
        sqs_service.client.receive_message.return_value = mock_response

        # Execute
        result = await sqs_service.receive_messages(queue_url)

        # Assert
        assert len(result) == 2
        assert result[0]["message_id"] == "msg1"
        assert result[0]["receipt_handle"] == "handle1"
        assert result[0]["body"] == "message1"
        assert result[0]["attributes"] == {"attr1": "value1"}
        assert result[0]["message_attributes"] == {"msg_attr1": "msg_value1"}

        assert result[1]["message_id"] == "msg2"
        assert result[1]["receipt_handle"] == "handle2"
        assert result[1]["body"] == "message2"
        assert result[1]["attributes"] == {}
        assert result[1]["message_attributes"] == {}

        sqs_service.client.receive_message.assert_called_once_with(
            QueueUrl=queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=20,
            VisibilityTimeout=30,
        )

    @pytest.mark.asyncio
    async def test_receive_messages_with_parameters(self, sqs_service):
        """Test message receiving with custom parameters."""
        # Setup mocks
        queue_url = "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
        max_messages = 5
        wait_time = 10
        visibility_timeout = 60
        message_attribute_names = ["attr1", "attr2"]
        mock_response = {"Messages": []}
        sqs_service.client.receive_message.return_value = mock_response

        # Execute
        result = await sqs_service.receive_messages(
            queue_url,
            max_messages,
            wait_time,
            visibility_timeout,
            message_attribute_names,
        )

        # Assert
        assert result == []
        sqs_service.client.receive_message.assert_called_once_with(
            QueueUrl=queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time,
            VisibilityTimeout=visibility_timeout,
            MessageAttributeNames=message_attribute_names,
        )

    @pytest.mark.asyncio
    async def test_receive_messages_no_messages(self, sqs_service):
        """Test message receiving when no messages available."""
        # Setup mocks
        queue_url = "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
        mock_response = {}  # No Messages key
        sqs_service.client.receive_message.return_value = mock_response

        # Execute
        result = await sqs_service.receive_messages(queue_url)

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test_receive_messages_failure(self, sqs_service):
        """Test message receiving failure."""
        # Setup mocks
        queue_url = "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
        sqs_service.client.receive_message.side_effect = Exception("Receive failed")

        # Execute and assert
        with pytest.raises(Exception, match="Receive failed"):
            await sqs_service.receive_messages(queue_url)

    @pytest.mark.asyncio
    async def test_delete_message_success(self, sqs_service):
        """Test successful message deletion."""
        # Setup mocks
        queue_url = "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
        receipt_handle = "test-receipt-handle"
        mock_response = {"ResponseMetadata": {"RequestId": "test-request-id"}}
        sqs_service.client.delete_message.return_value = mock_response

        # Execute
        result = await sqs_service.delete_message(queue_url, receipt_handle)

        # Assert
        assert result == mock_response
        sqs_service.client.delete_message.assert_called_once_with(
            QueueUrl=queue_url, ReceiptHandle=receipt_handle
        )

    @pytest.mark.asyncio
    async def test_delete_message_failure(self, sqs_service):
        """Test message deletion failure."""
        # Setup mocks
        queue_url = "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
        receipt_handle = "test-receipt-handle"
        sqs_service.client.delete_message.side_effect = Exception("Delete failed")

        # Execute and assert
        with pytest.raises(Exception, match="Delete failed"):
            await sqs_service.delete_message(queue_url, receipt_handle)

    @pytest.mark.asyncio
    async def test_send_message_missing_message_id(self, sqs_service):
        """Test message sending when response is missing MessageId."""
        # Setup mocks
        queue_url = "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"
        message_body = "test message"
        mock_response = {}  # No MessageId
        sqs_service.client.send_message.return_value = mock_response

        # Execute
        result = await sqs_service.send_message(queue_url, message_body)

        # Assert
        assert result is None

    def test_executor_cleanup(self, mock_client):
        """Test that executor is properly initialized."""
        service = SQSQueueService(mock_client, max_workers=3)
        assert service._executor._max_workers == 3

        # Cleanup
        service._executor.shutdown(wait=False)
