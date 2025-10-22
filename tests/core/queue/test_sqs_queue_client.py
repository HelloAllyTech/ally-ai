"""Tests for SQSQueueClient."""

from unittest.mock import MagicMock, patch

import pytest

import app.core.queue.sqs_queue_client as client_module
from app.core.queue.sqs_queue_client import SQSQueueClient


class TestSQSQueueClient:
    """Test cases for SQSQueueClient."""

    def setup_method(self):
        """Reset global client before each test."""
        client_module._sqs_client = None

    def test_get_client_not_created(self):
        """Test getting client when it hasn't been created."""
        with pytest.raises(Exception, match="SQS client has not been created"):
            SQSQueueClient.get_client()

    @patch("app.core.queue.sqs_queue_client.boto3.client")
    @patch("app.core.queue.sqs_queue_client.settings")
    def test_create_client_development(self, mock_settings, mock_boto_client):
        """Test SQS client creation in development environment."""
        # Setup mocks
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_settings.ENV.ENV = "DEVELOPMENT"
        mock_settings.AWS.ACCESS_KEY_ID = "test_access_key"
        mock_settings.AWS.SECRET_ACCESS_KEY = "test_secret_key"
        mock_settings.AWS.REGION = "us-east-1"
        mock_settings.AWS.ENDPOINT_URL = "http://localhost:4566"

        # Execute
        SQSQueueClient.create_client()

        # Assert
        mock_boto_client.assert_called_once()
        call_args = mock_boto_client.call_args
        assert call_args[0][0] == "sqs"
        # Check that credentials are in the call
        assert "aws_access_key_id" in call_args[1]
        assert "aws_secret_access_key" in call_args[1]
        assert call_args[1]["region_name"] == "us-east-1"
        assert call_args[1]["endpoint_url"] == "http://localhost:4566"

    @patch("app.core.queue.sqs_queue_client.boto3.client")
    @patch("app.core.queue.sqs_queue_client.settings")
    def test_create_client_production(self, mock_settings, mock_boto_client):
        """Test SQS client creation in production environment."""
        # Setup mocks
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_settings.ENV.ENV = "production"
        mock_settings.AWS.REGION = "us-east-1"

        # Execute
        SQSQueueClient.create_client()

        # Assert
        mock_boto_client.assert_called_once()
        call_args = mock_boto_client.call_args
        assert call_args[0][0] == "sqs"
        assert call_args[1]["region_name"] == "us-east-1"
        assert "aws_access_key_id" not in call_args[1]
        assert "aws_secret_access_key" not in call_args[1]
        assert "endpoint_url" not in call_args[1]

    @patch("app.core.queue.sqs_queue_client.boto3.client")
    @patch("app.core.queue.sqs_queue_client.settings")
    def test_create_client_already_exists(self, mock_settings, mock_boto_client):
        """Test creating client when one already exists."""
        # Setup mocks
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_settings.ENV.ENV = "DEVELOPMENT"
        mock_settings.AWS.ACCESS_KEY_ID = "test_access_key"
        mock_settings.AWS.SECRET_ACCESS_KEY = "test_secret_key"
        mock_settings.AWS.REGION = "us-east-1"
        mock_settings.AWS.ENDPOINT_URL = "http://localhost:4566"

        # Create client first time
        SQSQueueClient.create_client()

        # Reset mock call count
        mock_boto_client.reset_mock()

        # Try to create client again
        SQSQueueClient.create_client()

        # Assert - should not call boto3.client again
        mock_boto_client.assert_not_called()

    @patch("app.core.queue.sqs_queue_client.boto3.client")
    @patch("app.core.queue.sqs_queue_client.settings")
    def test_get_client_success(self, mock_settings, mock_boto_client):
        """Test getting client after successful creation."""
        # Setup mocks
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_settings.ENV.ENV = "DEVELOPMENT"
        mock_settings.AWS.ACCESS_KEY_ID = "test_access_key"
        mock_settings.AWS.SECRET_ACCESS_KEY = "test_secret_key"
        mock_settings.AWS.REGION = "us-east-1"
        mock_settings.AWS.ENDPOINT_URL = "http://localhost:4566"

        # Create client
        SQSQueueClient.create_client()

        # Get client
        result = SQSQueueClient.get_client()

        # Assert
        assert result == mock_client

    def test_close_client_no_client(self):
        """Test closing client when no client exists."""
        # Execute (should not raise exception)
        SQSQueueClient.close_client()

    @patch("app.core.queue.sqs_queue_client.boto3.client")
    @patch("app.core.queue.sqs_queue_client.settings")
    def test_close_client_success(self, mock_settings, mock_boto_client):
        """Test successful client closure."""
        # Setup mocks
        mock_client = MagicMock()
        mock_endpoint = MagicMock()
        mock_http_session = MagicMock()
        mock_client._endpoint = mock_endpoint
        mock_endpoint.http_session = mock_http_session
        mock_boto_client.return_value = mock_client
        mock_settings.ENV.ENV = "DEVELOPMENT"
        mock_settings.AWS.ACCESS_KEY_ID = "test_access_key"
        mock_settings.AWS.SECRET_ACCESS_KEY = "test_secret_key"
        mock_settings.AWS.REGION = "us-east-1"
        mock_settings.AWS.ENDPOINT_URL = "http://localhost:4566"

        # Create and close client
        SQSQueueClient.create_client()
        SQSQueueClient.close_client()

        # Assert
        mock_http_session.close.assert_called_once()

    @patch("app.core.queue.sqs_queue_client.boto3.client")
    @patch("app.core.queue.sqs_queue_client.settings")
    def test_close_client_no_http_session(self, mock_settings, mock_boto_client):
        """Test closing client when no HTTP session exists."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client._endpoint = None
        mock_boto_client.return_value = mock_client
        mock_settings.ENV.ENV = "DEVELOPMENT"
        mock_settings.AWS.ACCESS_KEY_ID = "test_access_key"
        mock_settings.AWS.SECRET_ACCESS_KEY = "test_secret_key"
        mock_settings.AWS.REGION = "us-east-1"
        mock_settings.AWS.ENDPOINT_URL = "http://localhost:4566"

        # Create and close client
        SQSQueueClient.create_client()
        SQSQueueClient.close_client()

        # Should not raise exception

    @patch("app.core.queue.sqs_queue_client.boto3.client")
    @patch("app.core.queue.sqs_queue_client.settings")
    def test_close_client_exception(self, mock_settings, mock_boto_client):
        """Test closing client with exception during closure."""
        # Setup mocks
        mock_client = MagicMock()
        mock_endpoint = MagicMock()
        mock_http_session = MagicMock()
        mock_http_session.close.side_effect = Exception("Close failed")
        mock_client._endpoint = mock_endpoint
        mock_endpoint.http_session = mock_http_session
        mock_boto_client.return_value = mock_client
        mock_settings.ENV.ENV = "DEVELOPMENT"
        mock_settings.AWS.ACCESS_KEY_ID = "test_access_key"
        mock_settings.AWS.SECRET_ACCESS_KEY = "test_secret_key"
        mock_settings.AWS.REGION = "us-east-1"
        mock_settings.AWS.ENDPOINT_URL = "http://localhost:4566"

        # Create and close client
        SQSQueueClient.create_client()
        SQSQueueClient.close_client()

        # Should not raise exception (caught internally)
        mock_http_session.close.assert_called_once()

    @patch("app.core.queue.sqs_queue_client.boto3.client")
    @patch("app.core.queue.sqs_queue_client.settings")
    def test_client_config(self, mock_settings, mock_boto_client):
        """Test that client is created with correct configuration."""
        # Setup mocks
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_settings.ENV.ENV = "DEVELOPMENT"
        mock_settings.AWS.ACCESS_KEY_ID = "test_access_key"
        mock_settings.AWS.SECRET_ACCESS_KEY = "test_secret_key"
        mock_settings.AWS.REGION = "us-east-1"
        mock_settings.AWS.ENDPOINT_URL = "http://localhost:4566"

        # Execute
        SQSQueueClient.create_client()

        # Assert config is passed
        call_args = mock_boto_client.call_args
        assert "config" in call_args[1]
        config = call_args[1]["config"]
        assert config.connect_timeout == 5
        assert config.read_timeout == 60
        assert config.retries["max_attempts"] == 3
