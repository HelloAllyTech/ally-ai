"""Tests for WeaviateClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.vector_db.weaviate_client import WeaviateClient


class TestWeaviateClient:
    """Test cases for WeaviateClient."""

    def setup_method(self):
        """Reset global client before each test."""
        import app.core.vector_db.weaviate_client as weaviate_client_module

        weaviate_client_module._weaviate_client = None

    def test_get_client_not_created(self):
        """Test getting client when it hasn't been created."""
        with pytest.raises(Exception, match="Weaviate client has not been created"):
            WeaviateClient.get_client()

    @patch("app.core.vector_db.weaviate_client.weaviate.use_async_with_custom")
    @patch("app.core.vector_db.weaviate_client.settings")
    def test_create_client_success(self, mock_settings, mock_weaviate_use):
        """Test successful client creation."""
        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_use.return_value = mock_client
        mock_settings.WEAVIATE.HTTP_HOST = "localhost"
        mock_settings.WEAVIATE.HTTP_PORT = 8080
        mock_settings.WEAVIATE.HTTP_SECURE = False
        mock_settings.WEAVIATE.GRPC_HOST = "localhost"
        mock_settings.WEAVIATE.GRPC_PORT = 50051
        mock_settings.WEAVIATE.GRPC_SECURE = False

        # Execute
        WeaviateClient.create_client()

        # Assert
        mock_weaviate_use.assert_called_once_with(
            http_host="localhost",
            http_port=8080,
            http_secure=False,
            grpc_host="localhost",
            grpc_port=50051,
            grpc_secure=False,
        )

    @patch("app.core.vector_db.weaviate_client.weaviate.use_async_with_custom")
    @patch("app.core.vector_db.weaviate_client.settings")
    def test_create_client_already_exists(self, mock_settings, mock_weaviate_use):
        """Test creating client when one already exists."""
        # Setup - create client first time
        mock_client = MagicMock()
        mock_weaviate_use.return_value = mock_client
        mock_settings.WEAVIATE.HTTP_HOST = "localhost"
        mock_settings.WEAVIATE.HTTP_PORT = 8080
        mock_settings.WEAVIATE.HTTP_SECURE = False
        mock_settings.WEAVIATE.GRPC_HOST = "localhost"
        mock_settings.WEAVIATE.GRPC_PORT = 50051
        mock_settings.WEAVIATE.GRPC_SECURE = False

        # Create client first time
        WeaviateClient.create_client()

        # Reset mock call count
        mock_weaviate_use.reset_mock()

        # Try to create client again
        WeaviateClient.create_client()

        # Assert - should not call weaviate.use_async_with_custom again
        mock_weaviate_use.assert_not_called()

    @patch("app.core.vector_db.weaviate_client.weaviate.use_async_with_custom")
    @patch("app.core.vector_db.weaviate_client.settings")
    def test_get_client_success(self, mock_settings, mock_weaviate_use):
        """Test getting client after successful creation."""
        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_use.return_value = mock_client
        mock_settings.WEAVIATE.HTTP_HOST = "localhost"
        mock_settings.WEAVIATE.HTTP_PORT = 8080
        mock_settings.WEAVIATE.HTTP_SECURE = False
        mock_settings.WEAVIATE.GRPC_HOST = "localhost"
        mock_settings.WEAVIATE.GRPC_PORT = 50051
        mock_settings.WEAVIATE.GRPC_SECURE = False

        # Create client
        WeaviateClient.create_client()

        # Get client
        result = WeaviateClient.get_client()

        # Assert
        assert result == mock_client

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection to Weaviate."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_client.is_live.return_value = True

        # Execute
        await WeaviateClient.connect(mock_client)

        # Assert
        mock_client.connect.assert_called_once()
        mock_client.is_live.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_not_live(self):
        """Test connection when client is not live."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_client.is_live.return_value = False

        # Execute
        await WeaviateClient.connect(mock_client)

        # Assert
        mock_client.connect.assert_called_once()
        mock_client.is_live.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_success(self):
        """Test successful client closure."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_client.close.return_value = None

        # Execute
        result = await WeaviateClient.close(mock_client)

        # Assert
        mock_client.close.assert_called_once()
        assert result is None

    @pytest.mark.asyncio
    async def test_close_with_return_value(self):
        """Test client closure with return value."""
        # Setup mocks
        mock_client = AsyncMock()
        expected_result = "closed"
        mock_client.close.return_value = expected_result

        # Execute
        result = await WeaviateClient.close(mock_client)

        # Assert
        mock_client.close.assert_called_once()
        assert result == expected_result
