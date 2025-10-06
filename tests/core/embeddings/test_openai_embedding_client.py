"""Tests for OpenAIEmbeddingClient."""

from unittest.mock import MagicMock, patch

import pytest
import app.core.embeddings.openai_embedding_client as client_module
from app.core.embeddings.openai_embedding_client import OpenAIEmbeddingClient


class TestOpenAIEmbeddingClient:
    """Test cases for OpenAIEmbeddingClient."""

    def setup_method(self):
        """Reset global client before each test."""
        client_module._openai_embedding_client = None

    def test_get_client_not_created(self):
        """Test getting client when it hasn't been created."""
        with pytest.raises(
            Exception, match="OpenAI embedding client has not been created"
        ):
            OpenAIEmbeddingClient.get_client()

    @patch("app.core.embeddings.openai_embedding_client.OpenAIEmbeddings")
    @patch("app.core.embeddings.openai_embedding_client.settings")
    def test_create_client_success(self, mock_settings, mock_openai_embeddings):
        """Test successful client creation."""
        # Setup mocks
        mock_client = MagicMock()
        mock_openai_embeddings.return_value = mock_client
        mock_settings.OPENAI.API_KEY = "test-api-key"
        mock_settings.OPENAI.ORGANIZATION_ID = "test-org-id"

        # Execute
        OpenAIEmbeddingClient.create_client("text-embedding-ada-002")

        # Assert
        mock_openai_embeddings.assert_called_once_with(
            model="text-embedding-ada-002",
            api_key="test-api-key",
            organization="test-org-id",
        )

    @patch("app.core.embeddings.openai_embedding_client.OpenAIEmbeddings")
    @patch("app.core.embeddings.openai_embedding_client.settings")
    def test_create_client_already_exists(self, mock_settings, mock_openai_embeddings):
        """Test creating client when one already exists."""
        # Setup mocks
        mock_client = MagicMock()
        mock_openai_embeddings.return_value = mock_client
        mock_settings.OPENAI.API_KEY = "test-api-key"
        mock_settings.OPENAI.ORGANIZATION_ID = "test-org-id"

        # Create client first time
        OpenAIEmbeddingClient.create_client("text-embedding-ada-002")

        # Reset mock call count
        mock_openai_embeddings.reset_mock()

        # Try to create client again
        OpenAIEmbeddingClient.create_client("text-embedding-3-small")

        # Assert - should not call OpenAIEmbeddings again
        mock_openai_embeddings.assert_not_called()

    @patch("app.core.embeddings.openai_embedding_client.OpenAIEmbeddings")
    @patch("app.core.embeddings.openai_embedding_client.settings")
    def test_get_client_success(self, mock_settings, mock_openai_embeddings):
        """Test getting client after successful creation."""
        # Setup mocks
        mock_client = MagicMock()
        mock_openai_embeddings.return_value = mock_client
        mock_settings.OPENAI.API_KEY = "test-api-key"
        mock_settings.OPENAI.ORGANIZATION_ID = "test-org-id"

        # Create client
        OpenAIEmbeddingClient.create_client("text-embedding-ada-002")

        # Get client
        result = OpenAIEmbeddingClient.get_client()

        # Assert
        assert result == mock_client

    @patch("app.core.embeddings.openai_embedding_client.OpenAIEmbeddings")
    @patch("app.core.embeddings.openai_embedding_client.settings")
    def test_create_client_with_different_models(
        self, mock_settings, mock_openai_embeddings
    ):
        """Test creating client with different model names."""
        # Setup mocks
        mock_client = MagicMock()
        mock_openai_embeddings.return_value = mock_client
        mock_settings.OPENAI.API_KEY = "test-api-key"
        mock_settings.OPENAI.ORGANIZATION_ID = "test-org-id"

        # Test with different models
        models = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]

        for model in models:
            # Reset global client for each test
            client_module._openai_embedding_client = None
            mock_openai_embeddings.reset_mock()

            # Create client
            OpenAIEmbeddingClient.create_client(model)

            # Assert
            mock_openai_embeddings.assert_called_once_with(
                model=model,
                api_key="test-api-key",
                organization="test-org-id",
            )
