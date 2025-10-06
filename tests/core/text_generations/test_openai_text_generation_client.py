"""Tests for OpenAITextGenerationClient."""

from unittest.mock import MagicMock, patch

import pytest

import app.core.text_generations.openai_text_generation_client as client_module
from app.core.text_generations.openai_text_generation_client import (
    OpenAITextGenerationClient,
)


class TestOpenAITextGenerationClient:
    """Test cases for OpenAITextGenerationClient."""

    def setup_method(self):
        """Reset global client before each test."""
        client_module._openai_chat_client = None

    def test_get_client_not_created(self):
        """Test getting client when it hasn't been created."""
        with pytest.raises(Exception, match="OpenAI chat client has not been created"):
            OpenAITextGenerationClient.get_client()

    @patch("app.core.text_generations.openai_text_generation_client.ChatOpenAI")
    @patch("app.core.text_generations.openai_text_generation_client.settings")
    def test_create_client_success(self, mock_settings, mock_chat_openai):
        """Test successful client creation."""
        # Setup mocks
        mock_client = MagicMock()
        mock_chat_openai.return_value = mock_client
        mock_settings.OPENAI.API_KEY = "test-api-key"
        mock_settings.OPENAI.ORGANIZATION_ID = "test-org-id"

        # Execute
        OpenAITextGenerationClient.create_client("gpt-4")

        # Assert
        mock_chat_openai.assert_called_once_with(
            model="gpt-4",
            api_key="test-api-key",
            organization="test-org-id",
        )

    @patch("app.core.text_generations.openai_text_generation_client.ChatOpenAI")
    @patch("app.core.text_generations.openai_text_generation_client.settings")
    def test_create_client_already_exists(self, mock_settings, mock_chat_openai):
        """Test creating client when one already exists."""
        # Setup mocks
        mock_client = MagicMock()
        mock_chat_openai.return_value = mock_client
        mock_settings.OPENAI.API_KEY = "test-api-key"
        mock_settings.OPENAI.ORGANIZATION_ID = "test-org-id"

        # Create client first time
        OpenAITextGenerationClient.create_client("gpt-4")

        # Reset mock call count
        mock_chat_openai.reset_mock()

        # Try to create client again
        OpenAITextGenerationClient.create_client("gpt-3.5-turbo")

        # Assert - should not call ChatOpenAI again
        mock_chat_openai.assert_not_called()

    @patch("app.core.text_generations.openai_text_generation_client.ChatOpenAI")
    @patch("app.core.text_generations.openai_text_generation_client.settings")
    def test_get_client_success(self, mock_settings, mock_chat_openai):
        """Test getting client after successful creation."""
        # Setup mocks
        mock_client = MagicMock()
        mock_chat_openai.return_value = mock_client
        mock_settings.OPENAI.API_KEY = "test-api-key"
        mock_settings.OPENAI.ORGANIZATION_ID = "test-org-id"

        # Create client
        OpenAITextGenerationClient.create_client("gpt-4")

        # Get client
        result = OpenAITextGenerationClient.get_client()

        # Assert
        assert result == mock_client

    @patch("app.core.text_generations.openai_text_generation_client.ChatOpenAI")
    @patch("app.core.text_generations.openai_text_generation_client.settings")
    def test_create_client_with_different_models(self, mock_settings, mock_chat_openai):
        """Test creating client with different model names."""
        # Setup mocks
        mock_client = MagicMock()
        mock_chat_openai.return_value = mock_client
        mock_settings.OPENAI.API_KEY = "test-api-key"
        mock_settings.OPENAI.ORGANIZATION_ID = "test-org-id"

        # Test with different models
        models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]

        for model in models:
            # Reset global client for each test
            client_module._openai_chat_client = None
            mock_chat_openai.reset_mock()

            # Create client
            OpenAITextGenerationClient.create_client(model)

            # Assert
            mock_chat_openai.assert_called_once_with(
                model=model,
                api_key="test-api-key",
                organization="test-org-id",
            )
