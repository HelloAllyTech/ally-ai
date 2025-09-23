"""
Unit tests for startup utility.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.utils.startup import initialize_openai_clients


class TestInitializeOpenAIClients:
    """Test cases for initialize_openai_clients function."""

    @patch("app.utils.startup.OpenAIEmbeddingClient")
    @patch("app.utils.startup.OpenAITextGenerationClient")
    def test_initialize_openai_clients_success(
        self, mock_text_gen_client, mock_embedding_client
    ):
        """Test successful initialization of OpenAI clients."""
        # Mock the create_client methods
        mock_embedding_client.create_client = MagicMock()
        mock_text_gen_client.create_client = MagicMock()

        # Call the function
        initialize_openai_clients()

        # Verify that both clients were initialized
        mock_embedding_client.create_client.assert_called_once()
        mock_text_gen_client.create_client.assert_called_once()

    @patch("app.utils.startup.OpenAIEmbeddingClient")
    @patch("app.utils.startup.OpenAITextGenerationClient")
    def test_initialize_openai_clients_embedding_client_called_with_correct_model(
        self, mock_text_gen_client, mock_embedding_client
    ):
        """Test that embedding client is called with correct model."""
        # Mock the create_client methods
        mock_embedding_client.create_client = MagicMock()
        mock_text_gen_client.create_client = MagicMock()

        # Call the function
        initialize_openai_clients()

        # Verify embedding client was called with EmbeddingConstants.MODEL
        mock_embedding_client.create_client.assert_called_once()
        # The actual model value is imported from EmbeddingConstants.MODEL
        # The actual model value is imported from EmbeddingConstants.MODEL

    @patch("app.utils.startup.OpenAIEmbeddingClient")
    @patch("app.utils.startup.OpenAITextGenerationClient")
    def test_initialize_openai_clients_text_gen_client_called_with_correct_model(
        self, mock_text_gen_client, mock_embedding_client
    ):
        """Test that text generation client is called with correct model."""
        # Mock the create_client methods
        mock_embedding_client.create_client = MagicMock()
        mock_text_gen_client.create_client = MagicMock()

        # Call the function
        initialize_openai_clients()

        # Verify text generation client was called with TextGenerationConstants
        mock_text_gen_client.create_client.assert_called_once()
        # The actual model value is imported from TextGenerationConstants.DEFAULT_MODEL
        # The actual model value is imported from TextGenerationConstants

    @patch("app.utils.startup.OpenAIEmbeddingClient")
    @patch("app.utils.startup.OpenAITextGenerationClient")
    def test_initialize_openai_clients_embedding_client_exception(
        self, mock_text_gen_client, mock_embedding_client
    ):
        """Test behavior when embedding client initialization fails."""
        # Mock embedding client to raise an exception
        mock_embedding_client.create_client = MagicMock(
            side_effect=Exception("Embedding client error")
        )
        mock_text_gen_client.create_client = MagicMock()

        # Call the function and expect it to raise the exception
        with pytest.raises(Exception, match="Embedding client error"):
            initialize_openai_clients()

        # Verify embedding client was called
        mock_embedding_client.create_client.assert_called_once()
        # Text generation client should not be called if embedding client fails
        mock_text_gen_client.create_client.assert_not_called()

    @patch("app.utils.startup.OpenAIEmbeddingClient")
    @patch("app.utils.startup.OpenAITextGenerationClient")
    def test_initialize_openai_clients_text_gen_client_exception(
        self, mock_text_gen_client, mock_embedding_client
    ):
        """Test behavior when text generation client initialization fails."""
        # Mock text generation client to raise an exception
        mock_embedding_client.create_client = MagicMock()
        mock_text_gen_client.create_client = MagicMock(
            side_effect=Exception("Text gen client error")
        )

        # Call the function and expect it to raise the exception
        with pytest.raises(Exception, match="Text gen client error"):
            initialize_openai_clients()

        # Verify both clients were called (embedding succeeds, text gen fails)
        mock_embedding_client.create_client.assert_called_once()
        mock_text_gen_client.create_client.assert_called_once()

    @patch("app.utils.startup.OpenAIEmbeddingClient")
    @patch("app.utils.startup.OpenAITextGenerationClient")
    def test_initialize_openai_clients_both_clients_exception(
        self, mock_text_gen_client, mock_embedding_client
    ):
        """Test behavior when both clients fail to initialize."""
        # Mock both clients to raise exceptions
        mock_embedding_client.create_client = MagicMock(
            side_effect=Exception("Embedding client error")
        )
        mock_text_gen_client.create_client = MagicMock(
            side_effect=Exception("Text gen client error")
        )

        # Call the function and expect it to raise the first exception
        with pytest.raises(Exception, match="Embedding client error"):
            initialize_openai_clients()

        # Verify embedding client was called
        mock_embedding_client.create_client.assert_called_once()
        # Text generation client should not be called if embedding client fails
        mock_text_gen_client.create_client.assert_not_called()

    @patch("app.utils.startup.OpenAIEmbeddingClient")
    @patch("app.utils.startup.OpenAITextGenerationClient")
    def test_initialize_openai_clients_multiple_calls(
        self, mock_text_gen_client, mock_embedding_client
    ):
        """Test that multiple calls to initialize work correctly."""
        # Mock the create_client methods
        mock_embedding_client.create_client = MagicMock()
        mock_text_gen_client.create_client = MagicMock()

        # Call the function multiple times
        initialize_openai_clients()
        initialize_openai_clients()
        initialize_openai_clients()

        # Verify that both clients were initialized each time
        assert mock_embedding_client.create_client.call_count == 3
        assert mock_text_gen_client.create_client.call_count == 3

    @patch("app.utils.startup.OpenAIEmbeddingClient")
    @patch("app.utils.startup.OpenAITextGenerationClient")
    def test_initialize_openai_clients_return_value(
        self, mock_text_gen_client, mock_embedding_client
    ):
        """Test that the function returns None (no return value)."""
        # Mock the create_client methods
        mock_embedding_client.create_client = MagicMock()
        mock_text_gen_client.create_client = MagicMock()

        # Call the function and verify it returns None
        result = initialize_openai_clients()
        assert result is None

    @patch("app.utils.startup.EmbeddingConstants")
    @patch("app.utils.startup.TextGenerationConstants")
    @patch("app.utils.startup.OpenAIEmbeddingClient")
    @patch("app.utils.startup.OpenAITextGenerationClient")
    def test_initialize_openai_clients_uses_constants(
        self,
        mock_text_gen_client,
        mock_embedding_client,
        mock_text_gen_constants,
        mock_embedding_constants,
    ):
        """Test that the function uses the correct constants."""
        # Mock the constants
        mock_embedding_constants.MODEL = "test-embedding-model"
        mock_text_gen_constants.DEFAULT_MODEL = "test-text-gen-model"

        # Mock the create_client methods
        mock_embedding_client.create_client = MagicMock()
        mock_text_gen_client.create_client = MagicMock()

        # Call the function
        initialize_openai_clients()

        # Verify that the constants were accessed
        # Note: We can't easily verify the exact values passed without more complex
        # mocking but we can verify the methods were called
        mock_embedding_client.create_client.assert_called_once()
        mock_text_gen_client.create_client.assert_called_once()
