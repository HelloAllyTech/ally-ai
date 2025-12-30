"""Tests for OpenAIEmbeddingService."""

from unittest.mock import AsyncMock

import openai
import pytest
from httpx import Request, Response

from app.core.embeddings.openai_embedding_service import OpenAIEmbeddingService
from app.exceptions.custom_exceptions import EmbeddingFailedException


class TestOpenAIEmbeddingService:
    """Test cases for OpenAIEmbeddingService."""

    @pytest.fixture
    def mock_client(self):
        """Mock OpenAI embeddings client."""
        return AsyncMock()

    @pytest.fixture
    def embedding_service(self, mock_client):
        """Create OpenAIEmbeddingService instance with mocked client."""
        return OpenAIEmbeddingService(mock_client)

    @pytest.mark.asyncio
    async def test_embed_success(self, embedding_service, mock_client):
        """Test successful text embedding."""
        # Setup mocks
        text = "This is a test text"
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_client.aembed_query.return_value = expected_embedding

        # Execute
        result = await embedding_service.embed(text)

        # Assert
        assert result == expected_embedding
        mock_client.aembed_query.assert_called_once_with(text)

    @pytest.mark.asyncio
    async def test_embed_rate_limit_error(self, embedding_service, mock_client):
        """Test embedding with rate limit error."""
        # Setup mocks
        text = "This is a test text"

        # Create a real httpx Request/Response with request set
        request = Request("POST", "https://api.openai.com/v1/embeddings")
        response = Response(status_code=429, request=request)

        mock_error = openai.RateLimitError(
            message="Rate limit exceeded", response=response, body=None
        )
        mock_client.aembed_query.side_effect = mock_error

        # Execute and assert
        with pytest.raises(EmbeddingFailedException) as exc_info:
            await embedding_service.embed(text)

        assert "rate limit exceeded" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_embed_connection_error(self, embedding_service, mock_client):
        """Test embedding with connection error."""
        # Setup mocks
        text = "This is a test text"

        # Create a real httpx Request/Response for APIConnectionError
        request = Request("POST", "https://api.openai.com/v1/embeddings")

        mock_error = openai.APIConnectionError(
            message="Connection error", request=request
        )
        mock_client.aembed_query.side_effect = mock_error

        # Execute and assert
        with pytest.raises(EmbeddingFailedException) as exc_info:
            await embedding_service.embed(text)

        assert "api error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_embed_empty_text(self, embedding_service, mock_client):
        """Test embedding with empty text."""
        # Setup mocks
        text = ""
        expected_embedding = [0.0, 0.0, 0.0]
        mock_client.aembed_query.return_value = expected_embedding

        # Execute
        result = await embedding_service.embed(text)

        # Assert
        assert result == expected_embedding
        mock_client.aembed_query.assert_called_once_with(text)

    @pytest.mark.asyncio
    async def test_embed_long_text(self, embedding_service, mock_client):
        """Test embedding with long text."""
        # Setup mocks
        text = "This is a very long text " * 1000  # Create a long text
        expected_embedding = [0.1] * 1536  # Typical OpenAI embedding dimension
        mock_client.aembed_query.return_value = expected_embedding

        # Execute
        result = await embedding_service.embed(text)

        # Assert
        assert result == expected_embedding
        assert len(result) == 1536
        mock_client.aembed_query.assert_called_once_with(text)

    @pytest.mark.asyncio
    async def test_embed_many_success(self, embedding_service, mock_client):
        """Test successful multiple text embeddings."""
        # Setup mocks
        texts = ["Text 1", "Text 2", "Text 3"]
        expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        mock_client.aembed_documents.return_value = expected_embeddings

        # Execute
        result = await embedding_service.embed_many(texts)

        # Assert
        assert result == expected_embeddings
        assert len(result) == 3
        mock_client.aembed_documents.assert_called_once_with(texts)

    @pytest.mark.asyncio
    async def test_embed_many_rate_limit_error(self, embedding_service, mock_client):
        """Test multiple embeddings with rate limit error."""
        # Setup mocks
        texts = ["Text 1", "Text 2"]

        # Create a real httpx Request/Response with request set
        request = Request("POST", "https://api.openai.com/v1/embeddings")
        response = Response(status_code=429, request=request)

        mock_error = openai.RateLimitError(
            message="Rate limit exceeded", response=response, body=None
        )
        mock_client.aembed_documents.side_effect = mock_error

        # Execute and assert
        with pytest.raises(EmbeddingFailedException) as exc_info:
            await embedding_service.embed_many(texts)

        assert "rate limit exceeded" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_embed_many_connection_error(self, embedding_service, mock_client):
        """Test multiple embeddings with connection error."""
        # Setup mocks
        texts = ["Text 1", "Text 2"]

        # Create a real httpx Request for APIConnectionError
        request = Request("POST", "https://api.openai.com/v1/embeddings")

        mock_error = openai.APIConnectionError(
            message="Connection error", request=request
        )
        mock_client.aembed_documents.side_effect = mock_error

        # Execute and assert
        with pytest.raises(EmbeddingFailedException) as exc_info:
            await embedding_service.embed_many(texts)

        assert "api error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_embed_many_empty_list(self, embedding_service, mock_client):
        """Test multiple embeddings with empty list."""
        # Setup mocks
        texts = []
        expected_embeddings = []
        mock_client.aembed_documents.return_value = expected_embeddings

        # Execute
        result = await embedding_service.embed_many(texts)

        # Assert
        assert result == expected_embeddings
        assert len(result) == 0
        mock_client.aembed_documents.assert_called_once_with(texts)

    @pytest.mark.asyncio
    async def test_embed_many_large_list(self, embedding_service, mock_client):
        """Test multiple embeddings with large list."""
        # Setup mocks
        texts = [f"Text {i}" for i in range(1000)]  # Large list
        expected_embeddings = [[0.1, 0.2, 0.3]] * 1000
        mock_client.aembed_documents.return_value = expected_embeddings

        # Execute
        result = await embedding_service.embed_many(texts)

        # Assert
        assert result == expected_embeddings
        assert len(result) == 1000
        mock_client.aembed_documents.assert_called_once_with(texts)

    @pytest.mark.asyncio
    async def test_embed_special_characters(self, embedding_service, mock_client):
        """Test embedding with special characters."""
        # Setup mocks
        text = "Hello! @#$%^&*()_+ 你好 🌟"
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_client.aembed_query.return_value = expected_embedding

        # Execute
        result = await embedding_service.embed(text)

        # Assert
        assert result == expected_embedding
        mock_client.aembed_query.assert_called_once_with(text)

    @pytest.mark.asyncio
    async def test_embed_unicode_text(self, embedding_service, mock_client):
        """Test embedding with unicode text."""
        # Setup mocks
        text = "مرحبا بالعالم"  # Arabic text
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_client.aembed_query.return_value = expected_embedding

        # Execute
        result = await embedding_service.embed(text)

        # Assert
        assert result == expected_embedding
        mock_client.aembed_query.assert_called_once_with(text)
