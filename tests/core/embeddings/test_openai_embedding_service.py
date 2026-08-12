"""Tests for OpenAIEmbeddingService."""

from unittest.mock import AsyncMock, patch

import openai
import pytest
from httpx import Request, Response

from app.core.constants import EmbeddingConstants
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

    @pytest.fixture(autouse=True)
    def no_retry_sleep(self):
        """Skip the real backoff sleeps.

        Transient failures are now retried with exponential backoff, so without this
        every error-path test would sit through the real delay (~1.5s each) for no added
        coverage.
        """
        with patch(
            "app.core.embeddings.openai_embedding_service.asyncio.sleep",
            new=AsyncMock(),
        ) as sleeper:
            yield sleeper

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
        """An empty list short-circuits without calling the API at all.

        Previously this asserted aembed_documents WAS called with []. Sending an empty
        request to OpenAI costs a round trip to be told nothing, so the call is now
        skipped.
        """
        result = await embedding_service.embed_many([])

        assert result == []
        mock_client.aembed_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_many_batches_large_list(self, embedding_service, mock_client):
        """A list longer than BATCH_SIZE is split into batches, in order.

        This is the behaviour that keeps a 300-page PDF (~500 chunks) from becoming a
        single ~200k-token request that fails as one indivisible unit.
        """
        texts = [f"Text {i}" for i in range(150)]

        # Return one distinct vector per requested text so ordering is verifiable.
        async def fake_embed_documents(batch):
            return [[float(texts.index(t))] for t in batch]

        mock_client.aembed_documents.side_effect = fake_embed_documents

        result = await embedding_service.embed_many(texts)

        # 150 texts at BATCH_SIZE 64 -> 64 + 64 + 22
        assert mock_client.aembed_documents.call_count == 3
        sent_batches = [
            call.args[0] for call in mock_client.aembed_documents.call_args_list
        ]
        assert [len(b) for b in sent_batches] == [
            EmbeddingConstants.BATCH_SIZE,
            EmbeddingConstants.BATCH_SIZE,
            150 - 2 * EmbeddingConstants.BATCH_SIZE,
        ]
        # Every text embedded exactly once, and vectors line up positionally with the
        # input. A reordering here would silently attach the wrong embedding to a
        # passage.
        assert [t for batch in sent_batches for t in batch] == texts
        assert result == [[float(i)] for i in range(150)]

    @pytest.mark.asyncio
    async def test_embed_many_rejects_count_mismatch(
        self, embedding_service, mock_client
    ):
        """A short or long result raises instead of misaligning every later vector."""
        texts = ["Text 1", "Text 2", "Text 3"]
        mock_client.aembed_documents.return_value = [[0.1], [0.2]]  # one too few

        with pytest.raises(EmbeddingFailedException) as exc_info:
            await embedding_service.embed_many(texts)

        assert "different number of embeddings" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_embed_many_retries_then_succeeds(
        self, embedding_service, mock_client
    ):
        """A transient rate limit is retried rather than surfaced to the caller."""
        texts = ["Text 1", "Text 2"]
        request = Request("POST", "https://api.openai.com/v1/embeddings")
        response = Response(status_code=429, request=request)
        rate_limited = openai.RateLimitError(
            message="Rate limit exceeded", response=response, body=None
        )
        mock_client.aembed_documents.side_effect = [
            rate_limited,
            [[0.1], [0.2]],
        ]

        result = await embedding_service.embed_many(texts)

        assert result == [[0.1], [0.2]]
        assert mock_client.aembed_documents.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_does_not_retry_non_transient_errors(
        self, embedding_service, mock_client
    ):
        """Errors that will fail identically every time are not retried.

        Retrying an invalid key or an over-length input just multiplies latency before
        the caller learns what is actually wrong.
        """
        mock_client.aembed_query.side_effect = ValueError("input too long")

        with pytest.raises(ValueError):
            await embedding_service.embed("text")

        assert mock_client.aembed_query.call_count == 1

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
