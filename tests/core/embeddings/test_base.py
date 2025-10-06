"""Tests for BaseEmbeddingService."""

from typing import List
from unittest.mock import MagicMock

import pytest

from app.core.embeddings.base import BaseEmbeddingService


class ConcreteEmbeddingService(BaseEmbeddingService[MagicMock]):
    """Concrete implementation of BaseEmbeddingService for testing."""

    async def embed(self, text: str) -> List[float]:
        """Concrete implementation of embed."""
        return [0.1, 0.2, 0.3]

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Concrete implementation of embed_many."""
        return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


class TestBaseEmbeddingService:
    """Test cases for BaseEmbeddingService."""

    @pytest.fixture
    def mock_client(self):
        """Mock client for testing."""
        return MagicMock()

    @pytest.fixture
    def embedding_service(self, mock_client):
        """Create BaseEmbeddingService instance for testing."""
        return ConcreteEmbeddingService(mock_client)

    def test_init(self, mock_client):
        """Test BaseEmbeddingService initialization."""
        service = ConcreteEmbeddingService(mock_client)
        assert service.client == mock_client

    @pytest.mark.asyncio
    async def test_embed(self, embedding_service):
        """Test embed method."""
        text = "test text"
        result = await embedding_service.embed(text)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_many(self, embedding_service):
        """Test embed_many method."""
        texts = ["text1", "text2"]
        result = await embedding_service.embed_many(texts)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError when not implemented."""

        class IncompleteEmbeddingService(BaseEmbeddingService[MagicMock]):
            """Incomplete implementation missing some abstract methods."""

            async def embed(self, text: str) -> List[float]:
                return []

            # Missing embed_many method

        # This should work since we're not instantiating the incomplete class
        # The abstract methods will be enforced when trying to instantiate
        with pytest.raises(TypeError):
            IncompleteEmbeddingService(MagicMock())
