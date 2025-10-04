"""Tests for VectorDB base class."""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from app.core.vector_db.base import VectorDB


class ConcreteVectorDB(VectorDB[MagicMock]):
    """Concrete implementation of VectorDB for testing."""

    async def similarity_search(
        self, vector: List[float], top_k: int = 1
    ) -> List[Dict[str, Any]]:
        """Concrete implementation of similarity_search."""
        return [{"id": "1", "content": "test", "score": 0.9}]

    async def fetch_relevant_conversations(
        self, query: str, top_k: int = 1
    ) -> List[Dict[str, Any]]:
        """Concrete implementation of fetch_relevant_conversations."""
        return [{"conversation": "test conversation", "score": 0.8}]

    async def create_document(
        self,
        collection_name: str,
        document_data: Dict[str, Any],
        vector: List[float],
        document_id: str,
    ) -> str:
        """Concrete implementation of create_document."""
        return document_id

    async def get_document_by_id(
        self, collection_name: str, document_id: str, include_vector: bool = True
    ) -> Dict[str, Any]:
        """Concrete implementation of get_document_by_id."""
        return {"id": document_id, "content": "test document"}

    async def update_document(
        self,
        collection_name: str,
        document_id: str,
        document_data: Dict[str, Any],
        vector: List[float],
    ) -> None:
        """Concrete implementation of update_document."""
        pass

    async def delete_document(self, collection_name: str, document_id: str) -> None:
        """Concrete implementation of delete_document."""
        pass

    async def search_documents(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        filters: Dict[str, Any] = None,
        include_vector: bool = False,
    ) -> Dict[str, Any]:
        """Concrete implementation of search_documents."""
        return {"documents": [], "total_count": 0}


class TestVectorDB:
    """Test cases for VectorDB base class."""

    @pytest.fixture
    def mock_client(self):
        """Mock client for testing."""
        return MagicMock()

    @pytest.fixture
    def vector_db(self, mock_client):
        """Create VectorDB instance for testing."""
        return ConcreteVectorDB(mock_client)

    def test_init(self, mock_client):
        """Test VectorDB initialization."""
        vector_db = ConcreteVectorDB(mock_client)
        assert vector_db.client == mock_client

    @pytest.mark.asyncio
    async def test_similarity_search(self, vector_db):
        """Test similarity search method."""
        vector = [0.1, 0.2, 0.3]
        result = await vector_db.similarity_search(vector, top_k=5)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == "1"

    @pytest.mark.asyncio
    async def test_fetch_relevant_conversations(self, vector_db):
        """Test fetch relevant conversations method."""
        query = "test query"
        result = await vector_db.fetch_relevant_conversations(query, top_k=3)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["conversation"] == "test conversation"

    @pytest.mark.asyncio
    async def test_create_document(self, vector_db):
        """Test create document method."""
        collection_name = "test_collection"
        document_data = {"content": "test content"}
        vector = [0.1, 0.2, 0.3]
        document_id = "test_id"

        result = await vector_db.create_document(
            collection_name, document_data, vector, document_id
        )

        assert result == document_id

    @pytest.mark.asyncio
    async def test_get_document_by_id(self, vector_db):
        """Test get document by ID method."""
        collection_name = "test_collection"
        document_id = "test_id"

        result = await vector_db.get_document_by_id(collection_name, document_id)

        assert result["id"] == document_id
        assert result["content"] == "test document"

    @pytest.mark.asyncio
    async def test_get_document_by_id_without_vector(self, vector_db):
        """Test get document by ID without vector."""
        collection_name = "test_collection"
        document_id = "test_id"

        result = await vector_db.get_document_by_id(
            collection_name, document_id, include_vector=False
        )

        assert result["id"] == document_id

    @pytest.mark.asyncio
    async def test_update_document(self, vector_db):
        """Test update document method."""
        collection_name = "test_collection"
        document_id = "test_id"
        document_data = {"content": "updated content"}
        vector = [0.4, 0.5, 0.6]

        # Should not raise any exception
        await vector_db.update_document(
            collection_name, document_id, document_data, vector
        )

    @pytest.mark.asyncio
    async def test_delete_document(self, vector_db):
        """Test delete document method."""
        collection_name = "test_collection"
        document_id = "test_id"

        # Should not raise any exception
        await vector_db.delete_document(collection_name, document_id)

    @pytest.mark.asyncio
    async def test_search_documents(self, vector_db):
        """Test search documents method."""
        collection_name = "test_collection"
        query = "test query"
        limit = 5
        filters = {"category": "test"}

        result = await vector_db.search_documents(
            collection_name, query, limit, filters, include_vector=True
        )

        assert isinstance(result, dict)
        assert "documents" in result
        assert "total_count" in result

    @pytest.mark.asyncio
    async def test_search_documents_without_filters(self, vector_db):
        """Test search documents without filters."""
        collection_name = "test_collection"
        query = "test query"

        result = await vector_db.search_documents(collection_name, query)

        assert isinstance(result, dict)
        assert "documents" in result
        assert "total_count" in result

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError when not implemented."""

        class IncompleteVectorDB(VectorDB[MagicMock]):
            """Incomplete implementation missing some abstract methods."""

            async def similarity_search(self, vector, top_k=1):
                return []

            async def fetch_relevant_conversations(self, query, top_k=1):
                return []

            # Missing other abstract methods

        # This should work since we're not instantiating the incomplete class
        # The abstract methods will be enforced when trying to instantiate
        with pytest.raises(TypeError):
            IncompleteVectorDB(MagicMock())
