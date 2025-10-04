"""Tests for WeaviateDB service."""

from unittest.mock import ANY, AsyncMock, MagicMock
from uuid import uuid4

import pytest
from weaviate.exceptions import AuthenticationFailedException, WeaviateConnectionError

from app.core.vector_db.weaviate import WeaviateDB
from app.exceptions.custom_exceptions import (
    DocumentNotFoundException,
    EmbeddingFailedException,
    VectorDBDeleteFailedException,
    VectorDBFetchFailedException,
    VectorDBInsertFailedException,
    VectorDBSearchFailedException,
    VectorDBUpdateFailedException,
)


class TestWeaviateDB:
    """Test cases for WeaviateDB service."""

    @pytest.fixture
    def mock_client(self):
        """Mock Weaviate client."""
        client = MagicMock()
        client.collections = MagicMock()
        return client

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        return AsyncMock()

    @pytest.fixture
    def weaviate_db(self, mock_client, mock_embedding_service):
        """Create WeaviateDB instance for testing."""
        return WeaviateDB(mock_client, mock_embedding_service)

    @pytest.fixture
    def mock_collection(self):
        """Mock Weaviate collection."""
        collection = MagicMock()
        collection.query = MagicMock()
        collection.query.near_vector = AsyncMock()
        collection.query.fetch_objects = AsyncMock()
        collection.data = MagicMock()
        collection.data.insert = AsyncMock()
        collection.data.update = AsyncMock()
        collection.data.delete_by_id = AsyncMock()
        collection.aggregate = MagicMock()
        collection.aggregate.near_vector = AsyncMock()
        return collection

    @pytest.mark.asyncio
    async def test_similarity_search_success(self, weaviate_db, mock_collection):
        """Test successful similarity search."""
        # Setup mocks
        vector = [0.1, 0.2, 0.3]
        top_k = 5
        mock_result = MagicMock()
        mock_collection.query.near_vector = AsyncMock(return_value=mock_result)
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute
        result = await weaviate_db.similarity_search(vector, top_k)

        # Assert
        assert result == mock_result
        mock_collection.query.near_vector.assert_called_once_with(
            near_vector=vector,
            limit=top_k,
            return_metadata=ANY,  # wvc.query.MetadataQuery(certainty=True)
        )

    @pytest.mark.asyncio
    async def test_similarity_search_connection_error(
        self, weaviate_db, mock_collection
    ):
        """Test similarity search with connection error."""
        # Setup mocks
        vector = [0.1, 0.2, 0.3]
        mock_collection.query.near_vector.side_effect = WeaviateConnectionError(
            "Connection failed"
        )
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute and assert
        with pytest.raises(
            VectorDBSearchFailedException, match="Weaviate connection error"
        ):
            await weaviate_db.similarity_search(vector)

    @pytest.mark.asyncio
    async def test_similarity_search_authentication_error(
        self, weaviate_db, mock_collection
    ):
        """Test similarity search with authentication error."""
        # Setup mocks
        vector = [0.1, 0.2, 0.3]
        mock_collection.query.near_vector.side_effect = AuthenticationFailedException(
            "Auth failed"
        )
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute and assert
        with pytest.raises(
            VectorDBSearchFailedException, match="Weaviate authentication failed"
        ):
            await weaviate_db.similarity_search(vector)

    @pytest.mark.asyncio
    async def test_fetch_relevant_conversations_success(
        self, weaviate_db, mock_collection
    ):
        """Test successful fetch relevant conversations."""
        # Setup mocks
        query = "test query"
        vector = [0.1, 0.2, 0.3]
        mock_result = MagicMock()

        weaviate_db.embedding_service.embed.return_value = vector
        mock_collection.query.near_vector.return_value = mock_result
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute
        result = await weaviate_db.fetch_relevant_conversations(query)

        # Assert
        assert result == mock_result
        weaviate_db.embedding_service.embed.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_fetch_relevant_conversations_embedding_error(self, weaviate_db):
        """Test fetch relevant conversations with embedding error."""
        # Setup mocks
        query = "test query"
        weaviate_db.embedding_service.embed.side_effect = EmbeddingFailedException(
            "Embedding failed"
        )

        # Execute and assert
        with pytest.raises(VectorDBFetchFailedException, match="Embedding failed"):
            await weaviate_db.fetch_relevant_conversations(query)

    @pytest.mark.asyncio
    async def test_create_document_success(self, weaviate_db, mock_collection):
        """Test successful document creation."""
        # Setup mocks
        collection_name = "test_collection"
        document_data = {"content": "test content"}
        vector = [0.1, 0.2, 0.3]
        document_id = str(uuid4())
        mock_result_id = uuid4()

        mock_collection.data.insert.return_value = mock_result_id
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute
        result = await weaviate_db.create_document(
            collection_name, document_data, vector, document_id
        )

        # Assert
        assert result == str(mock_result_id)
        mock_collection.data.insert.assert_called_once_with(
            properties=document_data, vector=vector, uuid=document_id
        )

    @pytest.mark.asyncio
    async def test_create_document_failure(self, weaviate_db, mock_collection):
        """Test document creation failure."""
        # Setup mocks
        collection_name = "test_collection"
        document_data = {"content": "test content"}
        vector = [0.1, 0.2, 0.3]
        document_id = str(uuid4())

        mock_collection.data.insert.side_effect = Exception("Insert failed")
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute and assert
        with pytest.raises(
            VectorDBInsertFailedException, match="Failed to create document"
        ):
            await weaviate_db.create_document(
                collection_name, document_data, vector, document_id
            )

    @pytest.mark.asyncio
    async def test_get_document_by_id_success(self, weaviate_db, mock_collection):
        """Test successful document retrieval by ID."""
        # Setup mocks
        collection_name = "test_collection"
        document_id = str(uuid4())

        mock_obj = MagicMock()
        mock_obj.uuid = document_id
        mock_obj.properties = {"content": "test content"}
        mock_obj.vector = [0.1, 0.2, 0.3]

        mock_result = MagicMock()
        mock_result.objects = [mock_obj]
        mock_collection.query.fetch_objects.return_value = mock_result
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute
        result = await weaviate_db.get_document_by_id(collection_name, document_id)

        # Assert
        expected = {
            "id": document_id,
            "content": "test content",
            "vector": [0.1, 0.2, 0.3],
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_get_document_by_id_not_found(self, weaviate_db, mock_collection):
        """Test document retrieval when document not found."""
        # Setup mocks
        collection_name = "test_collection"
        document_id = str(uuid4())

        mock_result = MagicMock()
        mock_result.objects = []
        mock_collection.query.fetch_objects.return_value = mock_result
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute and assert
        with pytest.raises(
            DocumentNotFoundException, match=f"Document with ID {document_id} not found"
        ):
            await weaviate_db.get_document_by_id(collection_name, document_id)

    @pytest.mark.asyncio
    async def test_get_document_by_id_without_vector(
        self, weaviate_db, mock_collection
    ):
        """Test document retrieval without vector."""
        # Setup mocks
        collection_name = "test_collection"
        document_id = str(uuid4())

        mock_obj = MagicMock()
        mock_obj.uuid = document_id
        mock_obj.properties = {"content": "test content"}
        mock_obj.vector = None

        mock_result = MagicMock()
        mock_result.objects = [mock_obj]
        mock_collection.query.fetch_objects.return_value = mock_result
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute
        result = await weaviate_db.get_document_by_id(
            collection_name, document_id, include_vector=False
        )

        # Assert
        expected = {"id": document_id, "content": "test content"}
        assert result == expected

    @pytest.mark.asyncio
    async def test_get_document_by_id_exception(self, weaviate_db, mock_collection):
        """Test document retrieval with general exception."""
        # Setup mocks
        collection_name = "test_collection"
        document_id = str(uuid4())

        mock_collection.query.fetch_objects.side_effect = Exception("Query failed")
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute and assert
        with pytest.raises(
            DocumentNotFoundException, match=f"Document with ID {document_id} not found"
        ):
            await weaviate_db.get_document_by_id(collection_name, document_id)

    @pytest.mark.asyncio
    async def test_update_document_success(self, weaviate_db, mock_collection):
        """Test successful document update."""
        # Setup mocks
        collection_name = "test_collection"
        document_id = str(uuid4())
        document_data = {"content": "updated content"}
        vector = [0.4, 0.5, 0.6]

        mock_collection.data.update.return_value = None
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute (should not raise exception)
        await weaviate_db.update_document(
            collection_name, document_id, document_data, vector
        )

        # Assert
        mock_collection.data.update.assert_called_once_with(
            uuid=document_id, properties=document_data, vector=vector
        )

    @pytest.mark.asyncio
    async def test_update_document_failure(self, weaviate_db, mock_collection):
        """Test document update failure."""
        # Setup mocks
        collection_name = "test_collection"
        document_id = str(uuid4())
        document_data = {"content": "updated content"}
        vector = [0.4, 0.5, 0.6]

        mock_collection.data.update.side_effect = Exception("Update failed")
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute and assert
        with pytest.raises(
            VectorDBUpdateFailedException, match="Failed to update document"
        ):
            await weaviate_db.update_document(
                collection_name, document_id, document_data, vector
            )

    @pytest.mark.asyncio
    async def test_delete_document_success(self, weaviate_db, mock_collection):
        """Test successful document deletion."""
        # Setup mocks
        collection_name = "test_collection"
        document_id = str(uuid4())

        mock_collection.data.delete_by_id.return_value = None
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute (should not raise exception)
        await weaviate_db.delete_document(collection_name, document_id)

        # Assert
        mock_collection.data.delete_by_id.assert_called_once_with(document_id)

    @pytest.mark.asyncio
    async def test_delete_document_failure(self, weaviate_db, mock_collection):
        """Test document deletion failure."""
        # Setup mocks
        collection_name = "test_collection"
        document_id = str(uuid4())

        mock_collection.data.delete_by_id.side_effect = Exception("Delete failed")
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute and assert
        with pytest.raises(
            VectorDBDeleteFailedException, match="Failed to delete document"
        ):
            await weaviate_db.delete_document(collection_name, document_id)

    @pytest.mark.asyncio
    async def test_search_documents_success(self, weaviate_db, mock_collection):
        """Test successful document search."""
        # Setup mocks
        collection_name = "test_collection"
        query = "test query"
        vector = [0.1, 0.2, 0.3]

        # Mock aggregation result
        mock_group = MagicMock()
        mock_group.grouped_by.value = "category1"
        mock_group.total_count = 5

        mock_agg_result = MagicMock()
        mock_agg_result.groups = [mock_group]
        mock_collection.aggregate.near_vector.return_value = mock_agg_result

        # Mock search result
        mock_obj = MagicMock()
        mock_obj.uuid = str(uuid4())
        mock_obj.properties = {"content": "test content", "category": "category1"}
        mock_obj.vector = [0.1, 0.2, 0.3]
        mock_obj.metadata = MagicMock()
        mock_obj.metadata.distance = 0.2

        mock_result = MagicMock()
        mock_result.objects = [mock_obj]
        mock_collection.query.near_vector.return_value = mock_result

        weaviate_db.embedding_service.embed.return_value = vector
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute
        result = await weaviate_db.search_documents(collection_name, query)

        # Assert
        assert "documents" in result
        assert "total" in result
        assert "categories" in result
        assert result["total"] == 5
        assert result["categories"]["category1"] == 5
        assert len(result["documents"]) == 1
        assert result["documents"][0]["score"] == 0.8  # 1.0 - 0.2

    @pytest.mark.asyncio
    async def test_search_documents_with_filters(self, weaviate_db, mock_collection):
        """Test document search with filters."""
        # Setup mocks
        collection_name = "test_collection"
        query = "test query"
        vector = [0.1, 0.2, 0.3]
        filters = {"category": "test_category", "id": str(uuid4())}

        # Mock aggregation result
        mock_agg_result = MagicMock()
        mock_agg_result.groups = []
        mock_collection.aggregate.near_vector.return_value = mock_agg_result

        # Mock search result
        mock_result = MagicMock()
        mock_result.objects = []
        mock_collection.query.near_vector.return_value = mock_result

        weaviate_db.embedding_service.embed.return_value = vector
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute
        result = await weaviate_db.search_documents(
            collection_name, query, filters=filters
        )

        # Assert
        assert "documents" in result
        assert "total" in result
        assert "categories" in result

    @pytest.mark.asyncio
    async def test_search_documents_with_list_filters(
        self, weaviate_db, mock_collection
    ):
        """Test document search with list filters."""
        # Setup mocks
        collection_name = "test_collection"
        query = "test query"
        vector = [0.1, 0.2, 0.3]
        filters = {"tags": ["tag1", "tag2"]}

        # Mock aggregation result
        mock_agg_result = MagicMock()
        mock_agg_result.groups = []
        mock_collection.aggregate.near_vector.return_value = mock_agg_result

        # Mock search result
        mock_result = MagicMock()
        mock_result.objects = []
        mock_collection.query.near_vector.return_value = mock_result

        weaviate_db.embedding_service.embed.return_value = vector
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute
        result = await weaviate_db.search_documents(
            collection_name, query, filters=filters
        )

        # Assert
        assert "documents" in result
        assert "total" in result
        assert "categories" in result

    @pytest.mark.asyncio
    async def test_search_documents_failure(self, weaviate_db, mock_collection):
        """Test document search failure."""
        # Setup mocks
        collection_name = "test_collection"
        query = "test query"

        weaviate_db.embedding_service.embed.side_effect = Exception("Embedding failed")
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute and assert
        with pytest.raises(
            VectorDBSearchFailedException, match="Failed to search documents"
        ):
            await weaviate_db.search_documents(collection_name, query)

    @pytest.mark.asyncio
    async def test_search_documents_with_include_vector(
        self, weaviate_db, mock_collection
    ):
        """Test document search with vector included."""
        # Setup mocks
        collection_name = "test_collection"
        query = "test query"
        vector = [0.1, 0.2, 0.3]

        # Mock aggregation result
        mock_agg_result = MagicMock()
        mock_agg_result.groups = []
        mock_collection.aggregate.near_vector.return_value = mock_agg_result

        # Mock search result
        mock_obj = MagicMock()
        mock_obj.uuid = str(uuid4())
        mock_obj.properties = {"content": "test content"}
        mock_obj.vector = [0.1, 0.2, 0.3]
        mock_obj.metadata = None

        mock_result = MagicMock()
        mock_result.objects = [mock_obj]
        mock_collection.query.near_vector.return_value = mock_result

        weaviate_db.embedding_service.embed.return_value = vector
        weaviate_db.client.collections.get.return_value = mock_collection

        # Execute
        result = await weaviate_db.search_documents(
            collection_name, query, include_vector=True
        )

        # Assert
        assert "documents" in result
        assert len(result["documents"]) == 1
        assert "vector" in result["documents"][0]
        assert result["documents"][0]["score"] is None  # No metadata
