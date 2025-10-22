"""Tests for ReferenceDocumentService."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core.reference_documents.reference_document_service import (
    ReferenceDocumentService,
)
from app.exceptions.custom_exceptions import (
    DocumentAlreadyExistsException,
    DocumentNotFoundException,
    EmbeddingFailedException,
    VectorDBDeleteFailedException,
    VectorDBInsertFailedException,
    VectorDBSearchFailedException,
    VectorDBUpdateFailedException,
)


class TestReferenceDocumentService:
    """Test cases for ReferenceDocumentService."""

    @pytest.fixture
    def mock_vector_db(self):
        """Mock vector database."""
        return AsyncMock()

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        return AsyncMock()

    @pytest.fixture
    def reference_document_service(self, mock_vector_db, mock_embedding_service):
        """Create ReferenceDocumentService instance with mocked dependencies."""
        return ReferenceDocumentService(mock_vector_db, mock_embedding_service)

    @pytest.fixture
    def sample_document_data(self):
        """Sample document data for testing."""
        return {
            "heading": "Test Document",
            "content": "This is a test document for reference.",
            "category": "test",
            "tags": ["test", "example"],
            "tenant_id": "test-tenant",
            "document_id": str(uuid4()),
        }

    @pytest.fixture
    def sample_embedding(self):
        """Sample embedding vector."""
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_create_document_success(
        self,
        reference_document_service,
        mock_vector_db,
        mock_embedding_service,
        sample_document_data,
        sample_embedding,
    ):
        """Test successful document creation."""
        # Setup mocks
        document_id = sample_document_data["document_id"]
        mock_vector_db.get_document_by_id.side_effect = DocumentNotFoundException(
            "Not found"
        )
        mock_embedding_service.embed.return_value = sample_embedding
        mock_vector_db.create_document.return_value = document_id

        # Execute
        result = await reference_document_service.create_document(
            heading=sample_document_data["heading"],
            content=sample_document_data["content"],
            category=sample_document_data["category"],
            tags=sample_document_data["tags"],
            tenant_id=sample_document_data["tenant_id"],
            document_id=uuid4(),
        )

        # Assert
        assert result == document_id
        mock_embedding_service.embed.assert_called_once_with(
            sample_document_data["content"]
        )
        mock_vector_db.create_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_document_already_exists(
        self, reference_document_service, mock_vector_db, sample_document_data
    ):
        """Test document creation when document already exists."""
        # Setup mocks
        document_id = sample_document_data["document_id"]
        existing_document = {"id": document_id, "heading": "Existing"}
        mock_vector_db.get_document_by_id.return_value = existing_document

        # Execute and assert
        with pytest.raises(DocumentAlreadyExistsException) as exc_info:
            await reference_document_service.create_document(
                heading=sample_document_data["heading"],
                content=sample_document_data["content"],
                category=sample_document_data["category"],
                tags=sample_document_data["tags"],
                tenant_id=sample_document_data["tenant_id"],
                document_id=uuid4(),
            )

        assert "already exists" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_document_embedding_failed(
        self,
        reference_document_service,
        mock_vector_db,
        mock_embedding_service,
        sample_document_data,
    ):
        """Test document creation when embedding generation fails."""
        # Setup mocks
        mock_vector_db.get_document_by_id.side_effect = DocumentNotFoundException(
            "Not found"
        )
        mock_embedding_service.embed.side_effect = EmbeddingFailedException(
            "Embedding error"
        )

        # Execute and assert
        with pytest.raises(EmbeddingFailedException):
            await reference_document_service.create_document(
                heading=sample_document_data["heading"],
                content=sample_document_data["content"],
                category=sample_document_data["category"],
                tags=sample_document_data["tags"],
                tenant_id=sample_document_data["tenant_id"],
                document_id=uuid4(),
            )

    @pytest.mark.asyncio
    async def test_create_document_vector_db_insert_failed(
        self,
        reference_document_service,
        mock_vector_db,
        mock_embedding_service,
        sample_document_data,
        sample_embedding,
    ):
        """Test document creation when vector DB insert fails."""
        # Setup mocks
        mock_vector_db.get_document_by_id.side_effect = DocumentNotFoundException(
            "Not found"
        )
        mock_embedding_service.embed.return_value = sample_embedding
        mock_vector_db.create_document.side_effect = VectorDBInsertFailedException(
            "Insert error"
        )

        # Execute and assert
        with pytest.raises(VectorDBInsertFailedException):
            await reference_document_service.create_document(
                heading=sample_document_data["heading"],
                content=sample_document_data["content"],
                category=sample_document_data["category"],
                tags=sample_document_data["tags"],
                tenant_id=sample_document_data["tenant_id"],
                document_id=uuid4(),
            )

    @pytest.mark.asyncio
    async def test_update_document_success(
        self,
        reference_document_service,
        mock_vector_db,
        mock_embedding_service,
        sample_embedding,
    ):
        """Test successful document update."""
        # Setup mocks
        document_id = "test-doc-id"
        existing_document = {
            "heading": "Original Heading",
            "content": "Original content",
            "category": "original",
            "tags": ["original"],
            "tenant_id": "test-tenant",
            "vector": {"default": sample_embedding},
        }
        mock_vector_db.get_document_by_id.return_value = existing_document
        mock_vector_db.update_document.return_value = None

        # Execute
        await reference_document_service.update_document(
            document_id=document_id,
            heading="Updated Heading",
            content="Updated content",
        )

        # Assert
        mock_vector_db.update_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_document_not_found(
        self, reference_document_service, mock_vector_db
    ):
        """Test document update when document is not found."""
        # Setup mocks
        document_id = "non-existent-id"
        mock_vector_db.get_document_by_id.side_effect = DocumentNotFoundException(
            "Not found"
        )

        # Execute and assert
        with pytest.raises(DocumentNotFoundException):
            await reference_document_service.update_document(
                document_id=document_id, heading="Updated Heading"
            )

    @pytest.mark.asyncio
    async def test_update_document_content_unchanged(
        self,
        reference_document_service,
        mock_vector_db,
        mock_embedding_service,
        sample_embedding,
    ):
        """Test document update when content is unchanged."""
        # Setup mocks
        document_id = "test-doc-id"
        existing_document = {
            "heading": "Original Heading",
            "content": "Original content",
            "category": "original",
            "tags": ["original"],
            "tenant_id": "test-tenant",
            "vector": {"default": sample_embedding},
        }
        mock_vector_db.get_document_by_id.return_value = existing_document
        mock_vector_db.update_document.return_value = None

        # Execute - same content as existing
        await reference_document_service.update_document(
            document_id=document_id, content="Original content"  # Same as existing
        )

        # Assert - embedding service should not be called for unchanged content
        mock_embedding_service.embed.assert_not_called()
        mock_vector_db.update_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_success(
        self, reference_document_service, mock_vector_db
    ):
        """Test successful document deletion."""
        # Setup mocks
        document_id = "test-doc-id"
        existing_document = {"id": document_id, "heading": "Test Document"}
        mock_vector_db.get_document_by_id.return_value = existing_document
        mock_vector_db.delete_document.return_value = None

        # Execute
        await reference_document_service.delete_document(document_id)

        # Assert
        mock_vector_db.delete_document.assert_called_once_with(
            collection_name=reference_document_service.collection_name,
            document_id=document_id,
        )

    @pytest.mark.asyncio
    async def test_delete_document_not_found(
        self, reference_document_service, mock_vector_db
    ):
        """Test document deletion when document is not found."""
        # Setup mocks
        document_id = "non-existent-id"
        mock_vector_db.get_document_by_id.side_effect = DocumentNotFoundException(
            "Not found"
        )

        # Execute and assert
        with pytest.raises(DocumentNotFoundException):
            await reference_document_service.delete_document(document_id)

    @pytest.mark.asyncio
    async def test_get_document_success(
        self, reference_document_service, mock_vector_db
    ):
        """Test successful document retrieval."""
        # Setup mocks
        document_id = "test-doc-id"
        mock_document = {
            "heading": "Test Document",
            "content": "Test content",
            "category": "test",
            "tags": ["test"],
            "tenant_id": "test-tenant",
        }
        mock_vector_db.get_document_by_id.return_value = mock_document

        # Execute
        result = await reference_document_service.get_document(document_id)

        # Assert
        expected_document = {
            "id": document_id,
            "heading": "Test Document",
            "content": "Test content",
            "category": "test",
            "tags": ["test"],
            "tenant_id": "test-tenant",
        }
        assert result == expected_document

    @pytest.mark.asyncio
    async def test_get_document_not_found(
        self, reference_document_service, mock_vector_db
    ):
        """Test document retrieval when document is not found."""
        # Setup mocks
        document_id = "non-existent-id"
        mock_vector_db.get_document_by_id.return_value = None

        # Execute and assert
        with pytest.raises(DocumentNotFoundException):
            await reference_document_service.get_document(document_id)

    @pytest.mark.asyncio
    async def test_get_document_with_vector(
        self, reference_document_service, mock_vector_db, sample_embedding
    ):
        """Test document retrieval with vector included."""
        # Setup mocks
        document_id = "test-doc-id"
        mock_document = {
            "heading": "Test Document",
            "content": "Test content",
            "category": "test",
            "tags": ["test"],
            "tenant_id": "test-tenant",
            "vector": {"default": sample_embedding},
        }
        mock_vector_db.get_document_by_id.return_value = mock_document

        # Execute
        result = await reference_document_service.get_document(
            document_id, include_vector=True
        )

        # Assert
        assert "vector" in result
        assert result["vector"] == {"default": sample_embedding}

    @pytest.mark.asyncio
    async def test_search_documents_success(
        self, reference_document_service, mock_vector_db
    ):
        """Test successful document search."""
        # Setup mocks
        query = "test query"
        mock_results = {
            "documents": [
                {
                    "id": "doc1",
                    "heading": "Test Document 1",
                    "content": "Test content 1",
                    "category": "test",
                    "tags": ["test"],
                    "tenant_id": "test-tenant",
                    "score": 0.95,
                }
            ],
            "total": 1,
            "categories": ["test"],
        }
        mock_vector_db.search_documents.return_value = mock_results

        # Execute
        result = await reference_document_service.search_documents(query)

        # Assert
        assert result["documents"][0]["id"] == "doc1"
        assert result["total"] == 1
        assert result["limit"] == 10  # default limit
        mock_vector_db.search_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_documents_with_filters(
        self, reference_document_service, mock_vector_db
    ):
        """Test document search with filters."""
        # Setup mocks
        query = "test query"
        filters = {"category": "test", "tenant_id": "test-tenant"}
        mock_results = {"documents": [], "total": 0, "categories": []}
        mock_vector_db.search_documents.return_value = mock_results

        # Execute
        result = await reference_document_service.search_documents(
            query, limit=5, filters=filters
        )

        # Assert
        assert result["limit"] == 5
        mock_vector_db.search_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_documents_invalid_filters(self, reference_document_service):
        """Test document search with invalid filters."""
        # Setup
        query = "test query"
        invalid_filters = {"invalid_key": "value"}

        # Execute and assert
        with pytest.raises(ValueError) as exc_info:
            await reference_document_service.search_documents(
                query, filters=invalid_filters
            )

        assert "Invalid filter keys" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_documents_embedding_failed(
        self, reference_document_service, mock_vector_db
    ):
        """Test document search when embedding generation fails."""
        # Setup mocks
        query = "test query"
        mock_vector_db.search_documents.side_effect = EmbeddingFailedException(
            "Embedding error"
        )

        # Execute and assert
        with pytest.raises(EmbeddingFailedException):
            await reference_document_service.search_documents(query)

    @pytest.mark.asyncio
    async def test_search_documents_vector_db_failed(
        self, reference_document_service, mock_vector_db
    ):
        """Test document search when vector DB search fails."""
        # Setup mocks
        query = "test query"
        mock_vector_db.search_documents.side_effect = Exception("Vector DB error")

        # Execute and assert
        with pytest.raises(VectorDBSearchFailedException) as exc_info:
            await reference_document_service.search_documents(query)

        assert "Failed to search reference documents" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_document_embedding_failed(
        self,
        reference_document_service,
        mock_vector_db,
        mock_embedding_service,
        sample_embedding,
    ):
        """Embedding generation fails when content changes during update."""
        document_id = "doc-1"
        existing_document = {
            "heading": "H1",
            "content": "old",
            "category": "c1",
            "tags": ["t1"],
            "tenant_id": "tenant-1",
            "vector": {"default": sample_embedding},
        }
        mock_vector_db.get_document_by_id.return_value = existing_document
        mock_embedding_service.embed.side_effect = EmbeddingFailedException("fail")

        with pytest.raises(EmbeddingFailedException):
            await reference_document_service.update_document(
                document_id=document_id, content="new content"
            )

        mock_vector_db.update_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_document_vector_db_update_failed(
        self,
        reference_document_service,
        mock_vector_db,
        mock_embedding_service,
        sample_embedding,
    ):
        """Vector DB update failure path during update_document."""
        document_id = "doc-2"
        existing_document = {
            "heading": "H1",
            "content": "old",
            "category": "c1",
            "tags": ["t1"],
            "tenant_id": "tenant-1",
            "vector": {"default": sample_embedding},
        }
        mock_vector_db.get_document_by_id.return_value = existing_document
        mock_embedding_service.embed.return_value = sample_embedding
        mock_vector_db.update_document.side_effect = VectorDBUpdateFailedException(
            "update failed"
        )

        with pytest.raises(VectorDBUpdateFailedException):
            await reference_document_service.update_document(
                document_id=document_id, content="changed"
            )

    @pytest.mark.asyncio
    async def test_delete_document_vector_db_delete_failed(
        self, reference_document_service, mock_vector_db
    ):
        """Vector DB delete failure path during delete_document."""
        document_id = "doc-3"
        mock_vector_db.get_document_by_id.return_value = {"id": document_id}
        mock_vector_db.delete_document.side_effect = VectorDBDeleteFailedException(
            "delete failed"
        )

        with pytest.raises(VectorDBDeleteFailedException):
            await reference_document_service.delete_document(document_id)

    @pytest.mark.asyncio
    async def test_get_document_generic_error_raises_not_found(
        self, reference_document_service, mock_vector_db
    ):
        """Generic exceptions in vector DB get should map to DocumentNotFoundException."""  # noqa: E501
        document_id = "doc-4"
        mock_vector_db.get_document_by_id.side_effect = Exception("boom")

        with pytest.raises(DocumentNotFoundException):
            await reference_document_service.get_document(document_id)

    @pytest.mark.asyncio
    async def test_search_documents_with_document_ids_and_sorting_asc(
        self, reference_document_service, mock_vector_db
    ):
        """document_ids filter is applied and client-side sorting asc works."""
        query = "q"
        docs = {
            "documents": [
                {
                    "id": "2",
                    "heading": "Bravo",
                    "category": "x",
                    "tags": [],
                    "tenant_id": "t",
                },
                {
                    "id": "1",
                    "heading": "Alpha",
                    "category": "x",
                    "tags": [],
                    "tenant_id": "t",
                },
            ],
            "total": 2,
            "categories": ["x"],
        }
        mock_vector_db.search_documents.return_value = docs

        result = await reference_document_service.search_documents(
            query,
            document_ids=["1", "2"],
            limit=10,
            filters={"tenant_id": "t"},
            sort_by="heading",
            sort_order="asc",
        )

        # Assert sorting
        headings = [d["heading"] for d in result["documents"]]
        assert headings == ["Alpha", "Bravo"]

        # Ensure the underlying call received the combined filters with ids
        called_kwargs = mock_vector_db.search_documents.call_args.kwargs
        assert set(called_kwargs["filters"]["id"]) == {"1", "2"}
        assert called_kwargs["filters"]["tenant_id"] == "t"

    @pytest.mark.asyncio
    async def test_search_documents_sort_desc(
        self, reference_document_service, mock_vector_db
    ):
        """Client-side sorting desc works."""
        query = "q"
        docs = {
            "documents": [
                {"id": "1", "heading": "Alpha"},
                {"id": "2", "heading": "Bravo"},
            ],
            "total": 2,
            "categories": [],
        }
        mock_vector_db.search_documents.return_value = docs

        result = await reference_document_service.search_documents(
            query, sort_by="heading", sort_order="desc"
        )

        headings = [d["heading"] for d in result["documents"]]
        assert headings == ["Bravo", "Alpha"]
