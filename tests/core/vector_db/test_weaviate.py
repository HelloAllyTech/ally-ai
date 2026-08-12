"""Tests for WeaviateDB service."""

from unittest.mock import ANY, AsyncMock, MagicMock
from uuid import uuid4

import pytest
from weaviate.exceptions import (
    AuthenticationFailedException,
    WeaviateConnectionError,
    WeaviateInsertManyAllFailedError,
)

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
    async def test_list_document_ids_returns_ids_only(
        self, weaviate_db, mock_collection
    ):
        """Ids only, and the cursor is passed straight through to fetch_objects."""
        ids = [str(uuid4()) for _ in range(3)]
        result = MagicMock()
        result.objects = [MagicMock(uuid=i) for i in ids]
        mock_collection.query.fetch_objects.return_value = result
        weaviate_db.client.collections.get.return_value = mock_collection

        got = await weaviate_db.list_document_ids(
            "test_collection", limit=3, after=ids[0]
        )

        assert got == ids
        # return_properties=[] keeps a whole-collection sweep to ids on the wire.
        mock_collection.query.fetch_objects.assert_called_once_with(
            limit=3, after=ids[0], return_properties=[]
        )

    @pytest.mark.asyncio
    async def test_list_document_ids_starts_without_a_cursor(
        self, weaviate_db, mock_collection
    ):
        """The first page passes after=None rather than omitting the argument."""
        result = MagicMock()
        result.objects = []
        mock_collection.query.fetch_objects.return_value = result
        weaviate_db.client.collections.get.return_value = mock_collection

        assert await weaviate_db.list_document_ids("test_collection") == []
        mock_collection.query.fetch_objects.assert_called_once_with(
            limit=200, after=None, return_properties=[]
        )

    @pytest.mark.asyncio
    async def test_list_document_ids_failure(self, weaviate_db, mock_collection):
        """A listing failure must RAISE, never return a short list.

        A reconciliation sweep deletes on the basis of absence, so a silently truncated
        page would make live objects look orphaned.
        """
        mock_collection.query.fetch_objects.side_effect = Exception("Listing failed")
        weaviate_db.client.collections.get.return_value = mock_collection

        with pytest.raises(
            VectorDBSearchFailedException, match="Failed to list document ids"
        ):
            await weaviate_db.list_document_ids("test_collection")

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

    # ---- create_documents_bulk ----

    @staticmethod
    def _bulk_doc(doc_id, vector=(0.1, 0.2)):
        return {
            "id": str(doc_id),
            "properties": {"text": "passage"},
            "vector": list(vector),
        }

    @pytest.mark.asyncio
    async def test_create_documents_bulk_success(self, weaviate_db, mock_collection):
        """All objects written; ids come back as succeeded."""
        ids = [str(uuid4()), str(uuid4())]
        mock_collection.data.insert_many = AsyncMock(return_value=MagicMock(errors={}))
        weaviate_db.client.collections.get.return_value = mock_collection

        result = await weaviate_db.create_documents_bulk(
            "KnowledgeChunk", [self._bulk_doc(i) for i in ids]
        )

        assert result == {"succeeded": ids, "failed": []}
        # One round trip for the whole batch, which is the entire point of the method.
        mock_collection.data.insert_many.assert_awaited_once()
        submitted = mock_collection.data.insert_many.call_args.args[0]
        assert [str(o.uuid) for o in submitted] == ids

    @pytest.mark.asyncio
    async def test_create_documents_bulk_maps_errors_by_index_to_ids(
        self, weaviate_db, mock_collection
    ):
        """insert_many keys `errors` by BATCH POSITION, not by id.

        Ignoring that mapping is silent data loss: the caller reasons in chunk ids, so a
        positional error it cannot translate would be reported against the wrong chunk
        or dropped entirely.
        """
        ids = [str(uuid4()), str(uuid4()), str(uuid4())]
        mock_collection.data.insert_many = AsyncMock(
            return_value=MagicMock(errors={1: MagicMock(message="invalid property")})
        )
        weaviate_db.client.collections.get.return_value = mock_collection

        result = await weaviate_db.create_documents_bulk(
            "KnowledgeChunk", [self._bulk_doc(i) for i in ids]
        )

        assert result["succeeded"] == [ids[0], ids[2]]
        assert result["failed"] == [{"id": ids[1], "error": "invalid property"}]

    @pytest.mark.asyncio
    async def test_create_documents_bulk_rejects_missing_vector(
        self, weaviate_db, mock_collection
    ):
        """An object with no vector is refused before the request.

        These collections have vectorizer_config=none, so Weaviate will not generate
        one: the object would be accepted, stored, and then never returned by any
        similarity search. That is invisible until someone asks the question the chunk
        would have answered.
        """
        good_id, bad_id = str(uuid4()), str(uuid4())
        mock_collection.data.insert_many = AsyncMock(return_value=MagicMock(errors={}))
        weaviate_db.client.collections.get.return_value = mock_collection

        result = await weaviate_db.create_documents_bulk(
            "KnowledgeChunk",
            [
                self._bulk_doc(good_id),
                {"id": bad_id, "properties": {"text": "x"}, "vector": []},
            ],
        )

        assert result["succeeded"] == [good_id]
        assert len(result["failed"]) == 1
        assert result["failed"][0]["id"] == bad_id
        assert "vector" in result["failed"][0]["error"]
        # The valid object is still written — one bad chunk must not fail its batch.
        submitted = mock_collection.data.insert_many.call_args.args[0]
        assert [str(o.uuid) for o in submitted] == [good_id]

    @pytest.mark.asyncio
    async def test_create_documents_bulk_all_vectorless_skips_the_call(
        self, weaviate_db, mock_collection
    ):
        mock_collection.data.insert_many = AsyncMock()
        weaviate_db.client.collections.get.return_value = mock_collection

        result = await weaviate_db.create_documents_bulk(
            "KnowledgeChunk",
            [{"id": str(uuid4()), "properties": {}, "vector": []}],
        )

        assert result["succeeded"] == []
        assert len(result["failed"]) == 1
        mock_collection.data.insert_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_documents_bulk_all_failed_reports_per_object(
        self, weaviate_db, mock_collection
    ):
        """The client RAISES when every object fails, instead of returning errors.

        Left uncaught that would turn a per-object result into one whole-request
        exception, and the caller's per-chunk retry bookkeeping would silently lose the
        batch.
        """
        ids = [str(uuid4()), str(uuid4())]
        mock_collection.data.insert_many = AsyncMock(
            side_effect=WeaviateInsertManyAllFailedError("connection lost")
        )
        weaviate_db.client.collections.get.return_value = mock_collection

        result = await weaviate_db.create_documents_bulk(
            "KnowledgeChunk", [self._bulk_doc(i) for i in ids]
        )

        assert result["succeeded"] == []
        assert {f["id"] for f in result["failed"]} == set(ids)

    @pytest.mark.asyncio
    async def test_create_documents_bulk_empty_input(self, weaviate_db):
        assert await weaviate_db.create_documents_bulk("KnowledgeChunk", []) == {
            "succeeded": [],
            "failed": [],
        }

    @pytest.mark.asyncio
    async def test_create_documents_bulk_request_failure_raises(
        self, weaviate_db, mock_collection
    ):
        mock_collection.data.insert_many = AsyncMock(side_effect=Exception("boom"))
        weaviate_db.client.collections.get.return_value = mock_collection

        with pytest.raises(VectorDBInsertFailedException):
            await weaviate_db.create_documents_bulk(
                "KnowledgeChunk", [self._bulk_doc(uuid4())]
            )

    # ---- delete_by_filter ----

    @pytest.mark.asyncio
    async def test_delete_by_filter_returns_count(self, weaviate_db, mock_collection):
        mock_collection.data.delete_many = AsyncMock(
            return_value=MagicMock(successful=5, failed=0)
        )
        weaviate_db.client.collections.get.return_value = mock_collection

        deleted = await weaviate_db.delete_by_filter(
            "KnowledgeChunk", {"document_id": "doc-1"}
        )

        assert deleted == 5
        mock_collection.data.delete_many.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_by_filter_refuses_empty_filter(
        self, weaviate_db, mock_collection
    ):
        """An empty filter matches everything, so it raises before touching Weaviate.

        A caller that passed an accidentally-empty dict — a None id stringified away,
        say — deserves an exception, not a successful-looking wipe of the corpus.
        """
        mock_collection.data.delete_many = AsyncMock()
        weaviate_db.client.collections.get.return_value = mock_collection

        for empty in ({}, None, {"document_id": None}):
            with pytest.raises(ValueError):
                await weaviate_db.delete_by_filter("KnowledgeChunk", empty)

        mock_collection.data.delete_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_by_filter_failure_raises(self, weaviate_db, mock_collection):
        mock_collection.data.delete_many = AsyncMock(side_effect=Exception("boom"))
        weaviate_db.client.collections.get.return_value = mock_collection

        with pytest.raises(VectorDBDeleteFailedException):
            await weaviate_db.delete_by_filter(
                "KnowledgeChunk", {"document_id": "doc-1"}
            )
