"""Tests for KnowledgeChunkService.

The behaviours under test are the ones whose failure would be SILENT in production: a
chunk that gets indexed with the wrong neighbour's vector, a partial batch reported as a
total success, or a stale generation left retrievable after a re-chunk. Each of those
looks fine until someone asks the question it ruins.
"""

from unittest.mock import AsyncMock

import pytest

from app.core.constants import EmbeddingConstants
from app.core.knowledge_base.knowledge_chunk_service import KnowledgeChunkService
from app.core.vector_db.constants import VectorDBCollectionNames
from app.exceptions.custom_exceptions import EmbeddingFailedException

CHUNK_A = "11111111-1111-1111-1111-111111111111"
CHUNK_B = "22222222-2222-2222-2222-222222222222"
DOC_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def make_item(chunk_id: str, text: str, **overrides):
    item = {
        "chunk_id": chunk_id,
        "document_id": DOC_ID,
        "document_title": "Managing Suicidal Ideation",
        "chunk_index": 0,
        "text": text,
        "char_start": 0,
        "char_end": len(text),
        "page_from": 44,
        "page_to": 44,
        "section_path": "Chapter 3 > Risk assessment",
        "source_url": "",
        "language": "en",
        "tags": ["clinical"],
        "token_count": 12,
    }
    item.update(overrides)
    return item


class TestKnowledgeChunkService:
    @pytest.fixture
    def vector_db(self):
        db = AsyncMock()
        db.create_documents_bulk.return_value = {"succeeded": [], "failed": []}
        return db

    @pytest.fixture
    def embedding_service(self):
        return AsyncMock()

    @pytest.fixture
    def service(self, vector_db, embedding_service):
        return KnowledgeChunkService(vector_db, embedding_service)

    # ---- bulk_upsert ----

    @pytest.mark.asyncio
    async def test_bulk_upsert_writes_expected_properties(
        self, service, vector_db, embedding_service
    ):
        embedding_service.embed_many.return_value = [[0.1, 0.2]]
        vector_db.create_documents_bulk.return_value = {
            "succeeded": [CHUNK_A],
            "failed": [],
        }

        result = await service.bulk_upsert(
            [make_item(CHUNK_A, "Ask directly about intent.")]
        )

        assert [r["chunk_id"] for r in result["succeeded"]] == [CHUNK_A]
        assert result["failed"] == []

        collection, documents = vector_db.create_documents_bulk.call_args.args
        assert collection == VectorDBCollectionNames.KNOWLEDGE_CHUNKS
        assert len(documents) == 1
        written = documents[0]
        # The object UUID must BE the ally-be chunk id, or citations cannot resolve
        # back.
        assert written["id"] == CHUNK_A
        assert written["vector"] == [0.1, 0.2]
        props = written["properties"]
        assert props["text"] == "Ask directly about intent."
        assert props["document_id"] == DOC_ID
        assert props["page_from"] == 44
        assert props["section_path"] == "Chapter 3 > Risk assessment"
        assert props["embedding_model"] == EmbeddingConstants.MODEL
        assert props["text_hash"] == service.hash_text("Ask directly about intent.")

    @pytest.mark.asyncio
    async def test_bulk_upsert_pairs_each_chunk_with_its_own_vector(
        self, service, vector_db, embedding_service
    ):
        """Order in must equal order out.

        A misalignment here is the worst bug this class can have: the chunk is
        retrievable, the answer is plausible, and the citation points at text that never
        matched the question.
        """
        embedding_service.embed_many.return_value = [[1.0], [2.0]]
        vector_db.create_documents_bulk.return_value = {
            "succeeded": [CHUNK_A, CHUNK_B],
            "failed": [],
        }

        await service.bulk_upsert(
            [make_item(CHUNK_A, "first passage"), make_item(CHUNK_B, "second passage")]
        )

        _, documents = vector_db.create_documents_bulk.call_args.args
        by_id = {d["id"]: d for d in documents}
        assert by_id[CHUNK_A]["vector"] == [1.0]
        assert by_id[CHUNK_A]["properties"]["text"] == "first passage"
        assert by_id[CHUNK_B]["vector"] == [2.0]
        assert by_id[CHUNK_B]["properties"]["text"] == "second passage"

    @pytest.mark.asyncio
    async def test_bulk_upsert_fails_blank_chunk_individually(
        self, service, vector_db, embedding_service
    ):
        """A blank chunk is reported, not allowed to poison the batch."""
        embedding_service.embed_many.return_value = [[0.5]]
        vector_db.create_documents_bulk.return_value = {
            "succeeded": [CHUNK_B],
            "failed": [],
        }

        result = await service.bulk_upsert(
            [make_item(CHUNK_A, "   "), make_item(CHUNK_B, "real text")]
        )

        assert [f["chunk_id"] for f in result["failed"]] == [CHUNK_A]
        assert [s["chunk_id"] for s in result["succeeded"]] == [CHUNK_B]
        # The blank one is never sent for embedding.
        embedding_service.embed_many.assert_awaited_once_with(["real text"])

    @pytest.mark.asyncio
    async def test_bulk_upsert_reports_embedding_failure_per_chunk(
        self, service, vector_db, embedding_service
    ):
        """
        A whole-batch embedding failure is reported per chunk, so retry bookkeeping
        stays exact.
        """
        embedding_service.embed_many.side_effect = RuntimeError("boom")

        result = await service.bulk_upsert(
            [make_item(CHUNK_A, "a"), make_item(CHUNK_B, "b")]
        )

        assert result["succeeded"] == []
        assert {f["chunk_id"] for f in result["failed"]} == {CHUNK_A, CHUNK_B}
        assert all("embedding failed" in f["error"] for f in result["failed"])
        vector_db.create_documents_bulk.assert_not_called()

    @pytest.mark.asyncio
    async def test_bulk_upsert_rejects_embedding_count_mismatch(
        self, service, vector_db, embedding_service
    ):
        """Never zip mismatched lists — that attaches the wrong vector to a passage."""
        embedding_service.embed_many.return_value = [[1.0]]  # asked for 2

        result = await service.bulk_upsert(
            [make_item(CHUNK_A, "a"), make_item(CHUNK_B, "b")]
        )

        assert result["succeeded"] == []
        assert all("count mismatch" in f["error"] for f in result["failed"])
        vector_db.create_documents_bulk.assert_not_called()

    @pytest.mark.asyncio
    async def test_bulk_upsert_surfaces_partial_write_failure(
        self, service, vector_db, embedding_service
    ):
        """Only chunks the index confirmed are reported as succeeded."""
        embedding_service.embed_many.return_value = [[1.0], [2.0]]
        vector_db.create_documents_bulk.return_value = {
            "succeeded": [CHUNK_A],
            "failed": [{"id": CHUNK_B, "error": "duplicate uuid"}],
        }

        result = await service.bulk_upsert(
            [make_item(CHUNK_A, "a"), make_item(CHUNK_B, "b")]
        )

        assert [s["chunk_id"] for s in result["succeeded"]] == [CHUNK_A]
        assert result["failed"] == [{"chunk_id": CHUNK_B, "error": "duplicate uuid"}]

    @pytest.mark.asyncio
    async def test_bulk_upsert_all_blank_short_circuits(
        self, service, vector_db, embedding_service
    ):
        result = await service.bulk_upsert([make_item(CHUNK_A, "")])

        assert result["succeeded"] == []
        assert len(result["failed"]) == 1
        embedding_service.embed_many.assert_not_called()
        vector_db.create_documents_bulk.assert_not_called()

    # ---- delete_document_chunks ----

    @pytest.mark.asyncio
    async def test_delete_document_chunks_filters_by_document(self, service, vector_db):
        vector_db.delete_by_filter.return_value = 7

        deleted = await service.delete_document_chunks(DOC_ID)

        assert deleted == 7
        vector_db.delete_by_filter.assert_awaited_once_with(
            VectorDBCollectionNames.KNOWLEDGE_CHUNKS, {"document_id": DOC_ID}
        )

    # ---- search ----

    @pytest.mark.asyncio
    async def test_search_uses_near_vector_not_search_documents(
        self, service, vector_db, embedding_service
    ):
        """Retrieval must not route through search_documents.

        That method applies REFERENCE_DOCUMENTS_DISTANCE_THRESHOLD and a category
        aggregation built for the helpline corpus — an already-shipped feature whose
        behaviour must not shift here.
        """
        embedding_service.embed.return_value = [0.3]
        vector_db.near_vector_search.return_value = [
            {
                "id": CHUNK_A,
                "similarity": 0.512345,
                "document_id": DOC_ID,
                "document_title": "Guide",
                "chunk_index": 2,
                "text": "passage",
                "page_from": 44,
                "page_to": 44,
            }
        ]

        passages = await service.search("how do I ask about intent", limit=5)

        vector_db.search_documents.assert_not_called()
        kwargs = vector_db.near_vector_search.call_args.kwargs
        assert kwargs["collection_name"] == VectorDBCollectionNames.KNOWLEDGE_CHUNKS
        assert kwargs["vector"] == [0.3]
        assert kwargs["limit"] == 5
        assert passages[0]["chunk_id"] == CHUNK_A
        assert passages[0]["similarity"] == 0.5123  # rounded to 4dp
        assert passages[0]["page_from"] == 44

    @pytest.mark.asyncio
    async def test_search_overfetches_when_filtering_by_document(
        self, service, vector_db, embedding_service
    ):
        """A document restriction must not be able to starve the result set.

        near_vector_search supports property equality only, so document_ids is applied
        after the fact — asking for exactly `limit` first would let unrelated documents
        consume every slot.
        """
        embedding_service.embed.return_value = [0.1]
        vector_db.near_vector_search.return_value = [
            {"id": CHUNK_A, "similarity": 0.6, "document_id": "other-doc"},
            {"id": CHUNK_B, "similarity": 0.55, "document_id": DOC_ID},
        ]

        passages = await service.search("q", limit=2, document_ids=[DOC_ID])

        assert vector_db.near_vector_search.call_args.kwargs["limit"] == 8  # 2 * 4
        assert [p["chunk_id"] for p in passages] == [CHUNK_B]

    @pytest.mark.asyncio
    async def test_search_blank_query_returns_nothing(
        self, service, vector_db, embedding_service
    ):
        assert await service.search("   ") == []
        embedding_service.embed.assert_not_called()
        vector_db.near_vector_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_wraps_embedding_failure(
        self, service, vector_db, embedding_service
    ):
        embedding_service.embed.side_effect = RuntimeError("boom")

        with pytest.raises(EmbeddingFailedException):
            await service.search("a question")
