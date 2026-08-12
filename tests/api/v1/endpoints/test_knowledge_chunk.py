"""Tests for the /knowledge-chunks endpoints.

Service methods are patched on the CLASS rather than through the DI function, matching
the other endpoint test modules here: FastAPI binds
`Depends(get_knowledge_chunk_service)` at import time, so patching the dependency
function itself would have no effect on an already-bound route.
"""

from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.exceptions.custom_exceptions import (
    EmbeddingFailedException,
    VectorDBDeleteFailedException,
    VectorDBSearchFailedException,
)
from tests.api.v1.endpoints.base import BaseAPITest

SERVICE = "app.core.knowledge_base.knowledge_chunk_service.KnowledgeChunkService"

# The key AuthMiddleware actually checks, per tests/conftest.py's API__X_API_KEY.
#


class _ResolvedKeyAPITest(BaseAPITest):
    """A client whose API key is read from settings rather than hardcoded.

    AuthMiddleware compares the header against ``settings.API.X_API_KEY``. Which source
    supplies that differs by environment — in CI it is the env vars tests/conftest.py
    sets, on a developer machine a repo-root .env wins — so any literal in a test is
    right in one place and 401s in the other. Reading the same value the middleware
    reads is correct in both.
    """

    @pytest.fixture
    def client(self):
        from app.core.config import settings
        from app.main import app

        test_client = TestClient(app)
        test_client.headers.update({"x-api-key": settings.API.X_API_KEY})
        return test_client


def chunk_payload(**overrides):
    payload = {
        "chunk_id": str(uuid4()),
        "document_id": str(uuid4()),
        "document_title": "WHO mhGAP Intervention Guide",
        "chunk_index": 0,
        "text": "Ask directly about intent and plan.",
        "char_start": 0,
        "char_end": 35,
        "page_from": 44,
        "page_to": 44,
        "section_path": "Depression > Assessment",
        "source_url": "",
        "language": "en",
        "tags": ["clinical"],
        "token_count": 9,
    }
    payload.update(overrides)
    return payload


class TestBulkUpsertEndpoint(_ResolvedKeyAPITest):
    def test_bulk_upsert_success(self, client: TestClient):
        item = chunk_payload()

        with patch(f"{SERVICE}.bulk_upsert") as mock_bulk:
            mock_bulk.return_value = {
                "succeeded": [
                    {
                        "chunk_id": item["chunk_id"],
                        "text_hash": "abc123",
                        "embedding_model": "text-embedding-3-small",
                    }
                ],
                "failed": [],
            }

            response = client.post(
                "/api/v1/knowledge-chunks/bulk-upsert", json={"items": [item]}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["succeeded"][0]["chunk_id"] == item["chunk_id"]
        assert data["failed"] == []

    def test_bulk_upsert_reports_partial_failure_as_200(self, client: TestClient):
        """A partly-failed batch is a 200 with per-chunk detail, not an error status.

        ally-be advances indexed_chunk_count from `succeeded` and retries only `failed`.
        A non-2xx would make it retry the whole batch, re-indexing chunks that already
        landed.
        """
        ok, bad = chunk_payload(), chunk_payload()

        with patch(f"{SERVICE}.bulk_upsert") as mock_bulk:
            mock_bulk.return_value = {
                "succeeded": [
                    {
                        "chunk_id": ok["chunk_id"],
                        "text_hash": "h",
                        "embedding_model": "text-embedding-3-small",
                    }
                ],
                "failed": [{"chunk_id": bad["chunk_id"], "error": "empty chunk text"}],
            }

            response = client.post(
                "/api/v1/knowledge-chunks/bulk-upsert", json={"items": [ok, bad]}
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["succeeded"]) == 1
        assert data["failed"] == [
            {"chunk_id": bad["chunk_id"], "error": "empty chunk text"}
        ]

    def test_bulk_upsert_rejects_empty_items(self, client: TestClient):
        response = client.post(
            "/api/v1/knowledge-chunks/bulk-upsert", json={"items": []}
        )
        assert response.status_code == 422

    def test_bulk_upsert_rejects_blank_text(self, client: TestClient):
        """`text` has min_length=1, so a blank passage is refused at the boundary."""
        response = client.post(
            "/api/v1/knowledge-chunks/bulk-upsert",
            json={"items": [chunk_payload(text="")]},
        )
        assert response.status_code == 422


class TestSearchEndpoint(_ResolvedKeyAPITest):
    def test_search_returns_passages(self, client: TestClient):
        chunk_id = str(uuid4())

        with patch(f"{SERVICE}.search") as mock_search:
            mock_search.return_value = [
                {
                    "chunk_id": chunk_id,
                    "document_id": str(uuid4()),
                    "document_title": "WHO mhGAP Intervention Guide",
                    "chunk_index": 3,
                    "text": "Ask directly about intent.",
                    "char_start": 10,
                    "char_end": 36,
                    "page_from": 44,
                    "page_to": 44,
                    "section_path": "Depression > Assessment",
                    "source_url": "",
                    "language": "en",
                    "token_count": 9,
                    "similarity": 0.5123,
                }
            ]

            response = client.post(
                "/api/v1/knowledge-chunks/search",
                json={"query": "how do I ask about intent", "limit": 5},
            )

        assert response.status_code == 200
        passages = response.json()["passages"]
        assert passages[0]["chunk_id"] == chunk_id
        # Page and section survive to the caller — they are what a citation is rendered
        # from.
        assert passages[0]["page_from"] == 44
        assert passages[0]["section_path"] == "Depression > Assessment"
        assert passages[0]["similarity"] == 0.5123

    def test_search_embedding_failure_is_502(self, client: TestClient):
        with patch(f"{SERVICE}.search") as mock_search:
            mock_search.side_effect = EmbeddingFailedException("nope")
            response = client.post(
                "/api/v1/knowledge-chunks/search", json={"query": "q"}
            )
        assert response.status_code == 502

    def test_search_vector_db_failure_is_503(self, client: TestClient):
        with patch(f"{SERVICE}.search") as mock_search:
            mock_search.side_effect = VectorDBSearchFailedException("down")
            response = client.post(
                "/api/v1/knowledge-chunks/search", json={"query": "q"}
            )
        assert response.status_code == 503

    @pytest.mark.parametrize(
        "body",
        [
            {"query": ""},  # min_length=1
            {"query": "q", "limit": 0},  # ge=1
            {"query": "q", "limit": 51},  # le=50
            {"query": "q", "min_similarity": 1.5},  # le=1.0
        ],
    )
    def test_search_validation(self, client: TestClient, body):
        assert (
            client.post("/api/v1/knowledge-chunks/search", json=body).status_code == 422
        )


class TestDeleteDocumentChunksEndpoint(_ResolvedKeyAPITest):
    def test_delete_returns_count(self, client: TestClient):
        document_id = str(uuid4())

        with patch(f"{SERVICE}.delete_document_chunks") as mock_delete:
            mock_delete.return_value = 12
            response = client.delete(f"/api/v1/knowledge-chunks/document/{document_id}")

        assert response.status_code == 200
        assert response.json() == {"document_id": document_id, "deleted": 12}

    def test_delete_nothing_matched_is_still_200(self, client: TestClient):
        """The intent is "make sure these are gone", which zero matches satisfies."""
        with patch(f"{SERVICE}.delete_document_chunks") as mock_delete:
            mock_delete.return_value = 0
            response = client.delete(f"/api/v1/knowledge-chunks/document/{uuid4()}")

        assert response.status_code == 200
        assert response.json()["deleted"] == 0

    def test_delete_failure_is_503_not_swallowed(self, client: TestClient):
        """ally-be must not write the new generation believing the old one is gone.

        If it did, retrieval would serve two versions of the same passage and cite text
        the document no longer contains.
        """
        with patch(f"{SERVICE}.delete_document_chunks") as mock_delete:
            mock_delete.side_effect = VectorDBDeleteFailedException("failed")
            response = client.delete(f"/api/v1/knowledge-chunks/document/{uuid4()}")

        assert response.status_code == 503


class TestIdsEndpoint(_ResolvedKeyAPITest):
    def test_ids_route_is_not_shadowed_by_the_chunk_id_route(self, client: TestClient):
        """/ids must resolve as a literal, not bind to /{chunk_id}.

        FastAPI matches in declaration order, so if the parameterised route were
        declared first this would 422 on UUID validation — a routing bug that reads as a
        client error.
        """
        ids = [str(uuid4()), str(uuid4())]

        with patch(f"{SERVICE}.list_ids") as mock_list:
            mock_list.return_value = ids
            response = client.get("/api/v1/knowledge-chunks/ids?limit=2")

        assert response.status_code == 200
        data = response.json()
        assert data["ids"] == ids
        # A full page offers a cursor so the caller knows to ask again.
        assert data["next_cursor"] == ids[-1]

    def test_ids_short_page_has_no_cursor(self, client: TestClient):
        with patch(f"{SERVICE}.list_ids") as mock_list:
            mock_list.return_value = [str(uuid4())]
            response = client.get("/api/v1/knowledge-chunks/ids?limit=200")

        assert response.status_code == 200
        assert response.json()["next_cursor"] is None

    def test_ids_empty_collection(self, client: TestClient):
        with patch(f"{SERVICE}.list_ids") as mock_list:
            mock_list.return_value = []
            response = client.get("/api/v1/knowledge-chunks/ids")

        assert response.status_code == 200
        assert response.json() == {"ids": [], "next_cursor": None}


class TestGetChunkEndpoint(_ResolvedKeyAPITest):
    def test_get_chunk_success(self, client: TestClient):
        chunk_id = str(uuid4())

        with patch(f"{SERVICE}.get") as mock_get:
            mock_get.return_value = {"chunk_id": chunk_id, "text": "passage"}
            response = client.get(f"/api/v1/knowledge-chunks/{chunk_id}")

        assert response.status_code == 200
        assert response.json()["chunk_id"] == chunk_id

    def test_get_chunk_not_found(self, client: TestClient):
        with patch(f"{SERVICE}.get") as mock_get:
            mock_get.return_value = None
            response = client.get(f"/api/v1/knowledge-chunks/{uuid4()}")

        assert response.status_code == 404

    def test_get_chunk_rejects_non_uuid(self, client: TestClient):
        assert client.get("/api/v1/knowledge-chunks/not-a-uuid").status_code == 422
