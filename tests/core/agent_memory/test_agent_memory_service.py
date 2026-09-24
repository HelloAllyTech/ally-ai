"""The agent-memory index is scoped by agent and reports every lookup.

Two properties worth pinning. The `agent` filter is what keeps a Bug Hunter lookup
from surfacing a Builder lesson, so a search without one must refuse rather than
search everything. And every lookup lands in the retrieval log, because "the agent
asked its notebook and got nothing back" is exactly the fact the pipeline telemetry
(ally-be OPP-0741) wants to count.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.core.agent_memory.agent_memory_service import AgentMemoryService
from app.exceptions.custom_exceptions import VectorDBInsertFailedException

MODULE = "app.core.agent_memory.agent_memory_service"


@pytest.fixture
def service():
    return AgentMemoryService(AsyncMock(), AsyncMock())


@pytest.mark.asyncio
async def test_search_filters_by_agent_and_returns_ids_only(service):
    service.embedding_service.embed.return_value = [0.1] * 8
    service.vector_db.near_vector_search.return_value = [
        {"id": "mem-1", "agent": "bug_hunter", "similarity": 0.81},
        {"id": "mem-2", "agent": "bug_hunter", "similarity": 0.4},
    ]

    with patch(f"{MODULE}.emit_retrieval_log"):
        matches = await service.search(
            "flaky test in the scheduler suite", agent="bug_hunter", limit=5
        )

    kwargs = service.vector_db.near_vector_search.call_args.kwargs
    assert kwargs["filters"] == {"agent": "bug_hunter"}
    assert kwargs["limit"] == 5
    assert matches == [
        {"memory_id": "mem-1", "agent": "bug_hunter", "similarity": 0.81},
        {"memory_id": "mem-2", "agent": "bug_hunter", "similarity": 0.4},
    ]
    assert all("body" not in m for m in matches)


@pytest.mark.asyncio
async def test_search_refuses_without_an_agent_scope(service):
    with pytest.raises(VectorDBInsertFailedException):
        await service.search("anything", agent="")
    service.embedding_service.embed.assert_not_called()


@pytest.mark.asyncio
async def test_search_reports_itself_to_the_retrieval_log(service):
    service.embedding_service.embed.return_value = [0.1] * 8
    service.vector_db.near_vector_search.return_value = []

    with patch(f"{MODULE}.emit_retrieval_log") as emit:
        await service.search("a question", agent="bug_hunter", threshold=0.3)

    kwargs = emit.call_args.kwargs
    assert kwargs["corpus"] == "agent_memories"
    assert kwargs["consumer"] == "bug_hunter_memory"
    assert kwargs["min_similarity"] == 0.3
    assert kwargs["returned_count"] == 0
    assert kwargs["query_sensitive"] is False


@pytest.mark.asyncio
async def test_upsert_embeds_but_does_not_store_the_text(service):
    service.embedding_service.embed.return_value = [0.2] * 8

    result = await service.upsert(
        "mem-1", "ally-web tests need NX_DAEMON=false", "builder"
    )

    create_kwargs = service.vector_db.create_document.call_args.kwargs
    assert create_kwargs["document_id"] == "mem-1"
    assert create_kwargs["document_data"]["agent"] == "builder"
    assert "body" not in create_kwargs["document_data"]
    assert result["memory_id"] == "mem-1"
    assert result["text_hash"] == AgentMemoryService.hash_text(
        "ally-web tests need NX_DAEMON=false"
    )


@pytest.mark.asyncio
async def test_upsert_refuses_an_empty_entry_or_missing_agent(service):
    with pytest.raises(VectorDBInsertFailedException):
        await service.upsert("mem-1", "   ", "bug_hunter")
    with pytest.raises(VectorDBInsertFailedException):
        await service.upsert("mem-1", "a lesson", "")
