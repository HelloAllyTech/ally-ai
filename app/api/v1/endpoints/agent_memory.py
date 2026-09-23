from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.agent_memory.agent_memory_service import AgentMemoryService
from app.core.dependencies import get_agent_memory_service
from app.exceptions.custom_exceptions import (
    EmbeddingFailedException,
    VectorDBInsertFailedException,
    VectorDBSearchFailedException,
)
from app.schemas.agent_memory import (
    AgentMemoryDeleteResponse,
    AgentMemoryIdsResponse,
    AgentMemorySearchRequest,
    AgentMemorySearchResponse,
    AgentMemoryUpsert,
    AgentMemoryUpsertResponse,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.put(
    "/{memory_id}",
    response_model=AgentMemoryUpsertResponse,
    status_code=status.HTTP_200_OK,
    tags=["agent_memories"],
)
async def upsert_agent_memory(
    memory_id: UUID,
    payload: AgentMemoryUpsert,
    service: AgentMemoryService = Depends(get_agent_memory_service),
):
    """Index or re-index one notebook entry. PUT: the object id is the ally-be row
    id."""
    try:
        result = await service.upsert(str(memory_id), payload.body, payload.agent)
        return AgentMemoryUpsertResponse(**result)
    except EmbeddingFailedException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate an embedding for the memory entry",
        )
    except VectorDBInsertFailedException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        logger.exception("Unexpected error upserting an agent memory")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to index the memory entry",
        )


@router.post(
    "/search",
    response_model=AgentMemorySearchResponse,
    status_code=status.HTTP_200_OK,
    tags=["agent_memories"],
)
async def search_agent_memories(
    payload: AgentMemorySearchRequest,
    service: AgentMemoryService = Depends(get_agent_memory_service),
):
    """Nearest entries in one agent's notebook — ids and similarities only."""
    try:
        matches = await service.search(
            query=payload.query,
            agent=payload.agent,
            limit=payload.limit,
            threshold=payload.threshold,
        )
        return AgentMemorySearchResponse(matches=matches)
    except EmbeddingFailedException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to embed the memory query",
        )
    except VectorDBInsertFailedException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except VectorDBSearchFailedException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector search is unavailable",
        )
    except Exception:
        logger.exception("Unexpected error searching agent memories")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed",
        )


@router.delete(
    "/{memory_id}",
    response_model=AgentMemoryDeleteResponse,
    status_code=status.HTTP_200_OK,
    tags=["agent_memories"],
)
async def delete_agent_memory(
    memory_id: UUID,
    service: AgentMemoryService = Depends(get_agent_memory_service),
):
    """Ensure an entry is not in the index. Always 200; `deleted` False means it
    failed."""
    deleted = await service.delete(str(memory_id))
    return AgentMemoryDeleteResponse(memory_id=memory_id, deleted=deleted)


# ORDER MATTERS: stays above any GET /{memory_id} that may be added later.
@router.get(
    "/ids",
    response_model=AgentMemoryIdsResponse,
    status_code=status.HTTP_200_OK,
    tags=["agent_memories"],
)
async def list_agent_memory_ids(
    limit: int = Query(200, ge=1, le=1000),
    after: UUID | None = Query(
        None, description="Cursor: last id of the previous page"
    ),
    service: AgentMemoryService = Depends(get_agent_memory_service),
):
    """Enumerate indexed ids so ally-be can reconcile the index against its rows."""
    try:
        ids = await service.list_ids(limit=limit, after=str(after) if after else None)
        next_cursor = ids[-1] if len(ids) == limit else None
        return AgentMemoryIdsResponse(ids=ids, next_cursor=next_cursor)
    except VectorDBSearchFailedException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The vector index is unavailable",
        )
    except Exception:
        logger.exception("Unexpected error listing agent memory ids")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list indexed ids",
        )
