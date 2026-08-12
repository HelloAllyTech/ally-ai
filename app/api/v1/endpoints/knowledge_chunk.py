"""KnowledgeChunk index endpoints — the write and reconciliation surface for the
WhatsApp bot's knowledge corpus.

Retrieval-plus-answering lives at /knowledge-agent. This module is the plumbing
underneath it: ally-be pushes chunks here after extracting and chunking a document,
deletes a document's chunks when it re-chunks or archives, and reconciles ids against
its own rows.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.dependencies import get_knowledge_chunk_service
from app.core.knowledge_base.knowledge_chunk_service import KnowledgeChunkService
from app.exceptions.custom_exceptions import (
    EmbeddingFailedException,
    VectorDBDeleteFailedException,
    VectorDBInsertFailedException,
    VectorDBSearchFailedException,
)
from app.schemas.knowledge_chunk import (
    KnowledgeChunkBulkUpsert,
    KnowledgeChunkBulkUpsertResponse,
    KnowledgeChunkDeleteResponse,
    KnowledgeChunkIdsResponse,
    KnowledgeChunkSearchRequest,
    KnowledgeChunkSearchResponse,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/bulk-upsert",
    response_model=KnowledgeChunkBulkUpsertResponse,
    status_code=status.HTTP_200_OK,
    tags=["knowledge_chunks"],
)
async def bulk_upsert_knowledge_chunks(
    payload: KnowledgeChunkBulkUpsert,
    service: KnowledgeChunkService = Depends(get_knowledge_chunk_service),
):
    """
    Index a batch of chunks (ally-be sends 64 at a time; a 300-page PDF is ~500 chunks ≈
    8 calls).

    Answers 200 with per-chunk `succeeded` and `failed` lists rather than failing the
    whole request. ally-be advances kb_documents.indexed_chunk_count from the former and
    retries only the latter, so a batch that reported blanket success while dropping
    chunks would leave a document permanently short of passages while displaying as
    fully indexed.

    Does NOT delete the document's previous chunks — that is a separate explicit call,
    because re-index ordering is ally-be's decision to make.
    """
    try:
        result = await service.bulk_upsert(
            [item.model_dump() for item in payload.items]
        )
        return KnowledgeChunkBulkUpsertResponse(**result)
    except VectorDBInsertFailedException as e:
        # Only raised when the whole request could not be issued; per-object problems
        # come back in `failed` above.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
    except Exception:
        logger.exception("Unexpected error in knowledge chunk bulk upsert")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk upsert failed",
        )


@router.post(
    "/search",
    response_model=KnowledgeChunkSearchResponse,
    status_code=status.HTTP_200_OK,
    tags=["knowledge_chunks"],
)
async def search_knowledge_chunks(
    payload: KnowledgeChunkSearchRequest,
    service: KnowledgeChunkService = Depends(get_knowledge_chunk_service),
):
    """
    Retrieval only — no LLM call, no answer.

    Backs the admin retrieval console, where an operator tunes topK and the similarity
    threshold and needs to see exactly what the agent would see. Separated from
    /knowledge-agent/answer so tuning retrieval costs nothing in generation tokens and
    cannot be confounded by the prompt.
    """
    try:
        passages = await service.search(
            query=payload.query,
            limit=payload.limit,
            min_similarity=payload.min_similarity,
            document_ids=(
                [str(d) for d in payload.document_ids] if payload.document_ids else None
            ),
            language=payload.language,
        )
        return KnowledgeChunkSearchResponse(passages=passages)
    except EmbeddingFailedException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to embed the query",
        )
    except VectorDBSearchFailedException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector search is unavailable",
        )
    except Exception:
        logger.exception("Unexpected error searching knowledge chunks")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search failed"
        )


# ORDER MATTERS: this must stay ABOVE GET /{chunk_id}. FastAPI matches routes in
# declaration order, so if the parameterised route came first, "/ids" would bind to
# chunk_id and fail UUID validation with a 422 that looks like a client bug.
@router.get(
    "/ids",
    response_model=KnowledgeChunkIdsResponse,
    status_code=status.HTTP_200_OK,
    tags=["knowledge_chunks"],
)
async def list_knowledge_chunk_ids(
    limit: int = Query(200, ge=1, le=1000),
    after: UUID | None = Query(
        None, description="Cursor: the last id of the previous page"
    ),
    service: KnowledgeChunkService = Depends(get_knowledge_chunk_service),
):
    """
    Enumerate indexed chunk ids so ally-be can reconcile this index against its own
    rows.

    The only read here that can surface an object ally-be has FORGOTTEN — every other
    endpoint is keyed by an id the caller already holds. Without it, a vector whose
    Postgres row was hard-deleted is undetectable and permanent, and it still occupies a
    top-k slot a live passage needed.
    """
    try:
        ids = await service.list_ids(limit=limit, after=str(after) if after else None)
        # A short page means the end of the collection; only offer a cursor when there
        # may be more.
        next_cursor = ids[-1] if len(ids) == limit else None
        return KnowledgeChunkIdsResponse(ids=ids, next_cursor=next_cursor)
    except VectorDBSearchFailedException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The vector index is unavailable",
        )
    except Exception:
        logger.exception("Unexpected error listing knowledge chunk ids")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list indexed ids",
        )


@router.delete(
    "/document/{document_id}",
    response_model=KnowledgeChunkDeleteResponse,
    status_code=status.HTTP_200_OK,
    tags=["knowledge_chunks"],
)
async def delete_document_chunks(
    document_id: UUID,
    service: KnowledgeChunkService = Depends(get_knowledge_chunk_service),
):
    """
    Remove every chunk of one document. Called when ally-be re-chunks, archives or
    deletes it.

    Always 200, never 404: the caller's intent is "make sure these are gone", which is
    satisfied whether or not any existed. `deleted` is the count actually removed, and 0
    is a legitimate result rather than an error.

    MANDATORY on re-chunk. Skipping it leaves the previous generation retrievable, which
    after an edit means the bot can answer with — and cite — text the document no longer
    contains.
    """
    try:
        deleted = await service.delete_document_chunks(str(document_id))
        return KnowledgeChunkDeleteResponse(document_id=document_id, deleted=deleted)
    except VectorDBDeleteFailedException as e:
        # Surfaced rather than swallowed: ally-be must not proceed to write the new
        # generation believing the old one is gone, or retrieval serves two versions of
        # the same passage.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
    except Exception:
        logger.exception("Unexpected error deleting document chunks")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document chunks",
        )


@router.get(
    "/{chunk_id}",
    status_code=status.HTTP_200_OK,
    tags=["knowledge_chunks"],
)
async def get_knowledge_chunk(
    chunk_id: UUID,
    service: KnowledgeChunkService = Depends(get_knowledge_chunk_service),
):
    """
    One indexed chunk. Resolves a citation from the admin conversation log, and
    debugging.
    """
    passage = await service.get(str(chunk_id))
    if passage is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk {chunk_id} is not in the index",
        )
    return passage
