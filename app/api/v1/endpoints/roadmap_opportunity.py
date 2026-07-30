from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_roadmap_opportunity_service
from app.core.roadmap.roadmap_opportunity_service import RoadmapOpportunityService
from app.exceptions.custom_exceptions import (
    EmbeddingFailedException,
    VectorDBInsertFailedException,
    VectorDBSearchFailedException,
)
from app.schemas.roadmap_opportunity import (
    RoadmapOpportunityBulkUpsert,
    RoadmapOpportunityBulkUpsertResponse,
    RoadmapOpportunityDeleteResponse,
    RoadmapOpportunitySearchRequest,
    RoadmapOpportunitySearchResponse,
    RoadmapOpportunityUpsert,
    RoadmapOpportunityUpsertResponse,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.put(
    "/{opportunity_id}",
    response_model=RoadmapOpportunityUpsertResponse,
    status_code=status.HTTP_200_OK,
    tags=["roadmap_opportunities"],
)
async def upsert_roadmap_opportunity(
    opportunity_id: UUID,
    payload: RoadmapOpportunityUpsert,
    service: RoadmapOpportunityService = Depends(get_roadmap_opportunity_service),
):
    """
    Index or re-index one opportunity's vector.

    PUT rather than POST because the Weaviate object UUID *is* the ally-be opportunity id, so
    this is idempotent by construction — no create/update split, no 409 path.

    The description is embedded but NOT stored: ally-be's Postgres is the system of record.
    """
    try:
        result = await service.upsert(
            str(opportunity_id), payload.description, payload.product_goal
        )
        return RoadmapOpportunityUpsertResponse(**result)
    except EmbeddingFailedException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate an embedding for the opportunity",
        )
    except VectorDBInsertFailedException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        logger.exception("Unexpected error upserting a roadmap opportunity")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to index the opportunity",
        )


@router.post(
    "/bulk-upsert",
    response_model=RoadmapOpportunityBulkUpsertResponse,
    status_code=status.HTTP_200_OK,
    tags=["roadmap_opportunities"],
)
async def bulk_upsert_roadmap_opportunities(
    payload: RoadmapOpportunityBulkUpsert,
    service: RoadmapOpportunityService = Depends(get_roadmap_opportunity_service),
):
    """
    Index a batch (ally-be sends 64 at a time; ~505 opportunities ≈ 8 calls).

    Answers 200 with per-item `succeeded` and `failed` lists rather than failing the whole
    request. A partially-failed batch that reports success is how the standalone app's backfill
    silently corrupted 241 rows — the caller must be able to see and retry the failures.
    """
    try:
        succeeded, failed = await service.bulk_upsert(
            [item.model_dump() for item in payload.items]
        )
        return RoadmapOpportunityBulkUpsertResponse(
            succeeded=succeeded, failed=failed
        )
    except Exception:
        logger.exception("Unexpected error in bulk upsert")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk upsert failed",
        )


@router.post(
    "/search",
    response_model=RoadmapOpportunitySearchResponse,
    status_code=status.HTTP_200_OK,
    tags=["roadmap_opportunities"],
)
async def search_roadmap_opportunities(
    payload: RoadmapOpportunitySearchRequest,
    service: RoadmapOpportunityService = Depends(get_roadmap_opportunity_service),
):
    """
    Candidate duplicates for a draft, by cosine similarity.

    Returns ids and similarities only — no text. ally-be resolves the ids against Postgres,
    which is also what filters out opportunities that have since been soft-deleted.
    """
    try:
        matches = await service.search(
            description=payload.description,
            product_goal=payload.product_goal,
            limit=payload.limit,
            threshold=payload.threshold,
        )
        return RoadmapOpportunitySearchResponse(matches=matches)
    except EmbeddingFailedException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to embed the query description",
        )
    except VectorDBSearchFailedException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector search is unavailable",
        )
    except Exception:
        logger.exception("Unexpected error searching roadmap opportunities")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed",
        )


@router.delete(
    "/{opportunity_id}",
    response_model=RoadmapOpportunityDeleteResponse,
    status_code=status.HTTP_200_OK,
    tags=["roadmap_opportunities"],
)
async def delete_roadmap_opportunity(
    opportunity_id: UUID,
    service: RoadmapOpportunityService = Depends(get_roadmap_opportunity_service),
):
    """
    Ensure an opportunity is not in the index. Called when ally-be soft-deletes or merges one away.

    Always 200, never 404: the caller's intent is "make sure this is gone", which is satisfied
    whether or not it was there. `deleted` is True when the index no longer holds the id
    (including when it never did — Weaviate does not distinguish the two) and False only when
    the delete genuinely failed, which ally-be treats as drift for the reindex sweep to heal.
    """
    deleted = await service.delete(str(opportunity_id))
    return RoadmapOpportunityDeleteResponse(
        opportunity_id=opportunity_id, deleted=deleted
    )


@router.get(
    "/{opportunity_id}",
    status_code=status.HTTP_200_OK,
    tags=["roadmap_opportunities"],
)
async def get_roadmap_opportunity(
    opportunity_id: UUID,
    service: RoadmapOpportunityService = Depends(get_roadmap_opportunity_service),
):
    """Indexed metadata for one opportunity. Reconciliation and debugging only."""
    document = await service.get(str(opportunity_id))
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Opportunity {opportunity_id} is not in the index",
        )
    return document
