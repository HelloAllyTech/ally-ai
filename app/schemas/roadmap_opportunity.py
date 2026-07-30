from typing import List
from uuid import UUID

from pydantic import BaseModel, Field


class RoadmapOpportunityUpsert(BaseModel):
    """
    One opportunity to (re)index.

    `description` is used for EMBEDDING ONLY and is not persisted — see
    RoadmapOpportunityProperties for why.
    """

    description: str = Field(..., min_length=1, description="Text to embed")
    product_goal: str = Field(
        ..., description="Product goal name, stored so a search can be scoped to one goal"
    )


class RoadmapOpportunityUpsertResponse(BaseModel):
    opportunity_id: UUID
    text_hash: str = Field(
        ...,
        description="SHA-256 of the embedded text; ally-be stores this to detect a stale vector",
    )
    embedding_model: str


class RoadmapOpportunityBulkItem(RoadmapOpportunityUpsert):
    opportunity_id: UUID = Field(..., description="ally-be roadmap_opportunities.id")


class RoadmapOpportunityBulkUpsert(BaseModel):
    items: List[RoadmapOpportunityBulkItem] = Field(..., min_length=1)


class RoadmapOpportunityBulkFailure(BaseModel):
    opportunity_id: UUID
    error: str


class RoadmapOpportunityBulkUpsertResponse(BaseModel):
    """
    Per-item outcomes, deliberately split.

    A batch endpoint that reports overall success while silently dropping items is exactly how
    the standalone roadmap app's own backfill wrote 241 fallback classifications to a file
    labelled "Done. 241 classified." The caller MUST be able to see which items failed.
    """

    succeeded: List[RoadmapOpportunityUpsertResponse] = Field(default_factory=list)
    failed: List[RoadmapOpportunityBulkFailure] = Field(default_factory=list)


class RoadmapOpportunitySearchRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Draft text to match against")
    product_goal: str | None = Field(
        None, description="Optionally restrict candidates to one product goal"
    )
    limit: int = Field(20, ge=1, le=100)
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine SIMILARITY (not distance) for a candidate. The standalone app used "
            "0.5 against Voyage voyage-3-large; re-calibrate for text-embedding-3-small."
        ),
    )


class RoadmapOpportunityMatch(BaseModel):
    opportunity_id: UUID
    product_goal: str
    similarity: float = Field(..., description="Cosine similarity in [0, 1]")


class RoadmapOpportunitySearchResponse(BaseModel):
    matches: List[RoadmapOpportunityMatch] = Field(default_factory=list)


class RoadmapOpportunityDeleteResponse(BaseModel):
    opportunity_id: UUID
    deleted: bool = Field(
        ...,
        description=(
            "True when the index no longer holds this id, including when it never did — "
            "Weaviate does not distinguish the two. False means the delete failed."
        ),
    )


class RoadmapOpportunityIdsResponse(BaseModel):
    """
    One page of indexed ids, for ally-be's reconciliation sweep.

    `next_cursor` is None when the page was not full, i.e. the end of the collection. Paging is
    by CURSOR rather than offset because offset paging over a collection being written to can skip
    objects — and a sweep that skips an id would under-report drift while looking like it passed.
    """

    ids: List[UUID] = Field(default_factory=list)
    next_cursor: UUID | None = Field(
        None,
        description="Pass as `after` to fetch the next page; None means this was the last",
    )
