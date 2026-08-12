"""Request/response shapes for the KnowledgeChunk vector index.

ally-be owns kb_documents/kb_document_chunks in Postgres and pushes chunks here; the
Weaviate object UUID is kb_document_chunks.id, so every write is idempotent by
construction.
"""

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class KnowledgeChunkItem(BaseModel):
    """
    One chunk to index. Mirrors a kb_document_chunks row plus its parent's citation
    metadata.
    """

    chunk_id: UUID = Field(
        ..., description="ally-be kb_document_chunks.id; becomes the object UUID"
    )
    document_id: UUID = Field(..., description="ally-be kb_documents.id")
    document_title: str = Field(
        "",
        description=(
            "Denormalised so a citation renders from the hit alone, with no back-call"
        ),
    )
    chunk_index: int = Field(
        0, ge=0, description="Zero-based position within the document"
    )
    text: str = Field(..., min_length=1, description="The passage; embedded AND stored")
    char_start: int = Field(0, ge=0, description="Offset into kb_documents.raw_text")
    char_end: int = Field(0, ge=0, description="End offset into kb_documents.raw_text")
    page_from: int = Field(
        0, ge=0, description="First source page; 0 when the format has no pages"
    )
    page_to: int = Field(0, ge=0, description="Last source page; 0 when not paginated")
    section_path: str = Field(
        "", description="Heading trail, cited when there is no page number"
    )
    source_url: str = Field("", description="Original URL when fetched from one")
    language: str = Field("", description="BCP-47 tag of this passage")
    tags: List[str] = Field(default_factory=list)
    token_count: int = Field(
        0, ge=0, description="Tokens in `text`, so the agent can budget context"
    )


class KnowledgeChunkBulkUpsert(BaseModel):
    items: List[KnowledgeChunkItem] = Field(..., min_length=1)


class KnowledgeChunkUpsertResult(BaseModel):
    chunk_id: UUID
    text_hash: str = Field(
        ...,
        description=(
            "SHA-256 of the embedded text; ally-be stores this to detect a stale vector"
        ),
    )
    embedding_model: str


class KnowledgeChunkFailure(BaseModel):
    chunk_id: str = Field(
        ...,
        description=(
            "Plain string, not UUID: a malformed id is itself a reportable failure "
            "and must survive into the response rather than 422-ing the whole batch"
        ),
    )
    error: str


class KnowledgeChunkBulkUpsertResponse(BaseModel):
    """
    Per-chunk outcomes, deliberately split.

    ally-be advances kb_documents.indexed_chunk_count from `succeeded` and retries only
    what is in `failed`. A batch reporting blanket success while dropping chunks would
    leave a document permanently short of passages while displaying as fully indexed —
    the failure only becomes visible when a worker asks the question those chunks would
    have answered.
    """

    succeeded: List[KnowledgeChunkUpsertResult] = Field(default_factory=list)
    failed: List[KnowledgeChunkFailure] = Field(default_factory=list)


class KnowledgeChunkDeleteResponse(BaseModel):
    document_id: UUID
    deleted: int = Field(
        ..., description="Chunks removed. 0 is legitimate — nothing matched."
    )


class KnowledgeChunkSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(8, ge=1, le=50)
    min_similarity: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine SIMILARITY floor (not distance). Deliberately permissive — "
            "whether the corpus actually covers a question is a separate, higher "
            "threshold in the agent."
        ),
    )
    document_ids: Optional[List[UUID]] = Field(
        None, description="Restrict to these documents; None searches the whole corpus"
    )
    language: Optional[str] = Field(
        None, description="Restrict to one passage language"
    )


class KnowledgeChunkPassage(BaseModel):
    """A retrieved passage, carrying everything a citation needs."""

    chunk_id: UUID
    document_id: str
    document_title: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_from: int
    page_to: int
    section_path: str
    source_url: str
    language: str
    token_count: int
    similarity: float = Field(..., description="Cosine similarity in [0, 1]")


class KnowledgeChunkSearchResponse(BaseModel):
    passages: List[KnowledgeChunkPassage] = Field(default_factory=list)


class KnowledgeChunkIdsResponse(BaseModel):
    """
    One page of indexed chunk ids, for ally-be's reconciliation sweep.

    Cursor-paginated rather than offset-paginated: offset paging over a collection being
    written to can skip objects, and a sweep that skips an id under-reports drift while
    looking like it passed. `next_cursor` is None when the page was short, i.e. the end
    of the collection.
    """

    ids: List[UUID] = Field(default_factory=list)
    next_cursor: UUID | None = Field(
        None,
        description=(
            "Pass as `after` to fetch the next page; None means this was the last"
        ),
    )
