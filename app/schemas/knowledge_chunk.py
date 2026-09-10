"""Request/response shapes for the KnowledgeChunk vector index.

ally-be owns kb_documents/kb_document_chunks in Postgres and pushes chunks here; the
Weaviate object UUID is kb_document_chunks.id, so every write is idempotent by
construction.
"""

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.core.knowledge_base.knowledge_chunk_service import ChunkAudience


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
    is_global: bool = Field(
        False,
        description=(
            "True when the parent document is available to every organisation "
            "(ally-be kb_documents.is_global). Defaults to the CLOSED value: a caller "
            "that forgets it indexes a passage nobody retrieves, which is a visible "
            "bug, where defaulting to global would be an invisible leak"
        ),
    )
    tenant_ids: List[str] = Field(
        default_factory=list,
        description=(
            "ally-be tenant ids allowed to retrieve this passage, from "
            "kb_document_tenants. Empty alongside is_global=false means the document "
            "reaches nobody — a real state an admin can save"
        ),
    )
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


class AudienceSelector(BaseModel):
    """
    Who a retrieval is being performed for. Maps onto `ChunkAudience` in the service.

    Carried as an explicit object rather than a loose pair of optional fields so that a
    caller which omits the whole thing gets a 422 instead of the whole corpus. That is
    the structural answer to the note this collection shipped with — "retrieval that
    forgets a filter leaks, and an un-set filter is the easiest thing in the world to
    forget".
    """

    tenant_id: Optional[UUID] = Field(
        None,
        description=(
            "The organisation the asker belongs to. None means no "
            "organisation-targeted "
            "document is reachable"
        ),
    )
    include_global: bool = Field(
        True, description="Include documents available to every organisation"
    )
    unrestricted: bool = Field(
        False,
        description=(
            "Ignore targeting entirely and search the whole corpus. ADMIN TOOLING ONLY "
            "— the corpus browser and retrieval preview answer 'what is indexed', not "
            "'what may this worker see'. Never set on the WhatsApp answering path"
        ),
    )

    @model_validator(mode="after")
    def reject_empty_audience(self) -> "AudienceSelector":
        """
        An audience that can match nothing is refused at the boundary.

        Not served as an empty result: `include_global=false` with no `tenant_id` is
        almost always a caller that failed to resolve the asker's organisation and sent
        the default for the other half. Answering "the corpus does not cover that" would
        hide the bug behind a plausible reply.
        """
        if not self.unrestricted and self.tenant_id is None and not self.include_global:
            raise ValueError(
                "audience matches nothing: pass a tenant_id, include_global, or "
                "unrestricted"
            )
        return self

    def to_chunk_audience(self) -> ChunkAudience:
        """The service-layer value object, converted once rather than per endpoint."""
        if self.unrestricted:
            return ChunkAudience.unrestricted()
        return ChunkAudience(
            tenant_id=str(self.tenant_id) if self.tenant_id else None,
            include_global=self.include_global,
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
    audience: AudienceSelector = Field(
        default_factory=lambda: AudienceSelector(unrestricted=True),
        description=(
            "Whose documents to search. Defaults to UNRESTRICTED here, unlike the "
            "answering agent where it is required: this endpoint backs the admin "
            "retrieval preview, whose job is to show what is actually indexed. Pass a "
            "tenant_id to preview what one organisation's workers would retrieve"
        ),
    )


class KnowledgeChunkAudienceRequest(BaseModel):
    """
    Retarget an already-indexed document.

    Sent when an admin changes a document's organisations in ally-be. Rewrites the
    audience on the existing chunk objects in place rather than re-chunking — see
    `KnowledgeChunkService.set_document_audience`.
    """

    is_global: bool = Field(
        ..., description="Available to every organisation, ignoring tenant_ids"
    )
    tenant_ids: List[str] = Field(
        default_factory=list,
        description="Organisations that may retrieve it when is_global is false",
    )


class KnowledgeChunkAudienceResponse(BaseModel):
    document_id: UUID
    updated: int = Field(
        ...,
        description=(
            "Chunks retargeted. 0 is legitimate — a document that is queued, "
            "mid-ingest or archived has no vectors to update, and ally-be re-sends the "
            "audience with the chunks on its next ingest"
        ),
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
