from typing import List
from uuid import UUID

from pydantic import BaseModel, Field


class AgentMemoryUpsert(BaseModel):
    """
    One notebook entry to (re)index.

    `body` is used for EMBEDDING ONLY and is not persisted — see AgentMemoryProperties.
    """

    body: str = Field(..., min_length=1, max_length=600, description="Text to embed")
    agent: str = Field(
        ..., min_length=1, description="Whose notebook: bug_hunter or builder"
    )


class AgentMemoryUpsertResponse(BaseModel):
    memory_id: UUID
    text_hash: str = Field(
        ..., description="SHA-256 of the embedded text; ally-be stores this"
    )
    embedding_model: str


class AgentMemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="What the agent is asking")
    agent: str = Field(..., min_length=1, description="Whose notebook to search")
    limit: int = Field(10, ge=1, le=50)
    threshold: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine SIMILARITY for a hit. Starts low; ally-be applies "
        "its own configured floor on top and records what came back.",
    )


class AgentMemoryMatch(BaseModel):
    memory_id: UUID
    agent: str
    similarity: float = Field(..., description="Cosine similarity in [0, 1]")


class AgentMemorySearchResponse(BaseModel):
    matches: List[AgentMemoryMatch] = Field(default_factory=list)


class AgentMemoryDeleteResponse(BaseModel):
    memory_id: UUID
    deleted: bool


class AgentMemoryIdsResponse(BaseModel):
    ids: List[UUID] = Field(default_factory=list)
    next_cursor: UUID | None = None
