import time
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List, Optional

from app.core.constants import EmbeddingConstants
from app.core.embeddings.base import BaseEmbeddingService
from app.core.retrieval_log.emitter import emit_retrieval_log
from app.core.vector_db.base import VectorDB
from app.core.vector_db.constants import VectorDBCollectionNames
from app.exceptions.custom_exceptions import (
    EmbeddingFailedException,
    VectorDBDeleteFailedException,
    VectorDBInsertFailedException,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AgentMemoryService:
    """
    Semantic search over an agent's notebook of lessons.

    A DERIVED INDEX over ally-be's `agent_memories` table, built like
    RoadmapOpportunityService and with the same consequences: the Weaviate object UUID
    is the ally-be row id (idempotent upserts, no 409 path), the entry text is embedded
    but never stored (ally-be holds it and applies repo scope, status and pins on the
    ids that come back), and drift is healed by ally-be re-pushing rows.

    One difference is deliberate: `agent` is a REQUIRED filter on search. Bug Hunter's
    lookups must never surface a Builder lesson, and a scope the caller could forget is
    not a boundary.
    """

    def __init__(
        self, vector_db: VectorDB, embedding_service: BaseEmbeddingService
    ) -> None:
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.collection_name = VectorDBCollectionNames.AGENT_MEMORIES

    @staticmethod
    def hash_text(text: str) -> str:
        """SHA-256 of the embedded text; ally-be stores it to detect a stale vector."""
        return sha256(text.encode("utf-8")).hexdigest()

    async def upsert(self, memory_id: str, body: str, agent: str) -> Dict[str, Any]:
        """
        Index (or re-index) one entry. Delete-then-insert, because the object id is the
        row id and a blind delete followed by an insert is exactly idempotent.
        """
        text = body.strip()
        if not text:
            raise VectorDBInsertFailedException("Cannot embed an empty memory entry")
        if not agent:
            raise VectorDBInsertFailedException("A memory entry needs an agent scope")

        try:
            vector = await self.embedding_service.embed(text)
        except Exception as e:
            logger.exception(
                f"Embedding failed for memory {memory_id}: {type(e).__name__}"
            )
            raise EmbeddingFailedException("Failed to embed the memory entry")

        text_hash = self.hash_text(text)
        properties = {
            "agent": agent,
            "text_hash": text_hash,
            "embedding_model": EmbeddingConstants.MODEL,
            "embedded_at": datetime.now(timezone.utc),
        }

        try:
            await self.vector_db.delete_document(self.collection_name, memory_id)
        except VectorDBDeleteFailedException:
            # Expected on a first insert — the object does not exist yet.
            pass

        await self.vector_db.create_document(
            collection_name=self.collection_name,
            document_data=properties,
            vector=vector,
            document_id=memory_id,
        )

        return {
            "memory_id": memory_id,
            "text_hash": text_hash,
            "embedding_model": EmbeddingConstants.MODEL,
        }

    async def search(
        self,
        query: str,
        agent: str,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        The most similar entries in one agent's notebook.

        Returns ids and similarities only. ally-be narrows the ids to the repo in play
        and to active rows, so this asks for a few more than the caller will show.
        """
        text = query.strip()
        if not text:
            return []
        if not agent:
            raise VectorDBInsertFailedException("Memory search needs an agent scope")

        try:
            vector = await self.embedding_service.embed(text)
        except Exception as e:
            logger.exception(f"Query embedding failed: {type(e).__name__}")
            raise EmbeddingFailedException("Failed to embed the memory query")

        started_at = time.monotonic()
        hits = await self.vector_db.near_vector_search(
            collection_name=self.collection_name,
            vector=vector,
            limit=limit,
            min_similarity=threshold,
            filters={"agent": agent},
        )

        # Every lookup reports itself, so the relevance threshold has a distribution
        # behind it — and so "the agent asked its notebook and got nothing back" is a
        # fact somebody can read, not a silence.
        emit_retrieval_log(
            corpus="agent_memories",
            consumer=f"{agent}_memory",
            query=text,
            min_similarity=float(threshold),
            requested_limit=int(limit),
            returned_count=len(hits),
            latency_ms=int((time.monotonic() - started_at) * 1000),
            hits=[
                {
                    "chunk_id": hit.get("id"),
                    "document_id": hit.get("id"),
                    "similarity": hit.get("similarity"),
                }
                for hit in hits
            ],
            # An agent's own question about a codebase; no end-user text.
            query_sensitive=False,
        )

        return [
            {
                "memory_id": hit["id"],
                "agent": hit.get("agent") or agent,
                "similarity": round(float(hit.get("similarity", 0.0)), 4),
            }
            for hit in hits
        ]

    async def delete(self, memory_id: str) -> bool:
        """Ensure an entry is not in the index. Idempotent; False means the delete
        failed."""
        try:
            await self.vector_db.delete_document(self.collection_name, memory_id)
            return True
        except VectorDBDeleteFailedException:
            logger.warning(
                f"Failed to delete memory {memory_id} from the index; it may still "
                f"surface until ally-be re-syncs"
            )
            return False

    async def list_ids(
        self, limit: int = 200, after: Optional[str] = None
    ) -> List[str]:
        """One page of indexed ids, for reconciliation by ally-be."""
        return await self.vector_db.list_document_ids(
            self.collection_name, limit=limit, after=after
        )
