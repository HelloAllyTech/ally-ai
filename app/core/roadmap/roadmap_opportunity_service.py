from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List, Optional, Tuple

from app.core.constants import EmbeddingConstants
from app.core.embeddings.base import BaseEmbeddingService
from app.core.vector_db.base import VectorDB
from app.core.vector_db.constants import VectorDBCollectionNames
from app.exceptions.custom_exceptions import (
    EmbeddingFailedException,
    VectorDBDeleteFailedException,
    VectorDBInsertFailedException,
    VectorDBSearchFailedException,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RoadmapOpportunityService:
    """
    Semantic duplicate detection for the Ally Product Roadmap board.

    This collection is a DERIVED INDEX. ally-be's Postgres
    (roadmap_opportunities.description) is the system of record; here we store only the vector
    plus enough metadata to filter and to detect staleness. Consequences worth knowing:

      * The Weaviate object UUID IS roadmap_opportunities.id, so every upsert is idempotent by
        construction — there is no create-vs-update decision and no 409 path.
      * Nothing here can answer "what does opportunity X say?" That is on purpose. ally-be runs
        the LLM confirmation step and already holds every description.
      * Drift is possible (a failed delete leaves a phantom candidate), so ally-be re-validates
        every candidate against live Postgres rows and can rebuild the whole collection with
        POST /api/v1/product-roadmap/admin/reindex.
    """

    def __init__(
        self, vector_db: VectorDB, embedding_service: BaseEmbeddingService
    ) -> None:
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.collection_name = VectorDBCollectionNames.ROADMAP_OPPORTUNITIES

    @staticmethod
    def hash_text(text: str) -> str:
        """SHA-256 of the embedded text. ally-be stores this to detect a stale vector."""
        return sha256(text.encode("utf-8")).hexdigest()

    async def upsert(
        self, opportunity_id: str, description: str, product_goal: str
    ) -> Dict[str, Any]:
        """
        Index (or re-index) one opportunity.

        Delete-then-insert rather than an update: Weaviate's update path would need a
        get-then-decide round trip, and because the object UUID is the opportunity id, a blind
        delete followed by an insert is both simpler and exactly idempotent. A delete of a
        non-existent object is not an error here.
        """
        text = description.strip()
        if not text:
            raise VectorDBInsertFailedException("Cannot embed an empty description")

        try:
            vector = await self.embedding_service.embed(text)
        except Exception as e:
            logger.exception(f"Embedding failed for {opportunity_id}: {type(e).__name__}")
            raise EmbeddingFailedException("Failed to embed opportunity description")

        text_hash = self.hash_text(text)
        properties = {
            "product_goal": product_goal,
            "text_hash": text_hash,
            "embedding_model": EmbeddingConstants.MODEL,
            "embedded_at": datetime.now(timezone.utc),
        }

        try:
            await self.vector_db.delete_document(self.collection_name, opportunity_id)
        except VectorDBDeleteFailedException:
            # Expected on a first insert — the object does not exist yet.
            pass

        await self.vector_db.create_document(
            collection_name=self.collection_name,
            document_data=properties,
            vector=vector,
            document_id=opportunity_id,
        )

        return {
            "opportunity_id": opportunity_id,
            "text_hash": text_hash,
            "embedding_model": EmbeddingConstants.MODEL,
        }

    async def bulk_upsert(
        self, items: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        """
        Index a batch, embedding all descriptions in ONE embed_many call.

        Returns (succeeded, failed) SEPARATELY and never raises for a per-item problem. The
        caller has to be able to see partial failure: reporting a batch as successful while
        silently dropping items is precisely how the standalone roadmap app's backfill wrote 241
        fallback classifications and logged "Done. 241 classified."
        """
        succeeded: List[Dict[str, Any]] = []
        failed: List[Dict[str, str]] = []

        prepared = [
            (str(item["opportunity_id"]), item["description"].strip(), item["product_goal"])
            for item in items
        ]

        # Blank descriptions can never be embedded; fail them individually rather than poisoning
        # the whole batch.
        embeddable = [(oid, text, goal) for oid, text, goal in prepared if text]
        for oid, text, _goal in prepared:
            if not text:
                failed.append({"opportunity_id": oid, "error": "empty description"})

        if not embeddable:
            return succeeded, failed

        try:
            vectors = await self.embedding_service.embed_many([t for _, t, _ in embeddable])
        except Exception as e:
            # A whole-batch embedding failure is reported per item so the caller's counters and
            # retry bookkeeping stay accurate.
            logger.exception(f"Batch embedding failed: {type(e).__name__}")
            for oid, _text, _goal in embeddable:
                failed.append(
                    {"opportunity_id": oid, "error": f"embedding failed: {type(e).__name__}"}
                )
            return succeeded, failed

        if len(vectors) != len(embeddable):
            # Never silently pair up mismatched lists — that would attach the wrong vector to an
            # opportunity, which is worse than failing.
            logger.error(
                f"Embedding count mismatch: asked for {len(embeddable)}, got {len(vectors)}"
            )
            for oid, _text, _goal in embeddable:
                failed.append({"opportunity_id": oid, "error": "embedding count mismatch"})
            return succeeded, failed

        now = datetime.now(timezone.utc)
        for (oid, text, goal), vector in zip(embeddable, vectors):
            text_hash = self.hash_text(text)
            try:
                try:
                    await self.vector_db.delete_document(self.collection_name, oid)
                except VectorDBDeleteFailedException:
                    pass

                await self.vector_db.create_document(
                    collection_name=self.collection_name,
                    document_data={
                        "product_goal": goal,
                        "text_hash": text_hash,
                        "embedding_model": EmbeddingConstants.MODEL,
                        "embedded_at": now,
                    },
                    vector=vector,
                    document_id=oid,
                )
                succeeded.append(
                    {
                        "opportunity_id": oid,
                        "text_hash": text_hash,
                        "embedding_model": EmbeddingConstants.MODEL,
                    }
                )
            except Exception as e:
                logger.exception(f"Upsert failed for {oid}: {type(e).__name__}")
                failed.append({"opportunity_id": oid, "error": type(e).__name__})

        return succeeded, failed

    async def search(
        self,
        description: str,
        product_goal: Optional[str] = None,
        limit: int = 20,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar opportunities to a draft.

        `threshold` is a cosine SIMILARITY floor. Note the standalone app calibrated 0.5 against
        Voyage voyage-3-large at 1024 dimensions; this runs OpenAI text-embedding-3-small at
        1536, so the value needs re-calibrating against real data before it is trusted.

        Voyage also distinguished input_type document vs query; OpenAI has no such distinction,
        so the same embedding path serves both sides here. In practice that makes stored items
        and drafts directly comparable, which is what we want.
        """
        text = description.strip()
        if not text:
            return []

        try:
            vector = await self.embedding_service.embed(text)
        except Exception as e:
            logger.exception(f"Query embedding failed: {type(e).__name__}")
            raise EmbeddingFailedException("Failed to embed the query description")

        hits = await self.vector_db.near_vector_search(
            collection_name=self.collection_name,
            vector=vector,
            limit=limit,
            min_similarity=threshold,
            filters={"product_goal": product_goal} if product_goal else None,
        )

        return [
            {
                "opportunity_id": hit["id"],
                "product_goal": hit.get("product_goal") or "",
                "similarity": round(float(hit.get("similarity", 0.0)), 4),
            }
            for hit in hits
        ]

    async def delete(self, opportunity_id: str) -> bool:
        """
        Ensure an opportunity is not in the index.

        MANDATORY when ally-be soft-deletes or merges away an opportunity: Postgres reads filter
        on deletedAt IS NULL but this collection has no idea, so a skipped delete means the
        opportunity is proposed as a duplicate forever.

        Returns True when the index no longer contains the id — which includes the case where it
        never did. Weaviate's delete_by_id does NOT distinguish "deleted one" from "there was
        nothing to delete", and telling them apart would cost an extra read on every delete for
        a boolean no caller acts on. So this is idempotent by design: calling it twice returns
        True twice.

        False means the delete genuinely FAILED and the vector may still be there — ally-be
        treats that as drift to be healed by the reindex sweep.
        """
        try:
            await self.vector_db.delete_document(self.collection_name, opportunity_id)
            return True
        except VectorDBDeleteFailedException:
            logger.warning(
                f"Failed to delete opportunity {opportunity_id} from the index; it may still "
                f"surface as a duplicate candidate until the next reindex"
            )
            return False

    async def get(self, opportunity_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one indexed object's metadata. For reconciliation and debugging only."""
        try:
            return await self.vector_db.get_document_by_id(
                self.collection_name, opportunity_id, include_vector=False
            )
        except Exception:
            return None
