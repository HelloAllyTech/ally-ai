"""Write and read side of the KnowledgeChunk vector index.

This collection is a DERIVED INDEX. ally-be's Postgres (kb_documents +
kb_document_chunks) is the system of record; here we hold the vector, the chunk text and
the metadata a citation needs.
Consequences worth knowing:

  * The Weaviate object UUID IS kb_document_chunks.id, so every write is idempotent by
    construction
    and a citation's chunk_id resolves straight back to the row holding its offsets.
  * Chunk text is IMMUTABLE for a given (document_id, chunk_version, chunk_index).
    Editing a
    document in ally-be bumps chunk_version, writes new rows under new UUIDs, and
    deletes the old generation's vectors — nothing is ever updated in place, so there is
    no staleness window to reason about.
  * Nothing here decides what is stale. ally-be owns the system of record and therefore
    owns that
    decision; this service enumerates and deletes on instruction.
"""

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List, Optional

from app.core.constants import EmbeddingConstants
from app.core.embeddings.base import BaseEmbeddingService
from app.core.vector_db.base import VectorDB
from app.core.vector_db.constants import VectorDBCollectionNames
from app.exceptions.custom_exceptions import EmbeddingFailedException
from app.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeChunkService:
    """Indexing and retrieval for the WhatsApp bot's knowledge corpus."""

    def __init__(
        self, vector_db: VectorDB, embedding_service: BaseEmbeddingService
    ) -> None:
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.collection_name = VectorDBCollectionNames.KNOWLEDGE_CHUNKS

    @staticmethod
    def hash_text(text: str) -> str:
        """
        SHA-256 of the embedded text. ally-be stores this to detect a stale vector.
        """
        return sha256(text.encode("utf-8")).hexdigest()

    # NO embedding-cost emission here, deliberately.
    #
    # `emit_ai_usage`'s own quantity guard (`_has_quantity`) requires a non-zero TOKEN
    # count for service='llm' and accepts `characters` only for 'stt'/'tts'. An
    # embedding emission carrying character counts would therefore be dropped inside the
    # emitter, giving us a call that looks like instrumentation and reports nothing —
    # worse than an acknowledged gap, because the dashboard would show embeddings as
    # free rather than as unmeasured.
    #
    # Emitting real token counts needs a cl100k_base tokeniser this service does not
    # have, and fabricating an estimate would put an invented number in a cost
    # dashboard. The honest fix is an 'embedding' branch in the emitter plus matching
    # handling in ally-be's consumer, which lands with the usage dashboard (where that
    # consumer is being touched anyway) rather than being half-done here.
    # `LLM_USAGE.COUNT_EMBEDDING_TOKENS` already exists and is unused for exactly this
    # reason.

    async def bulk_upsert(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Index a batch of chunks, embedding all their texts together.

        Returns ``{"succeeded": [...], "failed": [{"chunk_id", "error"}]}`` and never
        raises for a per-item problem. Partial failure has to be visible: ally-be tracks
        kb_documents.indexed_chunk_count against chunk_count and retries only the chunks
        that failed, so a batch reporting blanket success while dropping items would
        leave a document permanently short of passages while looking fully indexed.

        Old vectors for the same document are NOT removed here. Deleting is a separate,
        explicit call (`delete_document_chunks`) because re-index ordering is ally-be's
        decision: it deletes the previous generation first and accepts a brief coverage
        gap, on the grounds that a missing passage yields an honest decline while a
        duplicated one yields a confidently wrong citation.
        """
        succeeded: List[Dict[str, Any]] = []
        failed: List[Dict[str, str]] = []

        # A chunk with no text can never be embedded; fail it individually rather than
        # poisoning the batch.
        embeddable: List[Dict[str, Any]] = []
        for item in items:
            text = (item.get("text") or "").strip()
            if not text:
                failed.append(
                    {
                        "chunk_id": str(item.get("chunk_id", "")),
                        "error": "empty chunk text",
                    }
                )
                continue
            embeddable.append({**item, "text": text})

        if not embeddable:
            return {"succeeded": succeeded, "failed": failed}

        texts = [item["text"] for item in embeddable]
        try:
            vectors = await self.embedding_service.embed_many(texts)
        except Exception as e:
            # A whole-batch embedding failure is reported per item so the caller's
            # counters and retry bookkeeping stay accurate.
            logger.exception(f"Batch embedding failed: {type(e).__name__}")
            failed.extend(
                {
                    "chunk_id": str(item["chunk_id"]),
                    "error": f"embedding failed: {type(e).__name__}",
                }
                for item in embeddable
            )
            return {"succeeded": succeeded, "failed": failed}

        if len(vectors) != len(embeddable):
            # Never pair mismatched lists: attaching the wrong vector to a passage
            # produces a retrievable, plausible, wrong citation, which is worse than
            # failing outright.
            logger.error(
                f"Embedding count mismatch: asked for {len(embeddable)}, "
                f"got {len(vectors)}"
            )
            failed.extend(
                {"chunk_id": str(item["chunk_id"]), "error": "embedding count mismatch"}
                for item in embeddable
            )
            return {"succeeded": succeeded, "failed": failed}

        now = datetime.now(timezone.utc)
        documents: List[Dict[str, Any]] = []
        prepared: List[Dict[str, Any]] = []

        for item, vector in zip(embeddable, vectors):
            chunk_id = str(item["chunk_id"])
            text_hash = self.hash_text(item["text"])
            documents.append(
                {
                    "id": chunk_id,
                    "vector": vector,
                    "properties": {
                        "document_id": str(item["document_id"]),
                        "document_title": item.get("document_title") or "",
                        "chunk_index": int(item.get("chunk_index") or 0),
                        "text": item["text"],
                        "char_start": int(item.get("char_start") or 0),
                        "char_end": int(item.get("char_end") or 0),
                        # 0 rather than null for the page fields: Weaviate INT has no
                        # null that survives a round trip cleanly, and callers read 0 as
                        # "not paginated".
                        "page_from": int(item.get("page_from") or 0),
                        "page_to": int(item.get("page_to") or 0),
                        "section_path": item.get("section_path") or "",
                        "source_url": item.get("source_url") or "",
                        "language": item.get("language") or "",
                        "tags": item.get("tags") or [],
                        "token_count": int(item.get("token_count") or 0),
                        "text_hash": text_hash,
                        "embedding_model": EmbeddingConstants.MODEL,
                        "embedded_at": now,
                    },
                }
            )
            prepared.append({"chunk_id": chunk_id, "text_hash": text_hash})

        result = await self.vector_db.create_documents_bulk(
            self.collection_name, documents
        )

        written = set(result.get("succeeded", []))
        for entry in prepared:
            if entry["chunk_id"] in written:
                succeeded.append(
                    {
                        "chunk_id": entry["chunk_id"],
                        "text_hash": entry["text_hash"],
                        "embedding_model": EmbeddingConstants.MODEL,
                    }
                )

        failed.extend(
            {"chunk_id": item["id"], "error": item["error"]}
            for item in result.get("failed", [])
        )

        return {"succeeded": succeeded, "failed": failed}

    async def delete_document_chunks(self, document_id: str) -> int:
        """
        Remove every chunk belonging to a document, returning how many went.

        MANDATORY when ally-be re-chunks, archives or deletes a document. Without it the
        previous generation stays retrievable: it would still be cited, still occupy
        top-k slots a current passage needed, and — after an edit — would answer with
        text the document no longer says.

        Filter-based rather than id-based on purpose. ally-be holds the NEW chunk ids
        after a re-chunk, not the old ones, so enumerating first would mean a read per
        page plus a delete per object, and any page missed would silently orphan
        vectors.
        """
        deleted = await self.vector_db.delete_by_filter(
            self.collection_name, {"document_id": str(document_id)}
        )
        logger.info(f"Deleted {deleted} chunk(s) for document {document_id}")
        return deleted

    async def search(
        self,
        query: str,
        limit: int = 8,
        min_similarity: float = 0.35,
        document_ids: Optional[List[str]] = None,
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the passages most similar to a query string.

        Uses `near_vector_search`, NOT `search_documents`. The latter is shaped around
        the helpline reference-document corpus — it applies
        REFERENCE_DOCUMENTS_DISTANCE_THRESHOLD and runs a category aggregation — and
        that corpus is a separate, already-shipped feature whose retrieval behaviour
        must not shift because of this one.

        The query is embedded with the SAME model that embedded the chunks. That is
        not a preference: a query vector from a different model lands in a different
        space, which makes every similarity number meaningless rather than merely
        worse, and every threshold built on top of them arbitrary.
        """
        text = (query or "").strip()
        if not text:
            return []

        try:
            vector = await self.embedding_service.embed(text)
        except Exception as e:
            logger.exception(f"Query embedding failed: {type(e).__name__}")
            raise EmbeddingFailedException("Failed to embed the query")

        # near_vector_search supports property equality only, so a document_ids
        # restriction is applied after the fact. Retrieval asks for more than `limit`
        # when filtering so the filter cannot starve the result set down to nothing.
        overfetch = limit * 4 if document_ids else limit

        hits = await self.vector_db.near_vector_search(
            collection_name=self.collection_name,
            vector=vector,
            limit=overfetch,
            min_similarity=min_similarity,
            filters={"language": language} if language else None,
        )

        if document_ids:
            allowed = {str(d) for d in document_ids}
            hits = [h for h in hits if str(h.get("document_id")) in allowed]

        return [self._to_passage(hit) for hit in hits[:limit]]

    @staticmethod
    def _to_passage(hit: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalise a raw Weaviate hit into the passage shape the agent and admin API use.
        """
        return {
            "chunk_id": hit["id"],
            "document_id": str(hit.get("document_id") or ""),
            "document_title": hit.get("document_title") or "",
            "chunk_index": int(hit.get("chunk_index") or 0),
            "text": hit.get("text") or "",
            "char_start": int(hit.get("char_start") or 0),
            "char_end": int(hit.get("char_end") or 0),
            "page_from": int(hit.get("page_from") or 0),
            "page_to": int(hit.get("page_to") or 0),
            "section_path": hit.get("section_path") or "",
            "source_url": hit.get("source_url") or "",
            "language": hit.get("language") or "",
            "token_count": int(hit.get("token_count") or 0),
            "similarity": round(float(hit.get("similarity", 0.0)), 4),
        }

    async def list_ids(
        self, limit: int = 200, after: Optional[str] = None
    ) -> List[str]:
        """
        One page of indexed chunk ids, for RECONCILIATION by ally-be.

        Every other read here is keyed by an id the caller already holds, which by
        construction cannot surface an object the caller has FORGOTTEN. That leaves one
        drift mode with no detection path: a vector whose Postgres row was hard-deleted.
        Nothing removes it — a document delete calls delete_document_chunks, but a row
        that vanished outright never triggered anything — so it lingers, still occupying
        a top-k slot a real passage needed.
        """
        return await self.vector_db.list_document_ids(
            self.collection_name, limit=limit, after=after
        )

    async def get(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        One indexed chunk. Backs citation resolution in the admin log and debugging.
        """
        try:
            document = await self.vector_db.get_document_by_id(
                self.collection_name, chunk_id, include_vector=False
            )
        except Exception:
            return None
        return self._to_passage({**document, "id": str(document.get("id", chunk_id))})
