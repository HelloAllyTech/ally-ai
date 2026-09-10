"""
Migration: add-knowledge-chunk-audience

Adds `is_global` and `tenant_ids` to the KnowledgeChunk collection, so the WhatsApp Q&A
bot's corpus can be targeted at one, some or all organisations instead of being global
everyone who has the number (ally-be kb_documents.is_global + kb_document_tenants).

WHY THE FILTER LIVES ON THE CHUNK rather than being applied to results afterwards: a
post-filter over a fixed top-k starves recall. An organisation with five documents among
ten thousand global chunks would retrieve none of them, and the bot would decline while
looking like it simply had no coverage. A Weaviate filter narrows the search itself.

WHY EXISTING OBJECTS ARE BACKFILLED TO is_global=true: today's corpus IS global — every
chunk in this collection was indexed under the "open to anyone with the number" rule, so
that is not a guess about intent, it is what the data means. A property added to a
collection reads back as null on objects written before it existed, and null satisfies
neither half of the audience filter, so skipping the backfill would take the entire
existing corpus out of retrieval the moment the filter shipped: every worker's question
would be declined with nothing in any log to explain it. ally-be's matching migration
marks the same documents `is_global = true`, so the two stores agree without either
having to read the other.

The backfill is idempotent — it only writes objects whose `is_global` is still null,
so a re-run after a partial failure resumes rather than rewriting the collection.
"""

from weaviate.classes.query import Filter

from app.core.vector_db.constants import (
    KnowledgeChunkProperties,
    VectorDBCollectionNames,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

#: Objects updated per pass during the backfill. Small enough that a failure costs one
#: page of retries, large enough that a book-sized corpus is a handful of round trips.
BACKFILL_PAGE_SIZE = 200

#: Stall guard for the self-draining loop below — 200 × 500 is 100,000 chunks, far above
#: any corpus this collection holds, so reaching it means the writes are not taking.
MAX_PASSES = 500


async def _existing_property_names(client, collection_name: str) -> set:
    """Property names already on the collection, so an add is skipped, not 422ed."""
    collection = client.collections.get(collection_name)
    config = await collection.config.get()
    return {prop.name for prop in (config.properties or [])}


async def up(client):
    """
    Run the migration up.

    Args:
        client: Weaviate client instance
    """
    logger.info("Running migration up: add-knowledge-chunk-audience")

    collection_name = VectorDBCollectionNames.KNOWLEDGE_CHUNKS

    collections = await client.collections.list_all()
    existing_collections = [
        col.name if hasattr(col, "name") else str(col) for col in collections
    ]
    if collection_name not in existing_collections:
        # Migration 004 creates it with these properties already present, so a fresh
        # environment legitimately has nothing to do here.
        logger.info(
            f"Collection {collection_name} does not exist yet; migration 004 will "
            "create it with the audience properties included"
        )
        return

    collection = client.collections.get(collection_name)
    present = await _existing_property_names(client, collection_name)

    for prop in (
        KnowledgeChunkProperties.IS_GLOBAL,
        KnowledgeChunkProperties.TENANT_IDS,
    ):
        if prop.name in present:
            logger.info(f"Property {prop.name} already present; skipping")
            continue
        logger.info(f"Adding property {prop.name} to {collection_name}")
        await collection.config.add_property(prop)

    await _backfill_global(collection)

    logger.info("Migration up completed: add-knowledge-chunk-audience")


async def _backfill_global(collection) -> None:
    """
    Mark every pre-existing chunk as available to all organisations.

    SELF-DRAINING rather than cursor-paged: each pass asks for objects whose `is_global`
    is still null and writes them, which takes them out of the filter, so the next pass
    returns the next lot and an empty page means done. No cursor is involved, which
    matters because cursor paging is the one mode that cannot be combined with a filter
    — and the filter is what makes this re-runnable after a partial failure.

    `MAX_PASSES` is a stall guard, not a size limit. Without it, a write that silently
    did not take (a schema mismatch, a permissions problem) would spin on the same page
    forever inside a migration.
    """
    updated = 0

    for _ in range(MAX_PASSES):
        result = await collection.query.fetch_objects(
            limit=BACKFILL_PAGE_SIZE,
            filters=Filter.by_property("is_global").is_none(True),
            return_properties=[],
        )
        if not result.objects:
            logger.info(f"Backfilled {updated} chunk(s) to is_global=true")
            return

        for obj in result.objects:
            await collection.data.update(
                uuid=obj.uuid,
                properties={"is_global": True, "tenant_ids": []},
            )
            updated += 1

    raise RuntimeError(
        f"Backfill stalled after {MAX_PASSES} passes ({updated} chunk(s) written): "
        "objects matching is_global IS NULL keep coming back, so the writes are not "
        "taking. Investigate before re-running — the collection is half-backfilled and "
        "the un-backfilled half is not retrievable."
    )


async def down(client):
    """
    Run the migration down (rollback).

    Weaviate cannot drop a property, so this rollback is deliberately a no-op rather
    than a pretence. The properties stay; nothing reads them once the code that filters
    on them is rolled back, and the collection is derived data that ally-be can rebuild
    (`POST /api/v1/knowledge-base/documents/:id/reindex`) if it ever needs to be clean.

    Args:
        client: Weaviate client instance
    """
    logger.info(
        "Migration down: add-knowledge-chunk-audience is not reversible — Weaviate "
        "cannot drop a property. Leaving is_global/tenant_ids in place; they are inert "
        "without the code that filters on them."
    )
