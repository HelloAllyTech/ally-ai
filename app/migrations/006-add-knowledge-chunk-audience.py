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

BOTH corpus collections are patched, not just KnowledgeChunk. `CharacterChunk` (migration
005) was created from the same property list before these two existed, so a store built
before this migration has the character corpus without them — and the audience filter is
applied by corpus-agnostic code. Filtering `is_global == true` over a collection that has
no such property takes the whole character corpus out of retrieval exactly as a missing
backfill would for the WhatsApp one.

WHY THE BACKFILL SCANS RATHER THAN FILTERING ON NULL: it used to ask for objects matching
`is_global IS NULL`, which cannot work here. Weaviate only answers a null filter when the
collection indexes null state, and migration 004 did not enable it — so the query failed
with `fetch doc ids for prop/value pair: Null` on every environment, meaning this
migration had never once completed and no store held state derived from it. A cursor scan
reads the property off each object instead and writes only the unset ones, which keeps
the idempotence the filter was there for without needing an index that isn't there.
"""

from weaviate.classes.query import Filter

from app.core.vector_db.constants import (
    KnowledgeChunkProperties,
    VectorDBCollectionNames,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

#: Objects the cursor fetches per round trip. Small enough that a failure costs one page
#: of retries, large enough that a book-sized corpus is a handful of round trips.
BACKFILL_PAGE_SIZE = 200


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

    collections = await client.collections.list_all()
    existing_collections = [
        col.name if hasattr(col, "name") else str(col) for col in collections
    ]

    for collection_name in (
        VectorDBCollectionNames.KNOWLEDGE_CHUNKS,
        VectorDBCollectionNames.CHARACTER_CHUNKS,
    ):
        if collection_name not in existing_collections:
            # Migrations 004 and 005 build these from the same property list, which now
            # includes the audience pair, so a fresh environment has nothing to do here.
            logger.info(
                f"Collection {collection_name} does not exist yet; its creating "
                "migration includes the audience properties"
            )
            continue

        collection = client.collections.get(collection_name)
        present = await _existing_property_names(client, collection_name)

        for prop in (
            KnowledgeChunkProperties.IS_GLOBAL,
            KnowledgeChunkProperties.TENANT_IDS,
        ):
            if prop.name in present:
                logger.info(
                    f"Property {prop.name} already present on {collection_name}; skipping"
                )
                continue
            logger.info(f"Adding property {prop.name} to {collection_name}")
            await collection.config.add_property(prop)

        await _backfill_global(collection, collection_name)

    logger.info("Migration up completed: add-knowledge-chunk-audience")


async def _backfill_global(collection, collection_name: str) -> None:
    """
    Mark every pre-existing chunk in one collection as available to all organisations.

    A CURSOR SCAN, reading `is_global` off each object and writing only the unset ones.
    The alternative — asking the engine for objects matching `is_global IS NULL` — is
    what this migration originally did, and it cannot work: a null filter needs the
    collection to index null state, which migration 004 did not enable, so the query
    failed outright rather than returning nothing.

    Scanning keeps the property that mattered about the filter version. Writing a chunk
    sets `is_global`, so a re-run after a partial failure simply skips what already
    landed; the scan costs one pass over a corpus that is thousands of objects, not
    millions. There is no stall guard because there is no longer a loop that can stall:
    the cursor advances whether or not a write takes, and it ends.
    """
    updated = 0
    seen = 0

    async for obj in collection.iterator(
        return_properties=[KnowledgeChunkProperties.IS_GLOBAL.name],
        cache_size=BACKFILL_PAGE_SIZE,
    ):
        seen += 1
        if obj.properties.get(KnowledgeChunkProperties.IS_GLOBAL.name) is not None:
            continue
        await collection.data.update(
            uuid=obj.uuid,
            properties={"is_global": True, "tenant_ids": []},
        )
        updated += 1

    logger.info(
        f"{collection_name}: backfilled {updated} of {seen} chunk(s) to is_global=true"
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
