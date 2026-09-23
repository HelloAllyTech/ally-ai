"""
Migration: create-agent-memory-collection

Semantic search over the notebook an Ally agent keeps of what it has learned — Bug
Hunter first (ally-be src/agent-memory, roadmap item OPP-0739/OPP-0714). ally-be's
Postgres is the system of record; this collection is a derived index of vectors plus
the minimum metadata to filter by agent and detect a stale vector.

Configured exactly like RoadmapOpportunity, and for the same reasons:

  * vectorizer_config = none — vectors are supplied by ally-ai's embedding service
    (text-embedding-3-small, 1536 dimensions). The entry text is deliberately not a
    property, so Weaviate must never try to vectorise these objects itself.
  * distance_metric = COSINE — the relevance threshold ally-be applies is a cosine
    similarity, and it only means something against a pinned metric.
"""

import weaviate.classes.config as wvc

from app.core.vector_db.constants import (
    AgentMemoryProperties,
    VectorDBCollectionNames,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def up(client):
    """
    Run the migration up.

    Args:
        client: Weaviate client instance
    """
    logger.info("Running migration up: create-agent-memory-collection")

    collection_name = VectorDBCollectionNames.AGENT_MEMORIES

    try:
        collections = await client.collections.list_all()
        existing_collections = [
            col.name if hasattr(col, "name") else str(col) for col in collections
        ]

        if collection_name not in existing_collections:
            logger.info(f"Creating collection: {collection_name}")

            await client.collections.create(
                name=collection_name,
                properties=AgentMemoryProperties.get_all_properties(),
                vectorizer_config=wvc.Configure.Vectorizer.none(),
                vector_index_config=wvc.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.VectorDistances.COSINE,
                ),
            )
            logger.info(f"Collection {collection_name} created successfully")
        else:
            logger.info(f"Collection {collection_name} already exists")

    except Exception as e:
        logger.error(
            f"Failed to create collection {collection_name}: {type(e).__name__}"
        )
        raise

    logger.info("Migration up completed: create-agent-memory-collection")


async def down(client):
    """
    Run the migration down (rollback).

    Safe to run: the collection holds only derived data and can be rebuilt from
    ally-be's agent_memories rows.

    Args:
        client: Weaviate client instance
    """
    logger.info("Running migration down: create-agent-memory-collection")

    collection_name = VectorDBCollectionNames.AGENT_MEMORIES

    try:
        collections = await client.collections.list_all()
        existing_collections = [
            col.name if hasattr(col, "name") else str(col) for col in collections
        ]

        if collection_name in existing_collections:
            logger.info(f"Dropping collection: {collection_name}")
            await client.collections.delete(collection_name)
            logger.info(f"Collection {collection_name} dropped successfully")
        else:
            logger.info(f"Collection {collection_name} does not exist")

    except Exception as e:
        logger.error(f"Failed to drop collection {collection_name}: {type(e).__name__}")
        raise

    logger.info("Migration down completed: create-agent-memory-collection")
