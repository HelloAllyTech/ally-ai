"""
Migration: create-roadmap-opportunity-collection

Semantic duplicate detection for the Ally Product Roadmap board (ally-be src/product-roadmap).
Replaces the pgvector/HNSW index the standalone roadmap app kept in Supabase; ally-be's
Postgres stays the system of record and this collection is a derived index.

Two things are set EXPLICITLY here rather than relying on client defaults, because the
duplicate-detection threshold depends on both:

  * vectorizer_config = none  — vectors are always supplied by ally-ai's OpenAI embedding
    service (text-embedding-3-small, 1536 dimensions). Weaviate must never try to vectorise
    these objects itself, since the properties deliberately exclude the opportunity text.
  * distance_metric = COSINE  — the standalone app's threshold (0.5) was cosine similarity on
    Voyage voyage-3-large at 1024 dimensions. The metric has to be pinned for any threshold
    to mean anything, and the value itself needs re-calibrating for the new model.
"""

import weaviate.classes.config as wvc

from app.core.vector_db.constants import (
    RoadmapOpportunityProperties,
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
    logger.info("Running migration up: create-roadmap-opportunity-collection")

    collection_name = VectorDBCollectionNames.ROADMAP_OPPORTUNITIES

    try:
        collections = await client.collections.list_all()
        existing_collections = [
            col.name if hasattr(col, "name") else str(col) for col in collections
        ]

        if collection_name not in existing_collections:
            logger.info(f"Creating collection: {collection_name}")

            await client.collections.create(
                name=collection_name,
                properties=RoadmapOpportunityProperties.get_all_properties(),
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

    logger.info("Migration up completed: create-roadmap-opportunity-collection")


async def down(client):
    """
    Run the migration down (rollback).

    Safe to run: the collection holds only derived data. Everything in it can be rebuilt from
    ally-be's Postgres via POST /api/v1/product-roadmap/admin/reindex.

    Args:
        client: Weaviate client instance
    """
    logger.info("Running migration down: create-roadmap-opportunity-collection")

    collection_name = VectorDBCollectionNames.ROADMAP_OPPORTUNITIES

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

    logger.info("Migration down completed: create-roadmap-opportunity-collection")
