"""
Migration: create-knowledge-chunk-collection

Passage-level retrieval for the WhatsApp Q&A bot that answers mental healthcare workers'
questions (ally-be src/knowledge-base + src/whatsapp). ally-be's Postgres (kb_documents
+ kb_document_chunks) is the system of record; this collection is a derived index.

Why a NEW collection rather than extending ReferenceDocument: that collection stores one
object per document carrying a SINGLE embedding of the entire body, which is why the
helpline search it backs can only ever cite a whole document. Retrieval precision and
citable passages both need one object per chunk. Extending it in place would also change
retrieval behaviour for an already-shipped feature, so it is left strictly alone.

Two things are set EXPLICITLY here rather than relying on client defaults, because the
retrieval thresholds depend on both:

  * vectorizer_config = none — vectors are always supplied by ally-ai's OpenAI embedding
    service (text-embedding-3-small, 1536 dimensions). Weaviate must never vectorise
    these objects itself: the query side embeds with that exact model, and a mismatched
    vector space makes every similarity number meaningless rather than merely worse.
  * distance_metric = COSINE — the agent's thresholds (min_similarity 0.35 as a
    permissive floor, decline_similarity 0.42 as the actual decision) are cosine
    similarities. The metric has to be pinned for any threshold to mean anything.
"""

import weaviate.classes.config as wvc

from app.core.vector_db.constants import (
    KnowledgeChunkProperties,
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
    logger.info("Running migration up: create-knowledge-chunk-collection")

    collection_name = VectorDBCollectionNames.KNOWLEDGE_CHUNKS

    try:
        collections = await client.collections.list_all()
        existing_collections = [
            col.name if hasattr(col, "name") else str(col) for col in collections
        ]

        if collection_name not in existing_collections:
            logger.info(f"Creating collection: {collection_name}")

            await client.collections.create(
                name=collection_name,
                properties=KnowledgeChunkProperties.get_all_properties(),
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

    logger.info("Migration up completed: create-knowledge-chunk-collection")


async def down(client):
    """
    Run the migration down (rollback).

    Safe to run: the collection holds only derived data. Every chunk can be rebuilt from
    ally-be's Postgres — kb_documents.raw_text is retained precisely so a re-chunk never
    needs to re-parse the original PDF/DOCX/EPUB — via POST
    /api/v1/knowledge-base/documents/:id/reindex.

    Args:
        client: Weaviate client instance
    """
    logger.info("Running migration down: create-knowledge-chunk-collection")

    collection_name = VectorDBCollectionNames.KNOWLEDGE_CHUNKS

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

    logger.info("Migration down completed: create-knowledge-chunk-collection")
