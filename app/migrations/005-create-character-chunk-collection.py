"""
Migration: create-character-chunk-collection

Passage-level retrieval for the character-library interview agent — the clinical and
lived-experience material it draws on so its questions are specific and its drafts are
real, instead of a plausible-sounding person invented from nothing (ally-be
src/knowledge-base with corpus='character_library', src/scenario-character). ally-be's
Postgres (kb_documents + kb_document_chunks) is the system of record; this collection is
a derived index, exactly as KnowledgeChunk is for the WhatsApp corpus.

Built from KnowledgeChunkProperties verbatim, because a passage is the same shape
whatever it grounds — the citation chain (Weaviate object uuid == kb_document_chunks.id,
char offsets into kb_documents.raw_text) is identical and deliberately not re-invented.

Why a SEPARATE COLLECTION rather than a `corpus` property on KnowledgeChunk, which the
pipeline could have filtered on:

  * A similarity threshold only means something against ONE distribution. Chunk size is
    chosen per corpus — 400 tokens for a 1600-character WhatsApp reply, 800 for a
    character vignette that has to hold a whole observation together — and a longer
    passage embeds more diffusely. One index holding both sizes would leave one
    threshold straddling two distributions, and every number built on it arbitrary.
  * Filtered ANN search is weaker than unfiltered. HNSW traverses a graph built over
    every vector in the collection, so scoping by a low-selectivity filter costs recall
    or degrades to a scan. A collection per corpus traverses only its own graph.
  * A scope passed as an argument can be forgotten; a collection cannot. Both corpora
    live behind one shared `KnowledgeChunk`-shaped pipeline, and an omitted filter there
    would ground a health worker's answer in character material — which reads as good
    clinical prose and is not an answer to their question.
  * Independent lifecycle. Re-chunking the character corpus (a changed profile, a fixed
    splitter) must not touch the index the live WhatsApp bot is serving from, and a
    future corpus wanting a different embedding model could not share a vector space at
    all: query vectors from another model land somewhere else entirely.

Same two explicit settings as 004, for the same reasons — vectors always come from
ally-ai's embedding service (never Weaviate's own vectoriser, or the query side's model
would not match), and COSINE is pinned because every retrieval threshold is expressed as
a cosine similarity.
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
    logger.info("Running migration up: create-character-chunk-collection")

    collection_name = VectorDBCollectionNames.CHARACTER_CHUNKS

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

    logger.info("Migration up completed: create-character-chunk-collection")


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
    logger.info("Running migration down: create-character-chunk-collection")

    collection_name = VectorDBCollectionNames.CHARACTER_CHUNKS

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

    logger.info("Migration down completed: create-character-chunk-collection")
