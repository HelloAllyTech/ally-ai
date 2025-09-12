"""
Migration: create-reference-document-collection
Generated on: 2024-01-15 10:35:00
"""

from app.core.constants import ReferenceDocumentProperties, VectorDBCollectionNames
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def up(client):
    """
    Run the migration up.

    Args:
        client: Weaviate client instance
    """
    logger.info("Running migration up: create-reference-document-collection")

    collection_name = VectorDBCollectionNames.REFERENCE_DOCUMENTS

    try:
        # Check if collection exists
        collections = await client.collections.list_all()
        existing_collections = [
            col.name if hasattr(col, "name") else str(col) for col in collections
        ]

        if collection_name not in existing_collections:
            logger.info(f"Creating collection: {collection_name}")

            # Create collection with schema using constants
            await client.collections.create(
                name=collection_name,
                properties=ReferenceDocumentProperties.get_all_properties(),
            )
            logger.info(f"Collection {collection_name} created successfully")
        else:
            logger.info(f"Collection {collection_name} already exists")

    except Exception as e:
        logger.error(
            f"Failed to create collection {collection_name}: {type(e).__name__}"
        )
        raise

    logger.info("Migration up completed: create-reference-document-collection")


async def down(client):
    """
    Run the migration down (rollback).

    Args:
        client: Weaviate client instance
    """
    logger.info("Running migration down: create-reference-document-collection")

    collection_name = VectorDBCollectionNames.REFERENCE_DOCUMENTS

    try:
        # Check if collection exists
        collections = await client.collections.list_all()
        existing_collections = [
            col.name if hasattr(col, "name") else str(col) for col in collections
        ]

        if collection_name in existing_collections:
            logger.info(f"Dropping collection: {collection_name}")

            # Delete the collection
            await client.collections.delete(collection_name)
            logger.info(f"Collection {collection_name} dropped successfully")
        else:
            logger.info(f"Collection {collection_name} does not exist")

    except Exception as e:
        logger.error(f"Failed to drop collection {collection_name}: {type(e).__name__}")
        raise

    logger.info("Migration down completed: create-reference-document-collection")
