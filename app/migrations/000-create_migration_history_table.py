"""
Migration: create migration history table
Generated on: 2025-09-11 13:43:06
"""

from app.core.constants import MigrationHistoryProperties, VectorDBCollectionNames
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def up(client):
    """
    Run the migration up.

    Args:
        client: Weaviate client instance
    """
    logger.info("Running migration up: create migration history table")

    collection_name = VectorDBCollectionNames.MIGRATION_HISTORY

    try:
        # Check if collection already exists
        collections = await client.collections.list_all()
        existing_collections = [
            col.name if hasattr(col, "name") else str(col) for col in collections
        ]

        if collection_name not in existing_collections:
            logger.info(f"Creating migration history collection: {collection_name}")

            # Create the MigrationHistory collection
            await client.collections.create(
                name=collection_name,
                properties=MigrationHistoryProperties.get_all_properties(),
            )
            logger.info(
                f"Migration history collection '{collection_name}' created successfully"
            )
        else:
            logger.info(
                f"Migration history collection '{collection_name}' already exists"
            )

    except Exception as e:
        logger.error(
            f"Failed to create migration history collection: {type(e).__name__}"
        )
        raise

    logger.info("Migration up completed: create migration history table")


async def down(client):
    """
    Run the migration down (rollback).

    Args:
        client: Weaviate client instance
    """
    logger.info("Running migration down: create migration history table")

    collection_name = VectorDBCollectionNames.MIGRATION_HISTORY

    try:
        # Check if collection exists
        collections = await client.collections.list_all()
        existing_collections = [
            col.name if hasattr(col, "name") else str(col) for col in collections
        ]

        if collection_name in existing_collections:
            logger.info(f"Dropping migration history collection: {collection_name}")

            # Delete the MigrationHistory collection
            await client.collections.delete(collection_name)
            logger.info(
                f"Migration history collection '{collection_name}' dropped successfully"
            )
        else:
            logger.info(
                f"Migration history collection '{collection_name}' does not exist"
            )

    except Exception as e:
        logger.error(f"Failed to drop migration history collection: {type(e).__name__}")
        raise

    logger.info("Migration down completed: create migration history table")
