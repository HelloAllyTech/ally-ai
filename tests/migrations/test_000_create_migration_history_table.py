"""
Tests for migration 000: create migration history table
"""

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.vector_db.constants import (
    MigrationHistoryProperties,
    VectorDBCollectionNames,
)

# Import the migration module using importlib since the filename has hyphens
migration_000_create_migration_history_table = importlib.import_module(
    "app.migrations.000-create_migration_history_table"
)


class TestMigration000:
    """Test cases for migration 000: create migration history table"""

    @pytest.mark.asyncio
    async def test_up_collection_does_not_exist(self):
        """Test migration up when collection doesn't exist"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "OtherCollection"

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.create = AsyncMock()

        await migration_000_create_migration_history_table.up(mock_client)

        mock_client.collections.create.assert_called_once_with(
            name=VectorDBCollectionNames.MIGRATION_HISTORY,
            properties=MigrationHistoryProperties.get_all_properties(),
        )

    @pytest.mark.asyncio
    async def test_up_collection_already_exists(self):
        """Test migration up when collection already exists"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = VectorDBCollectionNames.MIGRATION_HISTORY

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.create = AsyncMock()

        await migration_000_create_migration_history_table.up(mock_client)

        # Should not create collection if it already exists
        mock_client.collections.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_up_exception_during_creation(self):
        """Test migration up when exception occurs during collection creation"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "OtherCollection"

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.create = AsyncMock(
            side_effect=Exception("Creation failed")
        )

        with pytest.raises(Exception, match="Creation failed"):
            await migration_000_create_migration_history_table.up(mock_client)

    @pytest.mark.asyncio
    async def test_up_exception_during_list_all(self):
        """Test migration up when exception occurs during list_all"""
        mock_client = MagicMock()
        mock_client.collections.list_all = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        with pytest.raises(Exception, match="Connection failed"):
            await migration_000_create_migration_history_table.up(mock_client)

    @pytest.mark.asyncio
    async def test_down_collection_exists(self):
        """Test migration down when collection exists"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = VectorDBCollectionNames.MIGRATION_HISTORY

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.delete = AsyncMock()

        await migration_000_create_migration_history_table.down(mock_client)

        mock_client.collections.delete.assert_called_once_with(
            VectorDBCollectionNames.MIGRATION_HISTORY
        )

    @pytest.mark.asyncio
    async def test_down_collection_does_not_exist(self):
        """Test migration down when collection doesn't exist"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "OtherCollection"

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.delete = AsyncMock()

        await migration_000_create_migration_history_table.down(mock_client)

        # Should not delete collection if it doesn't exist
        mock_client.collections.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_down_exception_during_deletion(self):
        """Test migration down when exception occurs during collection deletion"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = VectorDBCollectionNames.MIGRATION_HISTORY

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.delete = AsyncMock(
            side_effect=Exception("Deletion failed")
        )

        with pytest.raises(Exception, match="Deletion failed"):
            await migration_000_create_migration_history_table.down(mock_client)

    @pytest.mark.asyncio
    async def test_down_exception_during_list_all(self):
        """Test migration down when exception occurs during list_all"""
        mock_client = MagicMock()
        mock_client.collections.list_all = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        with pytest.raises(Exception, match="Connection failed"):
            await migration_000_create_migration_history_table.down(mock_client)

    @pytest.mark.asyncio
    async def test_up_with_multiple_collections(self):
        """Test migration up with multiple existing collections"""
        mock_client = MagicMock()

        # Create multiple mock collections
        mock_collection1 = MagicMock()
        mock_collection1.name = "Conversations"
        mock_collection2 = MagicMock()
        mock_collection2.name = "ReferenceDocuments"
        mock_collection3 = MagicMock()
        mock_collection3.name = "OtherCollection"

        mock_client.collections.list_all = AsyncMock(
            return_value=[mock_collection1, mock_collection2, mock_collection3]
        )
        mock_client.collections.create = AsyncMock()

        await migration_000_create_migration_history_table.up(mock_client)

        mock_client.collections.create.assert_called_once_with(
            name=VectorDBCollectionNames.MIGRATION_HISTORY,
            properties=MigrationHistoryProperties.get_all_properties(),
        )

    @pytest.mark.asyncio
    async def test_down_with_multiple_collections(self):
        """Test migration down with multiple existing collections"""
        mock_client = MagicMock()

        # Create multiple mock collections including the target one
        mock_collection1 = MagicMock()
        mock_collection1.name = "Conversations"
        mock_collection2 = MagicMock()
        mock_collection2.name = VectorDBCollectionNames.MIGRATION_HISTORY
        mock_collection3 = MagicMock()
        mock_collection3.name = "ReferenceDocuments"

        mock_client.collections.list_all = AsyncMock(
            return_value=[mock_collection1, mock_collection2, mock_collection3]
        )
        mock_client.collections.delete = AsyncMock()

        await migration_000_create_migration_history_table.down(mock_client)

        mock_client.collections.delete.assert_called_once_with(
            VectorDBCollectionNames.MIGRATION_HISTORY
        )

    @pytest.mark.asyncio
    async def test_up_empty_collections_list(self):
        """Test migration up with empty collections list"""
        mock_client = MagicMock()
        mock_client.collections.list_all = AsyncMock(return_value=[])
        mock_client.collections.create = AsyncMock()

        await migration_000_create_migration_history_table.up(mock_client)

        mock_client.collections.create.assert_called_once_with(
            name=VectorDBCollectionNames.MIGRATION_HISTORY,
            properties=MigrationHistoryProperties.get_all_properties(),
        )

    @pytest.mark.asyncio
    async def test_down_empty_collections_list(self):
        """Test migration down with empty collections list"""
        mock_client = MagicMock()
        mock_client.collections.list_all = AsyncMock(return_value=[])
        mock_client.collections.delete = AsyncMock()

        await migration_000_create_migration_history_table.down(mock_client)

        # Should not delete collection if it doesn't exist
        mock_client.collections.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_up_collection_name_attribute_error(self):
        """Test migration up when collection object doesn't have name attribute"""
        mock_client = MagicMock()

        # Mock collection without name attribute
        mock_collection = MagicMock()
        del mock_collection.name

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.create = AsyncMock()

        await migration_000_create_migration_history_table.up(mock_client)

        mock_client.collections.create.assert_called_once_with(
            name=VectorDBCollectionNames.MIGRATION_HISTORY,
            properties=MigrationHistoryProperties.get_all_properties(),
        )

    @pytest.mark.asyncio
    async def test_down_collection_name_attribute_error(self):
        """Test migration down when collection object doesn't have name attribute"""
        mock_client = MagicMock()

        # Mock collection without name attribute
        mock_collection = MagicMock()
        del mock_collection.name

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.delete = AsyncMock()

        await migration_000_create_migration_history_table.down(mock_client)

        # Should not delete collection if it doesn't exist
        mock_client.collections.delete.assert_not_called()
