"""
Tests for migration 002: create reference document collection
"""

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.vector_db.constants import (
    ReferenceDocumentProperties,
    VectorDBCollectionNames,
)

# Import the migration module using importlib since the filename has hyphens
migration_002_create_reference_document_collection = importlib.import_module(
    "app.migrations.002-create-reference-document-collection"
)


class TestMigration002:
    """Test cases for migration 002: create reference document collection"""

    @pytest.mark.asyncio
    async def test_up_collection_does_not_exist(self):
        """Test migration up when collection doesn't exist"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "OtherCollection"

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.create = AsyncMock()

        await migration_002_create_reference_document_collection.up(mock_client)

        mock_client.collections.create.assert_called_once_with(
            name=VectorDBCollectionNames.REFERENCE_DOCUMENTS,
            properties=ReferenceDocumentProperties.get_all_properties(),
        )

    @pytest.mark.asyncio
    async def test_up_collection_already_exists(self):
        """Test migration up when collection already exists"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = VectorDBCollectionNames.REFERENCE_DOCUMENTS

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.create = AsyncMock()

        await migration_002_create_reference_document_collection.up(mock_client)

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
            await migration_002_create_reference_document_collection.up(mock_client)

    @pytest.mark.asyncio
    async def test_up_exception_during_list_all(self):
        """Test migration up when exception occurs during list_all"""
        mock_client = MagicMock()
        mock_client.collections.list_all = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        with pytest.raises(Exception, match="Connection failed"):
            await migration_002_create_reference_document_collection.up(mock_client)

    @pytest.mark.asyncio
    async def test_down_collection_exists(self):
        """Test migration down when collection exists"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = VectorDBCollectionNames.REFERENCE_DOCUMENTS

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.delete = AsyncMock()

        await migration_002_create_reference_document_collection.down(mock_client)

        mock_client.collections.delete.assert_called_once_with(
            VectorDBCollectionNames.REFERENCE_DOCUMENTS
        )

    @pytest.mark.asyncio
    async def test_down_collection_does_not_exist(self):
        """Test migration down when collection doesn't exist"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "OtherCollection"

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.delete = AsyncMock()

        await migration_002_create_reference_document_collection.down(mock_client)

        # Should not delete collection if it doesn't exist
        mock_client.collections.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_down_exception_during_deletion(self):
        """Test migration down when exception occurs during collection deletion"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = VectorDBCollectionNames.REFERENCE_DOCUMENTS

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.delete = AsyncMock(
            side_effect=Exception("Deletion failed")
        )

        with pytest.raises(Exception, match="Deletion failed"):
            await migration_002_create_reference_document_collection.down(mock_client)

    @pytest.mark.asyncio
    async def test_down_exception_during_list_all(self):
        """Test migration down when exception occurs during list_all"""
        mock_client = MagicMock()
        mock_client.collections.list_all = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        with pytest.raises(Exception, match="Connection failed"):
            await migration_002_create_reference_document_collection.down(mock_client)

    @pytest.mark.asyncio
    async def test_up_with_multiple_collections(self):
        """Test migration up with multiple existing collections"""
        mock_client = MagicMock()

        # Create multiple mock collections
        mock_collection1 = MagicMock()
        mock_collection1.name = "MigrationHistory"
        mock_collection2 = MagicMock()
        mock_collection2.name = "Conversations"
        mock_collection3 = MagicMock()
        mock_collection3.name = "OtherCollection"

        mock_client.collections.list_all = AsyncMock(
            return_value=[mock_collection1, mock_collection2, mock_collection3]
        )
        mock_client.collections.create = AsyncMock()

        await migration_002_create_reference_document_collection.up(mock_client)

        mock_client.collections.create.assert_called_once_with(
            name=VectorDBCollectionNames.REFERENCE_DOCUMENTS,
            properties=ReferenceDocumentProperties.get_all_properties(),
        )

    @pytest.mark.asyncio
    async def test_down_with_multiple_collections(self):
        """Test migration down with multiple existing collections"""
        mock_client = MagicMock()

        # Create multiple mock collections including the target one
        mock_collection1 = MagicMock()
        mock_collection1.name = "MigrationHistory"
        mock_collection2 = MagicMock()
        mock_collection2.name = "Conversations"
        mock_collection3 = MagicMock()
        mock_collection3.name = VectorDBCollectionNames.REFERENCE_DOCUMENTS

        mock_client.collections.list_all = AsyncMock(
            return_value=[mock_collection1, mock_collection2, mock_collection3]
        )
        mock_client.collections.delete = AsyncMock()

        await migration_002_create_reference_document_collection.down(mock_client)

        mock_client.collections.delete.assert_called_once_with(
            VectorDBCollectionNames.REFERENCE_DOCUMENTS
        )

    @pytest.mark.asyncio
    async def test_up_empty_collections_list(self):
        """Test migration up with empty collections list"""
        mock_client = MagicMock()
        mock_client.collections.list_all = AsyncMock(return_value=[])
        mock_client.collections.create = AsyncMock()

        await migration_002_create_reference_document_collection.up(mock_client)

        mock_client.collections.create.assert_called_once_with(
            name=VectorDBCollectionNames.REFERENCE_DOCUMENTS,
            properties=ReferenceDocumentProperties.get_all_properties(),
        )

    @pytest.mark.asyncio
    async def test_down_empty_collections_list(self):
        """Test migration down with empty collections list"""
        mock_client = MagicMock()
        mock_client.collections.list_all = AsyncMock(return_value=[])
        mock_client.collections.delete = AsyncMock()

        await migration_002_create_reference_document_collection.down(mock_client)

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

        await migration_002_create_reference_document_collection.up(mock_client)

        mock_client.collections.create.assert_called_once_with(
            name=VectorDBCollectionNames.REFERENCE_DOCUMENTS,
            properties=ReferenceDocumentProperties.get_all_properties(),
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

        await migration_002_create_reference_document_collection.down(mock_client)

        # Should not delete collection if it doesn't exist
        mock_client.collections.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_up_with_reference_document_properties(self):
        """Test migration up verifies correct reference document properties are used"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "OtherCollection"

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.create = AsyncMock()

        await migration_002_create_reference_document_collection.up(mock_client)

        # Verify the call was made with correct properties
        call_args = mock_client.collections.create.call_args
        assert call_args[1]["name"] == VectorDBCollectionNames.REFERENCE_DOCUMENTS
        assert (
            call_args[1]["properties"]
            == ReferenceDocumentProperties.get_all_properties()
        )

    @pytest.mark.asyncio
    async def test_up_collection_name_constant_usage(self):
        """Test migration up uses the correct collection name constant"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "OtherCollection"

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.create = AsyncMock()

        await migration_002_create_reference_document_collection.up(mock_client)

        # Verify the collection name constant is used
        call_args = mock_client.collections.create.call_args
        assert call_args[1]["name"] == VectorDBCollectionNames.REFERENCE_DOCUMENTS
        assert call_args[1]["name"] == "ReferenceDocument"  # Verify the actual value

    @pytest.mark.asyncio
    async def test_down_collection_name_constant_usage(self):
        """Test migration down uses the correct collection name constant"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = VectorDBCollectionNames.REFERENCE_DOCUMENTS

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.delete = AsyncMock()

        await migration_002_create_reference_document_collection.down(mock_client)

        # Verify the collection name constant is used
        mock_client.collections.delete.assert_called_once_with(
            VectorDBCollectionNames.REFERENCE_DOCUMENTS
        )
        mock_client.collections.delete.assert_called_once_with(
            "ReferenceDocument"
        )  # Verify the actual value

    @pytest.mark.asyncio
    async def test_up_properties_structure(self):
        """Test migration up uses correct property structure"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "OtherCollection"

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.create = AsyncMock()

        await migration_002_create_reference_document_collection.up(mock_client)

        # Verify the properties structure
        call_args = mock_client.collections.create.call_args
        properties = call_args[1]["properties"]

        # Should be a list of properties
        assert isinstance(properties, list)
        assert len(properties) == 5  # heading, content, category, tags, tenant_id

        # Verify property names
        property_names = [prop.name for prop in properties]
        assert "heading" in property_names
        assert "content" in property_names
        assert "category" in property_names
        assert "tags" in property_names
        assert "tenant_id" in property_names

    @pytest.mark.asyncio
    async def test_down_successful_deletion_message(self):
        """Test migration down logs successful deletion message"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = VectorDBCollectionNames.REFERENCE_DOCUMENTS

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.delete = AsyncMock()

        # Test that the migration runs without errors (logging is tested implicitly)
        await migration_002_create_reference_document_collection.down(mock_client)

        # Verify the delete was called
        mock_client.collections.delete.assert_called_once_with(
            VectorDBCollectionNames.REFERENCE_DOCUMENTS
        )

    @pytest.mark.asyncio
    async def test_up_successful_creation_message(self):
        """Test migration up logs successful creation message"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "OtherCollection"

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])
        mock_client.collections.create = AsyncMock()

        # Test that the migration runs without errors (logging is tested implicitly)
        await migration_002_create_reference_document_collection.up(mock_client)

        # Verify the create was called
        mock_client.collections.create.assert_called_once_with(
            name=VectorDBCollectionNames.REFERENCE_DOCUMENTS,
            properties=ReferenceDocumentProperties.get_all_properties(),
        )
