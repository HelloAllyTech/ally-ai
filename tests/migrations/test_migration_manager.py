"""
Tests for MigrationManager class
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.vector_db.constants import VectorDBCollectionNames
from app.migrations.manager import MIGRATION_FILE_PATTERN, MigrationManager


class TestMigrationManager:
    """Test cases for MigrationManager class"""

    def test_init(self):
        """Test MigrationManager initialization"""
        mock_client = MagicMock()
        manager = MigrationManager(client=mock_client)

        assert manager.client == mock_client
        assert isinstance(manager.migrations_dir, Path)
        assert isinstance(manager.project_root, Path)

    def test_init_without_client(self):
        """Test MigrationManager initialization without client"""
        manager = MigrationManager()
        assert manager.client is None

    @pytest.mark.asyncio
    async def test_ensure_migrations_collection_exists(self):
        """Test ensure_migrations_collection when collection exists"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = VectorDBCollectionNames.MIGRATION_HISTORY

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])

        manager = MigrationManager(client=mock_client)

        # Should not raise exception
        await manager.ensure_migrations_collection()

    @pytest.mark.asyncio
    async def test_ensure_migrations_collection_not_exists(self):
        """Test ensure_migrations_collection when collection doesn't exist"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "OtherCollection"

        mock_client.collections.list_all = AsyncMock(return_value=[mock_collection])

        manager = MigrationManager(client=mock_client)

        with pytest.raises(Exception, match="MigrationHistory collection.*not found"):
            await manager.ensure_migrations_collection()

    @pytest.mark.asyncio
    async def test_ensure_migrations_collection_exception(self):
        """Test ensure_migrations_collection when exception occurs"""
        mock_client = MagicMock()
        mock_client.collections.list_all = AsyncMock(
            side_effect=Exception("Connection error")
        )

        manager = MigrationManager(client=mock_client)

        with pytest.raises(Exception, match="Connection error"):
            await manager.ensure_migrations_collection()

    @pytest.mark.asyncio
    async def test_get_applied_migrations_success(self):
        """Test get_applied_migrations with successful response"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_object = MagicMock()
        mock_object.properties = {"version": "001"}

        mock_response = MagicMock()
        mock_response.objects = [mock_object]

        mock_collection.query.fetch_objects = AsyncMock(return_value=mock_response)
        mock_client.collections.get.return_value = mock_collection

        manager = MigrationManager(client=mock_client)

        result = await manager.get_applied_migrations()

        assert result == ["001"]

    @pytest.mark.asyncio
    async def test_get_applied_migrations_empty(self):
        """Test get_applied_migrations with empty response"""
        mock_client = MagicMock()
        mock_collection = MagicMock()

        mock_response = MagicMock()
        mock_response.objects = []

        mock_collection.query.fetch_objects = AsyncMock(return_value=mock_response)
        mock_client.collections.get.return_value = mock_collection

        manager = MigrationManager(client=mock_client)

        result = await manager.get_applied_migrations()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_applied_migrations_exception(self):
        """Test get_applied_migrations when exception occurs"""
        mock_client = MagicMock()
        mock_client.collections.get.side_effect = Exception("Collection not found")

        manager = MigrationManager(client=mock_client)

        result = await manager.get_applied_migrations()

        assert result == []

    def test_get_migration_files(self):
        """Test get_migration_files"""
        with patch("app.migrations.manager.Path"):
            mock_migrations_dir = MagicMock()
            mock_migrations_dir.exists.return_value = True

            # Mock migration files
            mock_file1 = MagicMock()
            mock_file1.name = "001-create-conversation-collection.py"
            mock_file2 = MagicMock()
            mock_file2.name = "002-create-reference-document-collection.py"
            mock_file3 = MagicMock()
            mock_file3.name = "__init__.py"  # Should be skipped
            mock_file4 = MagicMock()
            mock_file4.name = "invalid-name.py"  # Should be skipped

            mock_migrations_dir.glob.return_value = [
                mock_file1,
                mock_file2,
                mock_file3,
                mock_file4,
            ]

            manager = MigrationManager()
            manager.migrations_dir = mock_migrations_dir

            result = manager.get_migration_files()

            # Should only return valid migration files, sorted by name
            assert len(result) == 2
            assert result[0].name == "001-create-conversation-collection.py"
            assert result[1].name == "002-create-reference-document-collection.py"

    def test_get_migration_files_directory_not_exists(self):
        """Test get_migration_files when directory doesn't exist"""
        with patch("app.migrations.manager.Path"):
            mock_migrations_dir = MagicMock()
            mock_migrations_dir.exists.return_value = False

            manager = MigrationManager()
            manager.migrations_dir = mock_migrations_dir

            result = manager.get_migration_files()

            assert result == []

    def test_get_migration_version_valid(self):
        """Test get_migration_version with valid filename"""
        manager = MigrationManager()

        with patch("app.migrations.manager.Path"):
            mock_file = MagicMock()
            mock_file.name = "001-create-conversation-collection.py"

            result = manager.get_migration_version(mock_file)

            assert result == "001"

    def test_get_migration_version_invalid(self):
        """Test get_migration_version with invalid filename"""
        manager = MigrationManager()

        with patch("app.migrations.manager.Path"):
            mock_file = MagicMock()
            mock_file.name = "invalid-name.py"

            result = manager.get_migration_version(mock_file)

            assert result == "000"  # Fallback

    @pytest.mark.asyncio
    async def test_record_migration_success(self):
        """Test record_migration with successful execution"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.data.insert = AsyncMock()

        mock_client.collections.get.return_value = mock_collection

        manager = MigrationManager(client=mock_client)

        await manager.record_migration("001", "test-migration", "Test migration")

        mock_collection.data.insert.assert_called_once()
        call_args = mock_collection.data.insert.call_args[1]["properties"]
        assert call_args["version"] == "001"
        assert call_args["name"] == "test-migration"
        assert call_args["description"] == "Test migration"
        assert call_args["status"] == "completed"

    @pytest.mark.asyncio
    async def test_record_migration_exception(self):
        """Test record_migration when exception occurs"""
        mock_client = MagicMock()
        mock_client.collections.get.side_effect = Exception("Collection not found")

        manager = MigrationManager(client=mock_client)

        with pytest.raises(Exception, match="Collection not found"):
            await manager.record_migration("001", "test-migration", "Test migration")

    @pytest.mark.asyncio
    async def test_delete_migration_record_success(self):
        """Test delete_migration_record with successful execution"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_object = MagicMock()
        mock_object.uuid = "test-uuid"

        mock_response = MagicMock()
        mock_response.objects = [mock_object]

        mock_collection.query.fetch_objects = AsyncMock(return_value=mock_response)
        mock_collection.data.delete_by_id = AsyncMock()
        mock_client.collections.get.return_value = mock_collection

        manager = MigrationManager(client=mock_client)

        await manager.delete_migration_record("001")

        mock_collection.data.delete_by_id.assert_called_once_with("test-uuid")

    @pytest.mark.asyncio
    async def test_delete_migration_record_no_objects(self):
        """Test delete_migration_record when no objects found"""
        mock_client = MagicMock()
        mock_collection = MagicMock()

        mock_response = MagicMock()
        mock_response.objects = []

        mock_collection.query.fetch_objects = AsyncMock(return_value=mock_response)
        mock_client.collections.get.return_value = mock_collection

        manager = MigrationManager(client=mock_client)

        # Should not raise exception
        await manager.delete_migration_record("001")

    @pytest.mark.asyncio
    async def test_delete_migration_record_exception(self):
        """Test delete_migration_record when exception occurs"""
        mock_client = MagicMock()
        mock_client.collections.get.side_effect = Exception("Collection not found")

        manager = MigrationManager(client=mock_client)

        with pytest.raises(Exception, match="Collection not found"):
            await manager.delete_migration_record("001")

    @pytest.mark.asyncio
    async def test_run_migration_up_success(self):
        """Test run_migration_up with successful execution"""
        mock_client = MagicMock()

        # Mock migration file
        mock_file = MagicMock()
        mock_file.name = "001-create-conversation-collection.py"
        mock_file.stem = "001-create-conversation-collection"

        manager = MigrationManager(client=mock_client)

        with patch.object(manager, "get_migration_files", return_value=[mock_file]):
            with patch.object(manager, "get_migration_version", return_value="001"):
                with patch("importlib.import_module") as mock_import:
                    mock_module = MagicMock()
                    mock_module.up = AsyncMock()
                    mock_import.return_value = mock_module

                    with patch.object(
                        manager, "record_migration", new_callable=AsyncMock
                    ):
                        result = await manager.run_migration_up("001")

                        assert result is True
                        mock_module.up.assert_called_once_with(mock_client)

    @pytest.mark.asyncio
    async def test_run_migration_up_file_not_found(self):
        """Test run_migration_up when migration file not found"""
        mock_client = MagicMock()
        manager = MigrationManager(client=mock_client)

        with patch.object(manager, "get_migration_files", return_value=[]):
            result = await manager.run_migration_up("001")

            assert result is False

    @pytest.mark.asyncio
    async def test_run_migration_up_no_up_function(self):
        """Test run_migration_up when migration has no up function"""
        mock_client = MagicMock()

        mock_file = MagicMock()
        mock_file.name = "001-create-conversation-collection.py"
        mock_file.stem = "001-create-conversation-collection"

        manager = MigrationManager(client=mock_client)

        with patch.object(manager, "get_migration_files", return_value=[mock_file]):
            with patch.object(manager, "get_migration_version", return_value="001"):
                with patch("importlib.import_module") as mock_import:
                    mock_module = MagicMock()
                    # No 'up' function
                    del mock_module.up
                    mock_import.return_value = mock_module

                    result = await manager.run_migration_up("001")

                    assert result is False

    @pytest.mark.asyncio
    async def test_run_migration_up_exception(self):
        """Test run_migration_up when migration execution fails"""
        mock_client = MagicMock()

        mock_file = MagicMock()
        mock_file.name = "001-create-conversation-collection.py"
        mock_file.stem = "001-create-conversation-collection"

        manager = MigrationManager(client=mock_client)

        with patch.object(manager, "get_migration_files", return_value=[mock_file]):
            with patch.object(manager, "get_migration_version", return_value="001"):
                with patch("importlib.import_module") as mock_import:
                    mock_module = MagicMock()
                    mock_module.up = AsyncMock(
                        side_effect=Exception("Migration failed")
                    )
                    mock_import.return_value = mock_module

                    with patch.object(
                        manager, "record_migration", new_callable=AsyncMock
                    ):
                        result = await manager.run_migration_up("001")

                        assert result is False

    @pytest.mark.asyncio
    async def test_run_migration_down_success(self):
        """Test run_migration_down with successful execution"""
        mock_client = MagicMock()

        mock_file = MagicMock()
        mock_file.name = "001-create-conversation-collection.py"
        mock_file.stem = "001-create-conversation-collection"

        manager = MigrationManager(client=mock_client)

        with patch.object(manager, "get_migration_files", return_value=[mock_file]):
            with patch.object(manager, "get_migration_version", return_value="001"):
                with patch("importlib.import_module") as mock_import:
                    mock_module = MagicMock()
                    mock_module.down = AsyncMock()
                    mock_import.return_value = mock_module

                    with patch.object(
                        manager, "delete_migration_record", new_callable=AsyncMock
                    ):
                        result = await manager.run_migration_down("001")

                        assert result is True
                        mock_module.down.assert_called_once_with(mock_client)

    @pytest.mark.asyncio
    async def test_run_migration_down_no_down_function(self):
        """Test run_migration_down when migration has no down function"""
        mock_client = MagicMock()

        mock_file = MagicMock()
        mock_file.name = "001-create-conversation-collection.py"
        mock_file.stem = "001-create-conversation-collection"

        manager = MigrationManager(client=mock_client)

        with patch.object(manager, "get_migration_files", return_value=[mock_file]):
            with patch.object(manager, "get_migration_version", return_value="001"):
                with patch("importlib.import_module") as mock_import:
                    mock_module = MagicMock()
                    # No 'down' function
                    del mock_module.down
                    mock_import.return_value = mock_module

                    result = await manager.run_migration_down("001")

                    assert result is False

    @pytest.mark.asyncio
    async def test_run_all_migrations_success(self):
        """Test run_all_migrations with successful execution"""
        mock_client = MagicMock()

        mock_file1 = MagicMock()
        mock_file1.name = "001-create-conversation-collection.py"
        mock_file2 = MagicMock()
        mock_file2.name = "002-create-reference-document-collection.py"

        manager = MigrationManager(client=mock_client)

        with patch.object(
            manager, "ensure_migrations_collection", new_callable=AsyncMock
        ):
            with patch.object(manager, "get_applied_migrations", return_value=[]):
                with patch.object(
                    manager,
                    "get_migration_files",
                    return_value=[mock_file1, mock_file2],
                ):
                    with patch.object(
                        manager, "get_migration_version", side_effect=["001", "002"]
                    ):
                        with patch.object(
                            manager,
                            "run_migration_up",
                            new_callable=AsyncMock,
                            return_value=True,
                        ):
                            result = await manager.run_all_migrations()

                            assert result is True

    @pytest.mark.asyncio
    async def test_run_all_migrations_partial_failure(self):
        """Test run_all_migrations with partial failure"""
        mock_client = MagicMock()

        mock_file1 = MagicMock()
        mock_file1.name = "001-create-conversation-collection.py"
        mock_file2 = MagicMock()
        mock_file2.name = "002-create-reference-document-collection.py"

        manager = MigrationManager(client=mock_client)

        with patch.object(
            manager, "ensure_migrations_collection", new_callable=AsyncMock
        ):
            with patch.object(manager, "get_applied_migrations", return_value=[]):
                with patch.object(
                    manager,
                    "get_migration_files",
                    return_value=[mock_file1, mock_file2],
                ):
                    with patch.object(
                        manager, "get_migration_version", side_effect=["001", "002"]
                    ):
                        with patch.object(
                            manager,
                            "run_migration_up",
                            new_callable=AsyncMock,
                            side_effect=[True, False],
                        ):
                            result = await manager.run_all_migrations()

                            assert result is False

    @pytest.mark.asyncio
    async def test_run_all_migrations_exception(self):
        """Test run_all_migrations when exception occurs"""
        mock_client = MagicMock()
        manager = MigrationManager(client=mock_client)

        with patch.object(
            manager,
            "ensure_migrations_collection",
            new_callable=AsyncMock,
            side_effect=Exception("Connection error"),
        ):
            result = await manager.run_all_migrations()

            assert result is False

    @pytest.mark.asyncio
    async def test_get_migration_status_success(self):
        """Test get_migration_status with successful execution"""
        mock_client = MagicMock()

        mock_file1 = MagicMock()
        mock_file1.name = "001-create-conversation-collection.py"
        mock_file2 = MagicMock()
        mock_file2.name = "002-create-reference-document-collection.py"

        manager = MigrationManager(client=mock_client)

        with patch.object(
            manager, "ensure_migrations_collection", new_callable=AsyncMock
        ):
            with patch.object(manager, "get_applied_migrations", return_value=["001"]):
                with patch.object(
                    manager,
                    "get_migration_files",
                    return_value=[mock_file1, mock_file2],
                ):
                    # Fix the side_effect to return the correct versions
                    def mock_get_version(file_path):
                        if file_path.name == "001-create-conversation-collection.py":
                            return "001"
                        elif (
                            file_path.name
                            == "002-create-reference-document-collection.py"
                        ):
                            return "002"
                        return "000"

                    with patch.object(
                        manager, "get_migration_version", side_effect=mock_get_version
                    ):
                        result = await manager.get_migration_status()

                        assert result["total_migrations"] == 2
                        assert result["applied_migrations"] == 1
                        assert result["applied_versions"] == ["001"]
                        assert result["all_versions"] == ["001", "002"]
                        assert len(result["pending_migrations"]) == 1
                        assert result["pending_migrations"][0]["version"] == "002"

    @pytest.mark.asyncio
    async def test_get_migration_status_exception(self):
        """Test get_migration_status when exception occurs"""
        mock_client = MagicMock()
        manager = MigrationManager(client=mock_client)

        with patch.object(
            manager,
            "ensure_migrations_collection",
            new_callable=AsyncMock,
            side_effect=Exception("Connection error"),
        ):
            result = await manager.get_migration_status()

            assert "error" in result
            assert "Connection error" in result["error"]


class TestMigrationFilePattern:
    """Test cases for migration file pattern regex"""

    def test_migration_file_pattern_valid(self):
        """Test valid migration file patterns"""
        valid_patterns = [
            "001-create-conversation-collection.py",
            "002-create-reference-document-collection.py",
            "000-create-migration-history-table.py",
            "999-any-migration-name.py",
        ]

        for pattern in valid_patterns:
            assert MIGRATION_FILE_PATTERN.match(pattern) is not None

    def test_migration_file_pattern_invalid(self):
        """Test invalid migration file patterns"""
        invalid_patterns = [
            "01-create-conversation-collection.py",  # Only 2 digits
            "0001-create-conversation-collection.py",  # 4 digits
            "001create-conversation-collection.py",  # No dash
            "abc-create-conversation-collection.py",  # Non-numeric prefix
            "__init__.py",  # Special file
            "001-create-conversation-collection",  # No .py extension
        ]

        for pattern in invalid_patterns:
            assert MIGRATION_FILE_PATTERN.match(pattern) is None
