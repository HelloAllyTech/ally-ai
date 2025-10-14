"""
Migration Manager for Lifeline AI
Handles manual migration files and Weaviate database tracking
"""

import importlib
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import weaviate.classes.query as wvq

from app.core.vector_db.constants import VectorDBCollectionNames
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Regex pattern for migration files: starts with 3 digits followed by a dash
MIGRATION_FILE_PATTERN = re.compile(r"^\d{3}-.*\.py$")


class MigrationManager:
    """Manages manual migration files and Weaviate database tracking"""

    def __init__(self, client: Any = None):
        self.client = client
        self.migrations_dir = Path(__file__).parent
        self.project_root = Path(__file__).parent.parent.parent

    async def ensure_migrations_collection(self):
        """Ensure the MigrationHistory collection exists"""
        collection_name = VectorDBCollectionNames.MIGRATION_HISTORY

        try:
            # Check if collection exists
            collections = await self.client.collections.list_all()
            existing_collections = [
                col.name if hasattr(col, "name") else str(col) for col in collections
            ]

            if collection_name not in existing_collections:
                logger.info(
                    f"MigrationHistory collection '{collection_name}' does not exist. "
                    "This is expected for first-time setup."
                )
                logger.info(
                    "Migration 000 will create the MigrationHistory collection."
                )
                # Don't raise an exception - allow the first migration to run
                return False
            else:
                logger.info("Migration history collection already exists")
                return True

        except Exception as e:
            logger.error(
                f"MigrationHistory collection check failed: {type(e).__name__}"
            )
            # For first-time setup, allow migrations to run even if we can't check
            logger.info("Allowing migrations to run for first-time setup")
            return False

    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions from Weaviate"""
        try:
            collection = self.client.collections.get(
                VectorDBCollectionNames.MIGRATION_HISTORY
            )

            # Query all migrations with completed status
            response = await collection.query.fetch_objects(
                filters=wvq.Filter.by_property("status").equal("completed"),
                limit=100,
                sort=wvq.Sort.by_property("version", ascending=True),
            )

            applied_versions = []
            for obj in response.objects:
                version = obj.properties.get("version")
                if version:
                    applied_versions.append(version)

            return applied_versions

        except Exception as e:
            logger.error(f"Failed to get applied migrations: {type(e).__name__}")
            return []

    def get_migration_files(self) -> List[Path]:
        """Get all migration files in the migrations directory"""
        migration_files = []

        if not self.migrations_dir.exists():
            return migration_files

        for file_path in self.migrations_dir.glob("*.py"):
            # Skip __init__.py and check if file matches migration pattern
            if file_path.name != "__init__.py" and MIGRATION_FILE_PATTERN.match(
                file_path.name
            ):
                migration_files.append(file_path)

        # Sort by filename (which includes the sequential number)
        migration_files.sort(key=lambda x: x.name)
        return migration_files

    def get_migration_version(self, file_path: Path) -> str:
        """Extract version number from migration filename"""
        # Extract the 3-digit version number from the filename
        match = MIGRATION_FILE_PATTERN.match(file_path.name)
        if match:
            return file_path.name[:3]  # First 3 characters are the version
        return "000"  # Fallback for invalid filenames

    async def record_migration(
        self, version: str, name: str, description: str, status: str = "completed"
    ):
        """Record a migration in the history"""
        try:
            collection = self.client.collections.get(
                VectorDBCollectionNames.MIGRATION_HISTORY
            )

            # Format current time in RFC3339 format
            current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

            # Create migration record
            await collection.data.insert(
                properties={
                    "version": version,
                    "name": name,
                    "description": description,
                    "status": status,
                    "created_at": current_time,
                    "completed_at": current_time,
                }
            )

            logger.info(f"Recorded migration: {version} - {name}")

        except Exception as e:
            logger.error(f"Failed to record migration: {type(e).__name__}")
            raise

    async def delete_migration_record(self, version: str):
        """Delete a migration record from the history"""
        try:
            collection = self.client.collections.get(
                VectorDBCollectionNames.MIGRATION_HISTORY
            )

            # Query for the migration version to get the object IDs
            response = await collection.query.fetch_objects(
                filters=wvq.Filter.by_property("version").equal(version),
                limit=100,
            )

            if response.objects:
                # Delete each object by ID
                for obj in response.objects:
                    await collection.data.delete_by_id(obj.uuid)
                logger.info(f"Deleted migration record for version: {version}")
            else:
                logger.info(f"No migration record found for version: {version}")

        except Exception as e:
            logger.error(f"Failed to delete migration record: {type(e).__name__}")
            raise

    async def run_migration_up(self, version: str) -> bool:
        """Run a specific migration up"""
        try:
            migration_files = self.get_migration_files()

            # Find the migration file for this version
            target_file = None
            for file_path in migration_files:
                if self.get_migration_version(file_path) == version:
                    target_file = file_path
                    break

            if not target_file:
                logger.error(f"Migration file not found for version: {version}")
                return False

            # Import and run the migration
            module_name = f"app.migrations.{target_file.stem}"

            # Add project root to path for imports
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))

            try:
                module = importlib.import_module(module_name)

                if hasattr(module, "up"):
                    logger.info(f"Running migration up: {version}")
                    await module.up(self.client)

                    # Record successful migration
                    filename = target_file.stem
                    name = filename.split("-", 1)[1] if "-" in filename else filename
                    await self.record_migration(
                        version=version,
                        name=name,
                        description=f"Manual migration: {name}",
                        status="completed",
                    )

                    logger.info(f"Migration {version} completed successfully")
                    return True
                else:
                    logger.error(f"Migration {version} does not have an 'up' function")
                    return False

            except Exception as e:
                logger.error(f"Migration {version} failed: {type(e).__name__}")
                # Record failed migration
                filename = target_file.stem
                name = filename.split("-", 1)[1] if "-" in filename else filename
                await self.record_migration(
                    version=version,
                    name=name,
                    description=f"Manual migration: {name}",
                    status="failed",
                )
                return False

        except Exception as e:
            logger.error(f"Failed to run migration {version}: {type(e).__name__}")
            return False

    async def run_migration_down(self, version: str) -> bool:
        """Run a specific migration down"""
        try:
            migration_files = self.get_migration_files()

            # Find the migration file for this version
            target_file = None
            for file_path in migration_files:
                if self.get_migration_version(file_path) == version:
                    target_file = file_path
                    break

            if not target_file:
                logger.error(f"Migration file not found for version: {version}")
                return False

            # Import and run the migration
            module_name = f"app.migrations.{target_file.stem}"

            # Add project root to path for imports
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))

            try:
                module = importlib.import_module(module_name)

                if hasattr(module, "down"):
                    logger.info(f"Running migration down: {version}")
                    await module.down(self.client)

                    # Delete migration record
                    await self.delete_migration_record(version)

                    logger.info(f"Migration {version} rolled back successfully")
                    return True
                else:
                    logger.error(f"Migration {version} does not have a 'down' function")
                    return False

            except Exception as e:
                logger.error(f"Migration {version} rollback failed: {type(e).__name__}")
                return False

        except Exception as e:
            logger.error(f"Failed to rollback migration {version}: {type(e).__name__}")
            return False

    async def run_all_migrations(self) -> bool:
        """Run all pending migrations"""
        try:
            await self.ensure_migrations_collection()

            applied_migrations = await self.get_applied_migrations()
            migration_files = self.get_migration_files()

            logger.info(f"Found {len(migration_files)} migration files")
            logger.info(f"Applied migrations: {applied_migrations}")

            success = True
            for file_path in migration_files:
                version = self.get_migration_version(file_path)

                if version not in applied_migrations:
                    logger.info(f"Running pending migration: {version}")
                    if not await self.run_migration_up(version):
                        success = False
                        break
                else:
                    logger.info(f"Migration {version} already applied, skipping")

            if success:
                logger.info("All migrations completed successfully")
            else:
                logger.error("Some migrations failed")

            return success

        except Exception as e:
            logger.error(f"Failed to run migrations: {type(e).__name__}")
            return False

    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        try:
            await self.ensure_migrations_collection()

            applied_migrations = await self.get_applied_migrations()
            migration_files = self.get_migration_files()

            status = {
                "total_migrations": len(migration_files),
                "applied_migrations": len(applied_migrations),
                "pending_migrations": [],
                "applied_versions": applied_migrations,
                "all_versions": [
                    self.get_migration_version(f) for f in migration_files
                ],
            }

            for file_path in migration_files:
                version = self.get_migration_version(file_path)
                if version not in applied_migrations:
                    status["pending_migrations"].append(
                        {"version": version, "file": file_path.name}
                    )

            return status

        except Exception as e:
            logger.error(f"Failed to get migration status: {type(e).__name__}")
            return {"error": str(e)}
