#!/usr/bin/env python3
"""
Migration CLI for Lifeline AI
Handles manual migration generation and execution
"""

import argparse
import asyncio
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.vector_db.weaviate_client import WeaviateClient  # noqa: E402
from app.migrations.manager import MigrationManager  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def generate_migration(message: str) -> str:
    """Generate a new migration file"""
    migrations_dir = project_root / "app" / "migrations"
    migrations_dir.mkdir(exist_ok=True)

    # Get the next sequential number
    existing_files = list(migrations_dir.glob("*.py"))
    existing_files = [
        f for f in existing_files if f.name != "__init__.py" and f.name[0].isdigit()
    ]

    if existing_files:
        # Extract numbers from existing files and find the highest
        numbers = []
        for file_path in existing_files:
            try:
                number = int(file_path.name.split("-")[0])
                numbers.append(number)
            except ValueError:
                continue

        next_number = max(numbers) + 1 if numbers else 1
    else:
        next_number = 1

    # Create filename: number-message.py
    safe_message = "".join(c if c.isalnum() or c in "-_" else "_" for c in message)
    filename = f"{next_number:03d}-{safe_message}.py"
    file_path = migrations_dir / filename

    # Create migration template
    template = textwrap.dedent(
        f'''\
        """
        Migration: {message}
        Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """

        from app.utils.logger import get_logger

        logger = get_logger(__name__)


        async def up(client):
            """
            Run the migration up.

            Args:
                client: Weaviate client instance
            """
            logger.info("Running migration up: {message}")

            # TODO: Implement your migration logic here
            # Example:
            # collection = client.collections.get("YourCollection")
            # await collection.config.add_property(...)

            logger.info("Migration up completed: {message}")


        async def down(client):
            """
            Run the migration down (rollback).

            Args:
                client: Weaviate client instance
            """
            logger.info("Running migration down: {message}")

            # TODO: Implement your rollback logic here
            # Example:
            # collection = client.collections.get("YourCollection")
            # await collection.config.remove_property(...)

            logger.info("Migration down completed: {message}")
        '''
    )

    # Write the file
    with open(file_path, "w") as f:
        f.write(template)

    logger.info(f"Created migration file: {file_path}")
    return str(file_path)


async def run_migration_up():
    """Run the next pending migration up"""
    try:
        # Initialize Weaviate client
        WeaviateClient.create_client()
        client = WeaviateClient.get_client()
        await WeaviateClient.connect(client)

        manager = MigrationManager(client)
        await manager.ensure_migrations_collection()

        # Get migration status
        status = await manager.get_migration_status()

        if status.get("pending_migrations"):
            next_migration = status["pending_migrations"][0]
            version = next_migration["version"]

            logger.info(f"Running migration up: {version}")
            success = await manager.run_migration_up(version)

            if success:
                logger.info("Migration up completed successfully")
            else:
                logger.error("Migration up failed")
                sys.exit(1)
        else:
            logger.info("No pending migrations to run")

    except Exception:
        logger.exception("Migration up failed")
        sys.exit(1)
    finally:
        if "client" in locals():
            await WeaviateClient.close(client)


async def run_migration_down():
    """Run the last applied migration down"""
    try:
        # Initialize Weaviate client
        WeaviateClient.create_client()
        client = WeaviateClient.get_client()
        await WeaviateClient.connect(client)

        manager = MigrationManager(client)
        await manager.ensure_migrations_collection()

        # Get applied migrations
        applied_migrations = await manager.get_applied_migrations()

        if applied_migrations:
            last_migration = applied_migrations[-1]  # Get the last applied migration

            logger.info(f"Running migration down: {last_migration}")
            success = await manager.run_migration_down(last_migration)

            if success:
                logger.info("Migration down completed successfully")
            else:
                logger.error("Migration down failed")
                sys.exit(1)
        else:
            logger.info("No applied migrations to rollback")

    except Exception:
        logger.exception("Migration down failed")
        sys.exit(1)
    finally:
        if "client" in locals():
            await WeaviateClient.close(client)


async def run_all_migrations():
    """Run all pending migrations"""
    try:
        # Initialize Weaviate client
        WeaviateClient.create_client()
        client = WeaviateClient.get_client()
        await WeaviateClient.connect(client)

        manager = MigrationManager(client)
        success = await manager.run_all_migrations()

        if success:
            logger.info("All migrations completed successfully")
        else:
            logger.error("Some migrations failed")
            sys.exit(1)

    except Exception:
        logger.exception("Migration failed")
        sys.exit(1)
    finally:
        if "client" in locals():
            await WeaviateClient.close(client)


async def show_migration_status():
    """Show current migration status"""
    try:
        # Initialize Weaviate client
        WeaviateClient.create_client()
        client = WeaviateClient.get_client()
        await WeaviateClient.connect(client)

        manager = MigrationManager(client)
        status = await manager.get_migration_status()

        print("\nMigration Status:")
        print("=" * 50)
        print(f"Total migrations: {status.get('total_migrations', 0)}")
        print(f"Applied migrations: {status.get('applied_migrations', 0)}")
        print(f"Pending migrations: {len(status.get('pending_migrations', []))}")

        if status.get("pending_migrations"):
            print("\nPending Migrations:")
            for migration in status["pending_migrations"]:
                print(f"   • {migration['version']}: {migration['file']}")

        if status.get("applied_versions"):
            print("\nApplied Migrations:")
            for version in status["applied_versions"]:
                print(f"   • {version}")

    except Exception:
        logger.exception("Failed to get migration status")
        sys.exit(1)
    finally:
        if "client" in locals():
            await WeaviateClient.close(client)


async def show_migration_history():
    """Show migration history from Weaviate"""
    try:
        # Initialize Weaviate client
        WeaviateClient.create_client()
        client = WeaviateClient.get_client()
        await WeaviateClient.connect(client)

        manager = MigrationManager(client)
        await manager.ensure_migrations_collection()

        # Get migration history from Weaviate
        collection = client.collections.get("MigrationHistory")
        response = await collection.query.fetch_objects(
            limit=100,
            sort=client.collections.get("MigrationHistory").query.Sort.by_property(
                "created_at", ascending=False
            ),
        )

        print("\\n📋 Migration History:")
        print("=" * 50)

        if not response.objects:
            print("No migration history found")
        else:
            for obj in response.objects:
                props = obj.properties
                status_emoji = {
                    "completed": "✅",
                    "failed": "❌",
                    "running": "🔄",
                    "pending": "⏳",
                }.get(props.get("status", ""), "❓")

                print(
                    f"{status_emoji} Version {props.get('version', 'N/A')}: "
                    f"{props.get('name', 'N/A')}"
                )
                print(f"   Description: {props.get('description', 'N/A')}")
                print(f"   Status: {props.get('status', 'N/A')}")
                print(f"   Created: {props.get('created_at', 'N/A')}")
                if props.get("completed_at"):
                    print(f"   Completed: {props.get('completed_at', 'N/A')}")
                print()

    except Exception:
        logger.exception("Failed to get migration history")
        sys.exit(1)
    finally:
        if "client" in locals():
            await WeaviateClient.close(client)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Lifeline AI Migration CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate migration command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate a new migration file"
    )
    generate_parser.add_argument("message", help="Migration description message")

    # Migration execution commands
    subparsers.add_parser("up", help="Run the next pending migration up")
    subparsers.add_parser("down", help="Run the last applied migration down")
    subparsers.add_parser("all", help="Run all pending migrations")

    # Status and history commands
    subparsers.add_parser("status", help="Show current migration status")
    subparsers.add_parser("history", help="Show migration history")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the appropriate command
    if args.command == "generate":
        file_path = generate_migration(args.message)
        print(f"Created migration file: {file_path}")
    elif args.command == "up":
        asyncio.run(run_migration_up())
    elif args.command == "down":
        asyncio.run(run_migration_down())
    elif args.command == "all":
        asyncio.run(run_all_migrations())
    elif args.command == "status":
        asyncio.run(show_migration_status())
    elif args.command == "history":
        asyncio.run(show_migration_history())


if __name__ == "__main__":
    main()
