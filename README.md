# Lifeline AI

**Lifeline AI** is an advanced AI service designed to function as a copilot for mental health counselors. This project leverages FastAPI to deliver a robust and scalable API backend, integrating custom middleware and enhanced logging capabilities to ensure reliable performance and traceability. With environment-based configuration and containerization via Docker, Lifeline AI is built for seamless deployment and development. Managed with Poetry for dependency handling, the service provides intelligent insights and support tools to empower mental health professionals in their daily practice.

## Getting Started

### Prerequisites

- **Python 3.12** (or higher)
- **Poetry** (for dependency management)
- **Docker** (for containerization)

### Local Development


1. **Clone the Repository:**
```bash
git clone git@github.com:KeyValueSoftwareSystems/lifeline-ai.git
```

2. **Install Dependencies with Poetry:**
```bash
poetry install
```

3. **Run Database Migrations:**

Before starting the application, you need to run the database migrations:

```bash
# Run all pending migrations
poetry run python scripts/migrate.py all

# Or run migrations one by one
poetry run python scripts/migrate.py up
```

4. **Run the Application:**

You can start the application in development mode using:

```bash
poetry run python app/main.py
```

The application will start using Uvicorn on the host and port defined in your configuration (default: http://localhost:8000).

API Documentation:

Once the application is running, you can access the automatically generated API documentation at:
    Swagger UI: http://localhost:8000/docs

### Docker

This project includes a Dockerfile to build a containerized version of the application.

Build the Docker Image:
```bash
docker build -t my-fastapi-app .
```
Run the Docker Container:
```bash
docker run -p 8000:8000 my-fastapi-app
```
The application will be available at http://localhost:8000.


# Test Black formatting
poetry run black --check app/

# Test isort import sorting
poetry run isort --check-only app/

# Test flake8 linting (optional - many issues in your codebase)
poetry run flake8 app/ --count --show-source --statistics

# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run isort --all-files

# Any commit will automatically trigger hooks
git add .
git commit -m "your commit message"

## Database Migrations

Lifeline AI uses a manual migration system to manage Weaviate database schema changes. The migration system tracks applied migrations in the database itself and provides full rollback capabilities.

### Migration System Overview

- **Migration Files**: Located in `app/migrations/` directory
- **File Naming**: `{number}-{description}.py` (e.g., `001-create-conversation-collection.py`)
- **Database Tracking**: Migration history stored in Weaviate `MigrationHistory` collection
- **Rollback Support**: Each migration has both `up` and `down` functions

### Migration Commands

#### Generate a New Migration
```bash
# Create a new migration file
poetry run python scripts/migrate.py generate "add-user-session-collection"
```

This creates a file like `003-add-user-session-collection.py` with boilerplate code for `up` and `down` functions.

#### Run Migrations
```bash
# Run the next pending migration up
poetry run python scripts/migrate.py up

# Run the last applied migration down (rollback)
poetry run python scripts/migrate.py down

# Run all pending migrations
poetry run python scripts/migrate.py all
```

#### Check Migration Status
```bash
# Show current migration status
poetry run python scripts/migrate.py status

# Show migration history from database
poetry run python scripts/migrate.py history
```

### Migration File Structure

Each migration file contains two main functions:

```python
"""
Migration: your-migration-description
Generated on: 2024-01-15 10:30:00
"""

from app.utils.logger import get_logger

logger = get_logger(__name__)


async def up(client):
    """
    Run the migration up.

    Args:
        client: Weaviate client instance
    """
    logger.info("Running migration up: your-migration-description")

    # TODO: Implement your migration logic here
    # Example:
    # collection = client.collections.get("YourCollection")
    # await collection.config.add_property(...)

    logger.info("Migration up completed: your-migration-description")


async def down(client):
    """
    Run the migration down (rollback).

    Args:
        client: Weaviate client instance
    """
    logger.info("Running migration down: your-migration-description")

    # TODO: Implement your rollback logic here
    # Example:
    # collection = client.collections.get("YourCollection")
    # await collection.config.remove_property(...)

    logger.info("Migration down completed: your-migration-description")
```

### Migration Best Practices

1. **Always implement both `up` and `down` functions**
2. **Test migrations thoroughly before applying to production**
3. **Use descriptive migration names**
4. **Keep migrations small and focused**
5. **Never modify existing migration files after they've been applied**

### Migration Status Tracking

The system tracks migration status in the Weaviate database:

- **completed**: ✅ Successfully applied
- **failed**: ❌ Failed during execution
- **running**: 🔄 Currently executing
- **pending**: ⏳ Not yet started

### Example Migration Workflow

1. **Create a new migration:**
   ```bash
   poetry run python scripts/migrate.py generate "add-new-property-to-conversation"
   ```

2. **Edit the generated file** in `app/migrations/` to implement your changes

3. **Run the migration:**
   ```bash
   poetry run python scripts/migrate.py up
   ```

4. **Check status:**
   ```bash
   poetry run python scripts/migrate.py status
   ```

5. **If needed, rollback:**
   ```bash
   poetry run python scripts/migrate.py down
   ```

### Migration Examples

#### Creating a New Collection
```python
async def up(client):
    await client.collections.create(
        name="UserSession",
        properties=[
            wvc.Property(name="user_id", data_type=wvc.DataType.TEXT),
            wvc.Property(name="session_data", data_type=wvc.DataType.TEXT),
        ]
    )

async def down(client):
    await client.collections.delete("UserSession")
```

#### Adding a Property to Existing Collection
```python
async def up(client):
    collection = client.collections.get("Conversation")
    await collection.config.add_property(
        wvc.Property(name="new_field", data_type=wvc.DataType.TEXT)
    )

async def down(client):
    collection = client.collections.get("Conversation")
    await collection.config.remove_property("new_field")
```

This migration system provides full control over your Weaviate database schema evolution while maintaining HIPAA compliance and providing comprehensive tracking and rollback capabilities.

Note: If you need to "delete" a property, you must use the collection recreation approach as in current Weaviate versions, there is no delete_property API anymore..
Alternatives - Ignore the property: Just stop writing or reading it. It will remain in storage but unused.
