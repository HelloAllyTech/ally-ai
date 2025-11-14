# Lifeline AI

**Lifeline AI** is an advanced AI service designed to function as a copilot for mental health counselors. This project leverages FastAPI to deliver a robust and scalable API backend, integrating custom middleware and enhanced logging capabilities to ensure reliable performance and traceability. With environment-based configuration and containerization via Docker, Lifeline AI is built for seamless deployment and development. Managed with Poetry for dependency handling, the service provides intelligent insights and support tools to empower mental health professionals in their daily practice.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Getting Started
## 📋 Table of Contents

- [Overview](#-overview)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Development Setup](#-development-setup)
- [Database Management](#-database-management)
- [Code Quality & Testing](#-code-quality--testing)
- [Docker Deployment](#-docker-deployment)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

## ✨ Overview

- **AI-Powered Counseling Support**: Intelligent insights and analysis for mental health professionals
- **FastAPI Backend**: High-performance, async API with automatic documentation
- **Vector Database Integration**: Weaviate for semantic search and conversation analysis
- **Migration System**: Comprehensive database schema management with rollback capabilities
- **Environment-Based Configuration**: Flexible deployment across different environments
- **Docker Support**: Containerized deployment for consistent environments
- **Code Quality Tools**: Pre-commit hooks, linting, and formatting
- **HIPAA Compliance**: Secure handling of sensitive mental health data

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12+** - [Download Python](https://www.python.org/downloads/)
- **Poetry** - [Install Poetry](https://python-poetry.org/docs/#installation)
- **Docker** - [Install Docker](https://docs.docker.com/get-docker/)
- **Git** - [Install Git](https://git-scm.com/downloads)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone git@github.com:KeyValueSoftwareSystems/lifeline-ai.git
cd lifeline-ai
```

### 2. Install Dependencies

```bash
poetry install
```

### 3. Set Up Database

```bash
# Run all pending migrations
poetry run python scripts/migrate.py all
```

### 4. Start the Application

```bash
poetry run python app/main.py
```

The application will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs

## 🛠️ Development Setup

### Environment Configuration

Create a `.env` file in the project root with your configuration:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your specific settings
nano .env
```

### Database Setup

#### Weaviate Connection

For local development, you can use Docker Compose to run Weaviate:

```bash
# Start Weaviate locally
docker-compose up -d

# Or connect to remote Weaviate via SSH tunnel
ssh -L 8080:localhost:8080 -L 50051:localhost:50051 your-remote-server
```

#### Migration Management

```bash
# Check migration status
poetry run python scripts/migrate.py status

# View migration history
poetry run python scripts/migrate.py history

# Test database connection
poetry run python scripts/test_weaviate_connection.py

# List reference documents
poetry run python scripts/list_reference_documents.py
```

### Development Workflow

```bash
# Install pre-commit hooks
pre-commit install

# Run code formatting
poetry run black app/
poetry run isort app/

# Run linting
poetry run flake8 app/

# Run all pre-commit hooks
pre-commit run --all-files
```

## 🗄️ Database Management

### Migration System Overview

Lifeline AI uses a comprehensive migration system to manage Weaviate database schema changes:

- **Location**: `app/migrations/` directory
- **Naming Convention**: `{number}-{description}.py` (e.g., `001-create-conversation-collection.py`)
- **Tracking**: Migration history stored in Weaviate `MigrationHistory` collection
- **Rollback Support**: Each migration has both `up` and `down` functions

### Migration Commands

#### Generate New Migration

```bash
# Create a new migration file
poetry run python scripts/migrate.py generate "add-user-session-collection"
```

This creates a file like `003-add-user-session-collection.py` with boilerplate code.

#### Execute Migrations

```bash
# Run the next pending migration
poetry run python scripts/migrate.py up

# Run all pending migrations
poetry run python scripts/migrate.py all

# Rollback last migration
poetry run python scripts/migrate.py down
```

#### Monitor Migrations

```bash
# Check current status
poetry run python scripts/migrate.py status

# View detailed history
poetry run python scripts/migrate.py history
```

### Migration Status Tracking

The system tracks migration status with visual indicators:

- **completed**: ✅ Successfully applied
- **failed**: ❌ Failed during execution
- **running**: 🔄 Currently executing
- **pending**: ⏳ Not yet started

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

#### Adding Properties to Existing Collection

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

### Migration Best Practices

1. **Always implement both `up` and `down` functions**
2. **Test migrations thoroughly before applying to production**
3. **Use descriptive migration names**
4. **Keep migrations small and focused**
5. **Never modify existing migration files after they've been applied**

> **Note**: Weaviate doesn't support property deletion in current versions. To "delete" a property, either recreate the collection or simply stop using the property (it will remain in storage but unused).

## 🧪 Code Quality & Testing

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files

# Run specific hooks
pre-commit run black --all-files
pre-commit run isort --all-files
pre-commit run flake8 --all-files
```

### Code Formatting & Linting

```bash
# Format code with Black
poetry run black app/

# Sort imports with isort
poetry run isort app/

# Lint with flake8
poetry run flake8 app/ --count --show-source --statistics

# Check formatting without changes
poetry run black --check app/
poetry run isort --check-only app/
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=app --cov-report=term-missing --cov-report=html

# Run specific test file
poetry run pytest tests/core/test_specific.py
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
# Build the application image
docker build -t lifeline-ai .

# Build with specific tag
docker build -t lifeline-ai:v1.0.0 .
```

### Run Docker Container

```bash
# Run the container
docker run -p 8000:8000 lifeline-ai

# Run with environment variables
docker run -p 8000:8000 -e ENVIRONMENT=production lifeline-ai

# Run in detached mode
docker run -d -p 8000:8000 --name lifeline-ai lifeline-ai
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Full Local Stack

To run the API together with LocalStack (SQS/S3) and Weaviate, you need to start LocalStack separately first, then use the bundled compose file.

#### Step 1: Start Shared LocalStack

LocalStack must be running separately before starting the application stack. This allows multiple repositories to share the same LocalStack instance without port conflicts.

Start shared LocalStack using Docker Run:

```bash
# Start shared LocalStack container
docker run -d \
  --name shared-localstack \
  --restart unless-stopped \
  -p 4566:4566 \
  -p 4510-4559:4510-4559 \
  -e SERVICES=sqs,s3 \
  -e DEBUG=1 \
  -e AWS_DEFAULT_REGION=ap-southeast-1 \
  -e AWS_ACCESS_KEY_ID=test \
  -e AWS_SECRET_ACCESS_KEY=test \
  -v shared_localstack_data:/var/lib/localstack \
  localstack/localstack:latest

# Verify it's running
curl http://localhost:4566/_localstack/health
```

**Managing Shared LocalStack**

```bash
# Check if LocalStack is running
docker ps | grep shared-localstack

# View logs
docker logs -f shared-localstack

# Stop LocalStack
docker stop shared-localstack

# Start LocalStack (if already created)
docker start shared-localstack

# Remove LocalStack (if needed)
docker rm -f shared-localstack
```

#### Step 2: Start Application Stack

Once LocalStack is running, start the application:

```bash
docker compose -f docker-compose.full.yml up --build
```

Key details:

- The API is exposed on `http://localhost:8001`.
- The compose file injects `WEAVIATE__HTTP_HOST=weaviate`, so no manual `.env` change is needed for the in-cluster hostname.
- `scripts/bootstrap_localstack.sh` runs on container start and waits for LocalStack before creating the required queues (`TRANSCRIPTION_RESULTS_QUEUE`, `TRANSCRIBE_AND_SUMMARIZE_RESPONSE_QUEUE`) and the S3 bucket defined by `QUEUE__TRANSCRIBE_AND_SUMMARIZE_RESULTS_BUCKET`.
- Ensure your `.env` contains the queue URLs/bucket name you want the application to use; the bootstrap script derives queue names from those URLs when present.
- The application connects to LocalStack via `host.docker.internal:4566` to access the shared instance running on your host machine.
- When you shut the stack down, volumes keep Weaviate data so subsequent starts are fast.

To tear everything down:

```bash
# Stop application stack
docker compose -f docker-compose.full.yml down

# Stop shared LocalStack (if needed)
docker stop shared-localstack
```

**Note**: The shared LocalStack instance persists across application restarts, so queues and buckets created by the bootstrap script will remain available.

## 📚 API Documentation

### Interactive Documentation

Once the application is running, access the automatically generated API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 📁 Project Structure

```
lifeline-ai/
├── app/                          # Main application code
│   ├── api/                      # API routes and endpoints
│   ├── core/                     # Core business logic
│   │   ├── conversations/        # Conversation analysis services
│   │   ├── embeddings/           # Embedding services
│   │   ├── text_generations/     # Text generation services
│   │   ├── vector_db/            # Vector database services
│   │   └── config.py             # Application configuration
│   ├── migrations/               # Database migration files
│   ├── schemas/                  # Pydantic models
│   ├── utils/                    # Utility functions
│   └── main.py                   # Application entry point
├── scripts/                      # Utility scripts
│   ├── migrate.py                # Migration management
│   ├── list_reference_documents.py # Database exploration
│   └── test_weaviate_connection.py # Connection testing
├── tests/                        # Test files
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile                    # Docker image definition
├── pyproject.toml                # Poetry configuration
└── README.md                     # This file
```

## 🤝 Contributing

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Run tests and linting**: `pre-commit run --all-files`
5. **Commit your changes**: `git commit -m "Add your feature"`
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Create a Pull Request**

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Include tests for new functionality
- Ensure all pre-commit hooks pass

### Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant error messages

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Contact the development team
- Check the API documentation at `/docs`

---

**Lifeline AI** - Empowering mental health professionals with AI-driven insights and support.
