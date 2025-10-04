"""
Pytest configuration and shared fixtures for utility function tests.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

# Set test environment variables before importing anything
os.environ.update(
    {
        "ENV__ENV": "TEST",
        "LOG__LEVEL": "DEBUG",
        "WEAVIATE__HTTP_HOST": "localhost",
        "WEAVIATE__HTTP_PORT": "8080",
        "WEAVIATE__HTTP_SECURE": "false",
        "WEAVIATE__GRPC_HOST": "localhost",
        "WEAVIATE__GRPC_PORT": "50051",
        "WEAVIATE__GRPC_SECURE": "false",
        "WEAVIATE__CONCURRENT_REQUESTS": "10",
        "OPENAI__API_KEY": "test-key",
        "OPENAI__ORGANIZATION_ID": "test-org",
        "OPENAI__RATE_LIMIT": "100",
        "OPENAI__WINDOW_SECONDS": "60",
        "LANGSMITH__TRACING": "false",
        "LANGSMITH__ENDPOINT": "https://api.smith.langchain.com",
        "LANGSMITH__API_KEY": "test-key",
        "LANGSMITH__PROJECT": "test-project",
        "AWS__REGION": "us-east-1",
        "AWS__ACCESS_KEY_ID": "test-key",
        "AWS__SECRET_ACCESS_KEY": "test-secret",
        "AWS__ENDPOINT_URL": "http://localhost:4566",
        "QUEUE__TRANSCRIBE_AND_SUMMARIZE_RESULTS_BUCKET": "test-bucket",
        "QUEUE__TRANSCRIPTION_RESULTS_QUEUE_URL": "http://localhost:4566/test-queue",
        "QUEUE__TRANSCRIBE_AND_SUMMARIZE_RESPONSE_QUEUE_URL": (
            "http://localhost:4566/response-queue"
        ),
        "LLM__MAX_CONCURRENT_LLM_CALLS": "10",
        "SLACK_ALERTS__ENABLED": "false",
        "SLACK_ALERTS__API_TOKEN": "test-token",
        "SLACK_ALERTS__CHANNEL_ID": "test-channel",
        "SLACK_ALERTS__LOG_LEVEL": "WARNING",
        "SERVER__HOST": "localhost",
        "SERVER__PORT": "8000",
    }
)


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "counselor", "content": "How are you feeling today?"},
        {"role": "client", "content": "I'm feeling anxious about work."},
        {
            "role": "counselor",
            "content": (
                "I understand. Can you tell me more about what's causing this anxiety?"
            ),
        },
    ]


@pytest.fixture
def sample_affirmation_messages():
    """Sample messages with affirmations for testing."""
    return [
        {"role": "counselor", "content": "That makes complete sense."},
        {
            "role": "counselor",
            "content": "I can completely understand why you'd feel this way.",
        },
        {"role": "counselor", "content": "Your feelings are absolutely valid."},
    ]


@pytest.fixture
def sample_client_messages():
    """Sample client messages for positivity testing."""
    return [
        {"role": "client", "content": "I feel terrible about this situation."},
        {"role": "client", "content": "I'm feeling much better now."},
        {"role": "client", "content": "I'm really happy today!"},
    ]


# API Test Fixtures
@pytest.fixture(autouse=True)
def mock_openai_clients():
    """Mock OpenAI clients for API tests."""
    with (
        patch(
            "app.core.dependencies.get_openai_text_generation_client"
        ) as mock_text_client,
        patch(
            "app.core.dependencies.get_openai_embedding_client"
        ) as mock_embedding_client,
    ):

        # Mock text generation client
        mock_text_client_instance = AsyncMock()
        mock_text_client.return_value = mock_text_client_instance

        # Mock embedding client
        mock_embedding_client_instance = AsyncMock()
        mock_embedding_client.return_value = mock_embedding_client_instance

        yield {
            "text_client": mock_text_client_instance,
            "embedding_client": mock_embedding_client_instance,
        }


@pytest.fixture(autouse=True)
def mock_weaviate_client():
    """Mock Weaviate client for API tests."""
    with patch("app.core.dependencies.get_weaviate_client") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance
        yield mock_client_instance
