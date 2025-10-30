"""
Base test class for API endpoint tests.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.conversation import IdentifyResponse


class BaseAPITest:
    """Base class for API endpoint tests."""

    @pytest.fixture
    def client(self):
        """Create a test client with API key header."""
        client = TestClient(app)
        # Add API key header for all requests
        client.headers.update({"x-api-key": "test-api-key"})
        return client

    @pytest.fixture
    def mock_conversation_service(self):
        """Mock conversation service."""
        with patch("app.core.dependencies.get_conversation_service") as mock:
            service = AsyncMock()
            # Set default return values
            service.analyze.return_value = ("stage1", "nudge1")
            service.identify.return_value = IdentifyResponse(
                speaker0="client", speaker1="counselor"
            )
            mock.return_value = service
            yield service

    @pytest.fixture
    def mock_summary_service(self):
        """Mock summary service."""
        with patch("app.core.dependencies.get_summary_service") as mock:
            service = AsyncMock()
            mock.return_value = service
            yield service

    @pytest.fixture
    def mock_reference_document_service(self):
        """Mock reference document service."""
        with patch("app.core.dependencies.get_reference_document_service") as mock:
            service = AsyncMock()
            mock.return_value = service
            yield service

    @pytest.fixture
    def sample_chat_messages(self):
        """Sample chat messages for testing."""
        return [
            {"role": "counselor", "content": "How are you feeling today?"},
            {"role": "client", "content": "I'm feeling anxious about work."},
            {
                "role": "counselor",
                "content": (
                    "I understand. Can you tell me more about what's causing "
                    "this anxiety?"
                ),
            },
        ]

    @pytest.fixture
    def sample_analyze_request(self, sample_chat_messages):
        """Sample analyze request."""
        return {
            "latest_message": "I'm feeling anxious about work.",
            "chat_history": sample_chat_messages,
            "force_nudge": False,
        }

    @pytest.fixture
    def sample_identify_request(self, sample_chat_messages):
        """Sample identify request."""
        return {"chat_history": sample_chat_messages}

    @pytest.fixture
    def sample_summary_request(self, sample_chat_messages):
        """Sample summary request."""
        return {
            "chat_history": sample_chat_messages,
            "keys": None,
        }

    @pytest.fixture
    def sample_reference_document_create(self):
        """Sample reference document create request."""
        from uuid import uuid4

        return {
            "heading": "Test Document",
            "content": "This is a test document for reference.",
            "category": "test",
            "tags": ["test", "example"],
            "tenant_id": "test-tenant",
            "document_id": str(uuid4()),
        }

    @pytest.fixture
    def sample_reference_document_update(self):
        """Sample reference document update request."""
        return {
            "heading": "Updated Test Document",
            "content": "This is an updated test document.",
            "category": "updated",
            "tags": ["updated", "test"],
        }

    @pytest.fixture
    def sample_reference_document_search(self):
        """Sample reference document search request."""
        return {
            "query": "test search query",
            "limit": 10,
            # Remove filters to avoid validation error
        }
