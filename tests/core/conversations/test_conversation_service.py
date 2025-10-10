"""Tests for ConversationService."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.conversations.conversation_service import ConversationService
from app.exceptions.custom_exceptions import (
    ConversationAnalysisFailedException,
    NudgeGenerationFailedException,
    VectorDBFetchFailedException,
)
from app.schemas.common import ChatMessage
from app.schemas.conversation import IdentifyResponse


class TestConversationService:
    """Test cases for ConversationService."""

    @pytest.fixture
    def mock_text_generation_service(self):
        """Mock text generation service."""
        return AsyncMock()

    @pytest.fixture
    def mock_vector_db(self):
        """Mock vector database."""
        return AsyncMock()

    @pytest.fixture
    def conversation_service(self, mock_text_generation_service, mock_vector_db):
        """Create ConversationService instance with mocked dependencies."""
        return ConversationService(mock_text_generation_service, mock_vector_db)

    @pytest.fixture
    def sample_chat_messages(self):
        """Sample chat messages for testing."""
        return [
            ChatMessage(role="counselor", content="How are you feeling today?"),
            ChatMessage(role="client", content="I'm feeling anxious about work."),
            ChatMessage(
                role="counselor",
                content=(
                    "I understand. Can you tell me more about what's causing "
                    "this anxiety?"
                ),
            ),
        ]

    @pytest.fixture
    def sample_vector_db_result(self):
        """Sample vector database result."""
        mock_object = MagicMock()
        mock_object.properties = {
            "nudge": "Try asking about their coping strategies",
            "conversation": "Previous similar conversation",
            "stage": "stage1",
        }
        mock_result = MagicMock()
        mock_result.objects = [mock_object]
        return mock_result

    @pytest.mark.asyncio
    async def test_analyze_success_with_nudge(
        self,
        conversation_service,
        mock_vector_db,
        mock_text_generation_service,
        sample_chat_messages,
        sample_vector_db_result,
    ):
        """Test successful analysis with nudge generation."""
        # Setup mocks
        mock_vector_db.fetch_relevant_conversations.return_value = (
            sample_vector_db_result
        )
        mock_text_generation_service.generate_nudge.return_value = "Generated nudge"

        # Execute
        stage, nudge = await conversation_service.analyze(
            "I'm feeling anxious", sample_chat_messages, force_nudge=False
        )

        # Assert
        assert stage == "stage1"
        assert nudge == "Generated nudge"
        mock_vector_db.fetch_relevant_conversations.assert_called_once_with(
            "I'm feeling anxious", top_k=1
        )
        mock_text_generation_service.generate_nudge.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_success_with_force_nudge(
        self,
        conversation_service,
        mock_vector_db,
        mock_text_generation_service,
        sample_chat_messages,
        sample_vector_db_result,
    ):
        """Test successful analysis with forced nudge generation."""
        # Setup mocks
        mock_vector_db.fetch_relevant_conversations.return_value = (
            sample_vector_db_result
        )
        mock_text_generation_service.generate_nudge.return_value = "Forced nudge"

        # Execute
        stage, nudge = await conversation_service.analyze(
            "I'm feeling anxious", sample_chat_messages, force_nudge=True
        )

        # Assert
        assert stage == "stage1"
        assert nudge == "Forced nudge"
        mock_text_generation_service.generate_nudge.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_success_no_nudge(
        self,
        conversation_service,
        mock_vector_db,
        mock_text_generation_service,
        sample_chat_messages,
    ):
        """Test successful analysis without nudge generation."""
        # Setup mocks - no nudge in properties
        mock_object = MagicMock()
        mock_object.properties = {
            "conversation": "Previous similar conversation",
            "stage": "stage1",
        }
        mock_result = MagicMock()
        mock_result.objects = [mock_object]
        mock_vector_db.fetch_relevant_conversations.return_value = mock_result

        # Execute
        stage, nudge = await conversation_service.analyze(
            "I'm feeling anxious", sample_chat_messages, force_nudge=False
        )

        # Assert
        assert stage == "stage1"
        assert nudge is None
        mock_text_generation_service.generate_nudge.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_no_relevant_conversations(
        self,
        conversation_service,
        mock_vector_db,
        mock_text_generation_service,
        sample_chat_messages,
    ):
        """Test analysis when no relevant conversations are found."""
        # Setup mocks - no relevant conversations
        mock_vector_db.fetch_relevant_conversations.return_value = None

        # Execute
        stage, nudge = await conversation_service.analyze(
            "I'm feeling anxious", sample_chat_messages, force_nudge=False
        )

        # Assert
        assert stage == "Unknown"
        assert nudge is None
        mock_text_generation_service.generate_nudge.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_vector_db_fetch_failed(
        self, conversation_service, mock_vector_db, sample_chat_messages
    ):
        """Test analysis when vector DB fetch fails."""
        # Setup mocks
        mock_vector_db.fetch_relevant_conversations.side_effect = (
            VectorDBFetchFailedException("Vector DB error")
        )

        # Execute and assert
        with pytest.raises(ConversationAnalysisFailedException) as exc_info:
            await conversation_service.analyze(
                "I'm feeling anxious", sample_chat_messages, force_nudge=False
            )

        assert "Failed to fetch relevant conversations" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analyze_nudge_generation_failed(
        self,
        conversation_service,
        mock_vector_db,
        mock_text_generation_service,
        sample_chat_messages,
        sample_vector_db_result,
    ):
        """Test analysis when nudge generation fails."""
        # Setup mocks
        mock_vector_db.fetch_relevant_conversations.return_value = (
            sample_vector_db_result
        )
        mock_text_generation_service.generate_nudge.side_effect = (
            NudgeGenerationFailedException("Nudge generation error")
        )

        # Execute and assert
        with pytest.raises(ConversationAnalysisFailedException) as exc_info:
            await conversation_service.analyze(
                "I'm feeling anxious", sample_chat_messages, force_nudge=False
            )

        assert "Failed to generate nudge" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analyze_empty_objects(
        self,
        conversation_service,
        mock_vector_db,
        mock_text_generation_service,
        sample_chat_messages,
    ):
        """Test analysis when vector DB returns empty objects."""
        # Setup mocks - empty objects
        mock_result = MagicMock()
        mock_result.objects = []
        mock_vector_db.fetch_relevant_conversations.return_value = mock_result

        # Execute
        stage, nudge = await conversation_service.analyze(
            "I'm feeling anxious", sample_chat_messages, force_nudge=False
        )

        # Assert
        assert stage is None
        assert nudge is None
        mock_text_generation_service.generate_nudge.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_none_objects(
        self,
        conversation_service,
        mock_vector_db,
        mock_text_generation_service,
        sample_chat_messages,
    ):
        """Test analysis when vector DB returns None objects."""
        # Setup mocks - None objects
        mock_result = MagicMock()
        mock_result.objects = None
        mock_vector_db.fetch_relevant_conversations.return_value = mock_result

        # Execute
        stage, nudge = await conversation_service.analyze(
            "I'm feeling anxious", sample_chat_messages, force_nudge=False
        )

        # Assert
        assert stage is None
        assert nudge is None
        mock_text_generation_service.generate_nudge.assert_not_called()

    @pytest.mark.asyncio
    async def test_identify_success(
        self, conversation_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test successful user identification."""
        # Setup mocks
        expected_response = IdentifyResponse(speaker0="client", speaker1="counselor")
        mock_text_generation_service.identify_user.return_value = expected_response

        # Execute
        result = await conversation_service.identify(sample_chat_messages)

        # Assert
        assert result == expected_response
        mock_text_generation_service.identify_user.assert_called_once_with(
            sample_chat_messages
        )

    @pytest.mark.asyncio
    async def test_identify_with_empty_history(
        self, conversation_service, mock_text_generation_service
    ):
        """Test user identification with empty chat history."""
        # Setup mocks
        expected_response = IdentifyResponse(speaker0="unknown", speaker1="unknown")
        mock_text_generation_service.identify_user.return_value = expected_response

        # Execute
        result = await conversation_service.identify([])

        # Assert
        assert result == expected_response
        mock_text_generation_service.identify_user.assert_called_once_with([])
