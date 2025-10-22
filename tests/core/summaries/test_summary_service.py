"""Tests for SummaryService."""

from unittest.mock import AsyncMock

import pytest

from app.core.summaries.summary_service import SummaryService
from app.exceptions.custom_exceptions import (
    ContentEnhancementFailedException,
    CounselorTrainingAnalysisFailedException,
    LLMInvocationFailedException,
    SummarizationFailedException,
    SummaryNoteFailedException,
)
from app.schemas.common import ChatMessage
from app.schemas.summary import (
    DynamicSummaryNoteResponse,
    SummaryNoteAndTagsResponse,
    Tag,
)


class TestSummaryService:
    """Test cases for SummaryService."""

    @pytest.fixture
    def mock_text_generation_service(self):
        """Mock text generation service."""
        return AsyncMock()

    @pytest.fixture
    def summary_service(self, mock_text_generation_service):
        """Create SummaryService instance with mocked dependencies."""
        return SummaryService(mock_text_generation_service)

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

    @pytest.mark.asyncio
    async def test_generate_summary_and_tags_success(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test successful summary and tags generation."""
        # Setup mocks
        expected_response = SummaryNoteAndTagsResponse(
            session_summary="Test summary",
            tags=[Tag(tag="anxiety", positivity_rating=2)],
            call_quality=85,
        )
        mock_text_generation_service.generate_summary_notes.return_value = (
            expected_response
        )

        # Execute
        result = await summary_service.generate_summary_and_tags(sample_chat_messages)

        # Assert
        assert result == expected_response
        mock_text_generation_service.generate_summary_notes.assert_called_once_with(
            sample_chat_messages, None, chat_id=None
        )

    @pytest.mark.asyncio
    async def test_generate_summary_and_tags_with_keys(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test summary generation with specific keys."""
        # Setup mocks
        keys = ["session_summary", "tags"]
        expected_response = DynamicSummaryNoteResponse(
            fields={
                "session_summary": "Test summary",
                "tags": [{"tag": "anxiety", "positivity_rating": 2}],
            }
        )
        mock_text_generation_service.generate_summary_notes.return_value = (
            expected_response
        )

        # Execute
        result = await summary_service.generate_summary_and_tags(
            sample_chat_messages, None, keys
        )

        # Assert
        assert result == expected_response
        mock_text_generation_service.generate_summary_notes.assert_called_once_with(
            sample_chat_messages, keys, chat_id=None
        )

    @pytest.mark.asyncio
    async def test_generate_summary_and_tags_failed(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test summary generation failure."""
        # Setup mocks
        mock_text_generation_service.generate_summary_notes.side_effect = (
            SummaryNoteFailedException("Summary generation error")
        )

        # Execute and assert
        with pytest.raises(SummarizationFailedException) as exc_info:
            await summary_service.generate_summary_and_tags(sample_chat_messages)

        assert "Failed to generate the summary" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_enhance_content_success(
        self, summary_service, mock_text_generation_service
    ):
        """Test successful content enhancement."""
        # Setup mocks
        content = "Original content"
        enhanced_content = "Enhanced content"
        mock_text_generation_service.enhance_content.return_value = enhanced_content

        # Execute
        result = await summary_service.enhance_content(content)

        # Assert
        assert result == enhanced_content
        mock_text_generation_service.enhance_content.assert_called_once_with(
            content, chat_id=None
        )

    @pytest.mark.asyncio
    async def test_enhance_content_failed(
        self, summary_service, mock_text_generation_service
    ):
        """Test content enhancement failure."""
        # Setup mocks
        content = "Original content"
        mock_text_generation_service.enhance_content.side_effect = (
            ContentEnhancementFailedException("Enhancement error")
        )

        # Execute and assert
        with pytest.raises(SummarizationFailedException) as exc_info:
            await summary_service.enhance_content(content)

        assert "Failed to enhance content" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_tag_positivity_ratings_success(
        self, summary_service, mock_text_generation_service
    ):
        """Test successful tag positivity ratings retrieval."""
        # Setup mocks
        tags = ["anxiety", "depression"]
        mock_ratings = [
            {"tag": "anxiety", "positivity_rating": 2},
            {"tag": "depression", "positivity_rating": 1},
        ]
        mock_text_generation_service.get_tag_positivity_ratings.return_value = (
            mock_ratings
        )

        # Execute
        result = await summary_service.get_tag_positivity_ratings(tags)

        # Assert
        expected_tags = [
            Tag(tag="anxiety", positivity_rating=2),
            Tag(tag="depression", positivity_rating=1),
        ]
        assert result == expected_tags
        mock_text_generation_service.get_tag_positivity_ratings.assert_called_once_with(
            tags, chat_id=None
        )

    @pytest.mark.asyncio
    async def test_get_tag_positivity_ratings_empty_list(
        self, summary_service, mock_text_generation_service
    ):
        """Test tag positivity ratings with empty list."""
        # Setup mocks
        mock_text_generation_service.get_tag_positivity_ratings.return_value = []

        # Execute
        result = await summary_service.get_tag_positivity_ratings([])

        # Assert
        assert result == []
        mock_text_generation_service.get_tag_positivity_ratings.assert_called_once_with(
            [], chat_id=None
        )

    @pytest.mark.asyncio
    async def test_get_tag_positivity_ratings_failed(
        self, summary_service, mock_text_generation_service
    ):
        """Test tag positivity ratings failure."""
        # Setup mocks
        tags = ["anxiety"]
        mock_text_generation_service.get_tag_positivity_ratings.side_effect = Exception(
            "Rating error"
        )

        # Execute and assert
        with pytest.raises(SummarizationFailedException) as exc_info:
            await summary_service.get_tag_positivity_ratings(tags)

        assert "Failed to get positivity ratings for tags" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_success(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test successful simulation summary generation."""
        # Setup mocks
        goal = "Improve counseling skills"
        expected_response = {
            "improvements": ["Ask more open-ended questions"],
            "positives": ["Good rapport building"],
        }
        mock_text_generation_service.generate_simulation_summary.return_value = (
            expected_response
        )

        # Execute
        result = await summary_service.generate_simulation_summary(
            sample_chat_messages, goal
        )

        # Assert
        assert result == expected_response
        (
            mock_text_generation_service.generate_simulation_summary.assert_called_once_with(  # noqa: E501
                sample_chat_messages, goal, chat_id=None
            )
        )

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_failed(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test simulation summary generation failure."""
        # Setup mocks
        goal = "Improve counseling skills"
        mock_text_generation_service.generate_simulation_summary.side_effect = (
            LLMInvocationFailedException("LLM error")
        )

        # Execute and assert
        with pytest.raises(CounselorTrainingAnalysisFailedException) as exc_info:
            await summary_service.generate_simulation_summary(
                sample_chat_messages, goal
            )

        assert "Failed to generate counselor training analysis" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_empty_chat_history(
        self, summary_service, mock_text_generation_service
    ):
        """Test simulation summary with empty chat history."""
        # Setup mocks
        goal = "Improve counseling skills"
        expected_response = {"improvements": [], "positives": []}
        mock_text_generation_service.generate_simulation_summary.return_value = (
            expected_response
        )

        # Execute
        result = await summary_service.generate_simulation_summary([], goal)

        # Assert
        assert result == expected_response
        (
            mock_text_generation_service.generate_simulation_summary.assert_called_once_with(  # noqa: E501
                [], goal, chat_id=None
            )
        )
