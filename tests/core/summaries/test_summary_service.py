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
            ChatMessage(
                id="msg-1", role="counselor", content="How are you feeling today?"
            ),
            ChatMessage(
                id="msg-2", role="client", content="I'm feeling anxious about work."
            ),
            ChatMessage(
                id="msg-3",
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
            sample_chat_messages, None, chat_id=None, prompts=None, key_descriptions=None
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
            sample_chat_messages, keys, chat_id=None, prompts=None, key_descriptions=None
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
            content, chat_id=None, prompts=None
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
            tags, chat_id=None, prompts=None
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
            [], chat_id=None, prompts=None
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
    async def test_generate_scenario_evaluation_success(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test successful scenario evaluation generation."""
        mock_text_generation_service.generate_scenario_evaluation.return_value = {
            "improvements": ["Test improvement"],
            "positives": ["Test positive"],
            "message_tags": [
                {
                    "id": "msg-1",
                    "tags": [{"label": "Steady pacing", "category": "POSITIVE"}],
                }
            ],
            "emotional_movement": [{"message_id": "msg-2", "level": -2}],
            "skill_coverage": [
                {"category": "Listening Engagement", "percentage": 60},
                {"category": "Emotional Attunement", "percentage": 90},
                {"category": "Supportive engagement", "percentage": 40},
            ],
        }

        result = await summary_service.generate_scenario_evaluation(
            sample_chat_messages
        )

        assert set(result.keys()) == {
            "improvements",
            "positives",
            "message_tags",
            "emotional_movement",
            "skill_coverage",
        }
        assert isinstance(result["improvements"], list)
        assert isinstance(result["positives"], list)

        mock_text_generation_service.generate_scenario_evaluation.assert_called_once()
        call_args = mock_text_generation_service.generate_scenario_evaluation.call_args
        assert call_args[0][0] == sample_chat_messages
        assert call_args[1]["need_memory"] is False
        assert call_args[1]["prompts"] is None

    @pytest.mark.asyncio
    async def test_generate_scenario_evaluation_with_memory_success(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test scenario evaluation with memory generation."""
        mock_text_generation_service.generate_scenario_evaluation.return_value = {
            "improvements": ["Test improvement"],
            "positives": ["Test positive"],
            "message_tags": [],
            "emotional_movement": [],
            "skill_coverage": [
                {"category": "Listening Engagement", "percentage": 70},
                {"category": "Emotional Attunement", "percentage": 85},
                {"category": "Supportive engagement", "percentage": 55},
            ],
            "session_glimpse": "Short glimpse text",
            "cumulative_memory": "Longer cumulative text",
        }

        result = await summary_service.generate_scenario_evaluation(
            sample_chat_messages,
            need_memory=True,
            previous_memory="Previous context",
        )

        assert set(result.keys()) == {
            "improvements",
            "positives",
            "message_tags",
            "emotional_movement",
            "skill_coverage",
            "session_glimpse",
            "cumulative_memory",
        }

        mock_text_generation_service.generate_scenario_evaluation.assert_called_once()
        call_args = mock_text_generation_service.generate_scenario_evaluation.call_args
        assert call_args[1]["need_memory"] is True
        assert call_args[1]["previous_memory"] == "Previous context"
        assert call_args[1]["prompts"] is None

    @pytest.mark.asyncio
    async def test_generate_scenario_evaluation_with_empty_chat_history(
        self, summary_service, mock_text_generation_service
    ):
        """Test scenario evaluation with empty chat history."""
        mock_text_generation_service.generate_scenario_evaluation.return_value = {
            "improvements": ["Test"],
            "positives": ["Test"],
            "message_tags": [],
            "emotional_movement": [],
            "skill_coverage": [
                {"category": "Listening Engagement", "percentage": 0},
                {"category": "Emotional Attunement", "percentage": 0},
                {"category": "Supportive engagement", "percentage": 0},
            ],
        }

        result = await summary_service.generate_scenario_evaluation([])

        assert result["message_tags"] == []

    @pytest.mark.asyncio
    async def test_generate_scenario_evaluation_failed(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test scenario evaluation when LLM invocation fails."""
        mock_text_generation_service.generate_scenario_evaluation.side_effect = (
            LLMInvocationFailedException("LLM error")
        )

        with pytest.raises(CounselorTrainingAnalysisFailedException):
            await summary_service.generate_scenario_evaluation(sample_chat_messages)
