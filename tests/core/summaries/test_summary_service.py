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
    CompetencyItem,
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
        expected_response = {
            "improvements": ["Ask more open-ended questions"],
            "positives": ["Good rapport building"],
        }
        mock_text_generation_service.generate_simulation_summary.return_value = (
            expected_response
        )

        # Execute
        result = await summary_service.generate_simulation_summary(
            sample_chat_messages
        )

        # Assert
        assert result == expected_response
        mock_text_generation_service.generate_simulation_summary.assert_called_once_with(
            sample_chat_messages,
            need_memory=False,
            previous_memory=None,
            memory_prompt=None,
            chat_id=None,
        )

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_failed(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test simulation summary generation failure."""
        # Setup mocks
        mock_text_generation_service.generate_simulation_summary.side_effect = (
            LLMInvocationFailedException("LLM error")
        )

        # Execute and assert
        with pytest.raises(CounselorTrainingAnalysisFailedException) as exc_info:
            await summary_service.generate_simulation_summary(sample_chat_messages)

        assert "Failed to generate counselor training analysis" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_empty_chat_history(
        self, summary_service, mock_text_generation_service
    ):
        """Test simulation summary with empty chat history."""
        # Setup mocks
        expected_response = {"improvements": [], "positives": []}
        mock_text_generation_service.generate_simulation_summary.return_value = (
            expected_response
        )

        # Execute
        result = await summary_service.generate_simulation_summary([])

        # Assert
        assert result == expected_response
        mock_text_generation_service.generate_simulation_summary.assert_called_once_with(
            [],
            need_memory=False,
            previous_memory=None,
            memory_prompt=None,
            chat_id=None,
        )

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_need_memory_false_no_memory_fields(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """When need_memory=False, output has no session_glimpse/cumulative_memory."""
        base_response = {
            "improvements": ["Improve reflection"],
            "positives": ["Good listening"],
        }
        mock_text_generation_service.generate_simulation_summary.return_value = (
            base_response
        )

        result = await summary_service.generate_simulation_summary(
            sample_chat_messages,
            need_memory=False,
        )

        assert result == base_response
        assert "session_glimpse" not in result
        assert "cumulative_memory" not in result
        mock_text_generation_service.generate_simulation_summary.assert_called_once_with(
            sample_chat_messages,
            need_memory=False,
            previous_memory=None,
            memory_prompt=None,
            chat_id=None,
        )

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_need_memory_true_returns_all_fields(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """When need_memory=True, output includes session_glimpse and cumulative_memory."""
        full_response = {
            "improvements": ["Ask more open-ended questions"],
            "positives": ["Good rapport"],
            "session_glimpse": "Client discussed work anxiety; counselor reflected.",
            "cumulative_memory": "Session 1: Focus on work-related anxiety.",
        }
        mock_text_generation_service.generate_simulation_summary.return_value = (
            full_response
        )

        result = await summary_service.generate_simulation_summary(
            sample_chat_messages,
            need_memory=True,
        )

        assert result["improvements"] == full_response["improvements"]
        assert result["positives"] == full_response["positives"]
        assert result["session_glimpse"] == full_response["session_glimpse"]
        assert result["cumulative_memory"] == full_response["cumulative_memory"]
        mock_text_generation_service.generate_simulation_summary.assert_called_once_with(
            sample_chat_messages,
            need_memory=True,
            previous_memory=None,
            memory_prompt=None,
            chat_id=None,
        )

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_need_memory_true_with_previous_memory(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """When need_memory=True and previous_memory is set, it is passed through."""
        previous_memory = "Previous session: client was anxious about deadlines."
        full_response = {
            "improvements": [],
            "positives": ["Empathy"],
            "session_glimpse": "Brief glimpse.",
            "cumulative_memory": "Updated cumulative narrative.",
        }
        mock_text_generation_service.generate_simulation_summary.return_value = (
            full_response
        )

        result = await summary_service.generate_simulation_summary(
            sample_chat_messages,
            need_memory=True,
            previous_memory=previous_memory,
        )

        assert result["session_glimpse"] == full_response["session_glimpse"]
        assert result["cumulative_memory"] == full_response["cumulative_memory"]
        mock_text_generation_service.generate_simulation_summary.assert_called_once_with(
            sample_chat_messages,
            need_memory=True,
            previous_memory=previous_memory,
            memory_prompt=None,
            chat_id=None,
        )

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_need_memory_true_with_memory_prompt(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """When need_memory=True and memory_prompt is set, it is passed through."""
        memory_prompt = "Focus on therapeutic alliance and goals."
        full_response = {
            "improvements": [],
            "positives": [],
            "session_glimpse": "Glimpse with custom focus.",
            "cumulative_memory": "Memory with custom focus.",
        }
        mock_text_generation_service.generate_simulation_summary.return_value = (
            full_response
        )

        result = await summary_service.generate_simulation_summary(
            sample_chat_messages,
            need_memory=True,
            memory_prompt=memory_prompt,
        )

        assert result["session_glimpse"] == full_response["session_glimpse"]
        assert result["cumulative_memory"] == full_response["cumulative_memory"]
        mock_text_generation_service.generate_simulation_summary.assert_called_once_with(
            sample_chat_messages,
            need_memory=True,
            previous_memory=None,
            memory_prompt=memory_prompt,
            chat_id=None,
        )

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_need_memory_true_with_all_params(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """When need_memory=True with both previous_memory and memory_prompt, all are passed."""
        previous_memory = "Last time: focus on anxiety."
        memory_prompt = "Emphasize progress over time."
        full_response = {
            "improvements": ["More reflections"],
            "positives": ["Warmth", "Validation"],
            "session_glimpse": "Session focused on progress.",
            "cumulative_memory": "Cumulative narrative with progress emphasis.",
        }
        mock_text_generation_service.generate_simulation_summary.return_value = (
            full_response
        )

        result = await summary_service.generate_simulation_summary(
            sample_chat_messages,
            need_memory=True,
            previous_memory=previous_memory,
            memory_prompt=memory_prompt,
        )

        assert result["improvements"] == full_response["improvements"]
        assert result["positives"] == full_response["positives"]
        assert result["session_glimpse"] == full_response["session_glimpse"]
        assert result["cumulative_memory"] == full_response["cumulative_memory"]
        mock_text_generation_service.generate_simulation_summary.assert_called_once_with(
            sample_chat_messages,
            need_memory=True,
            previous_memory=previous_memory,
            memory_prompt=memory_prompt,
            chat_id=None,
        )

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_output_structure_with_memory(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Output with need_memory=True has correct keys and types."""
        mock_text_generation_service.generate_simulation_summary.return_value = {
            "improvements": ["A", "B"],
            "positives": ["X"],
            "session_glimpse": "Short glimpse text",
            "cumulative_memory": "Longer cumulative text",
        }

        result = await summary_service.generate_simulation_summary(
            sample_chat_messages, need_memory=True
        )

        assert set(result.keys()) == {
            "improvements",
            "positives",
            "session_glimpse",
            "cumulative_memory",
        }
        assert isinstance(result["improvements"], list)
        assert isinstance(result["positives"], list)
        assert isinstance(result["session_glimpse"], str)
        assert isinstance(result["cumulative_memory"], str)

    @pytest.mark.asyncio
    async def test_generate_scenario_evaluation_success(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test successful scenario evaluation generation."""
        competencies = [
            {"id": "comp-1", "competency": "Socialising the Client to Counselling"},
            {"id": "comp-2", "competency": "Explanation and Promotion of Ethics"},
        ]

        mock_text_generation_service.generate_scenario_evaluation.return_value = {
            "improvements": ["Test improvement"],
            "positives": ["Test positive"],
            "achieved_competency_ids": ["comp-1"],
        }

        result = await summary_service.generate_scenario_evaluation(
            sample_chat_messages, competencies
        )

        assert set(result.keys()) == {"improvements", "positives", "achieved_competency_ids"}
        assert isinstance(result["improvements"], list)
        assert isinstance(result["positives"], list)
        assert isinstance(result["achieved_competency_ids"], list)
        assert result["achieved_competency_ids"] == ["comp-1"]

        mock_text_generation_service.generate_scenario_evaluation.assert_called_once()
        call_args = mock_text_generation_service.generate_scenario_evaluation.call_args
        assert call_args[0][0] == sample_chat_messages
        assert call_args[1]["competencies"] == competencies
        assert call_args[1]["need_memory"] is False

    @pytest.mark.asyncio
    async def test_generate_scenario_evaluation_with_memory_success(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test scenario evaluation with memory generation."""
        competencies = [
            {"id": "comp-1", "competency": "Test Competency"},
        ]

        mock_text_generation_service.generate_scenario_evaluation.return_value = {
            "improvements": ["Test improvement"],
            "positives": ["Test positive"],
            "achieved_competency_ids": ["comp-1"],
            "session_glimpse": "Short glimpse text",
            "cumulative_memory": "Longer cumulative text",
        }

        result = await summary_service.generate_scenario_evaluation(
            sample_chat_messages,
            competencies,
            need_memory=True,
            previous_memory="Previous context",
        )

        assert set(result.keys()) == {
            "improvements",
            "positives",
            "achieved_competency_ids",
            "session_glimpse",
            "cumulative_memory",
        }
        assert result["achieved_competency_ids"] == ["comp-1"]

        mock_text_generation_service.generate_scenario_evaluation.assert_called_once()
        call_args = mock_text_generation_service.generate_scenario_evaluation.call_args
        assert call_args[1]["need_memory"] is True
        assert call_args[1]["previous_memory"] == "Previous context"

    @pytest.mark.asyncio
    async def test_generate_scenario_evaluation_with_empty_competencies(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test scenario evaluation with empty competencies list."""
        competencies = []

        mock_text_generation_service.generate_scenario_evaluation.return_value = {
            "improvements": ["Test"],
            "positives": ["Test"],
            "achieved_competency_ids": [],
        }

        result = await summary_service.generate_scenario_evaluation(
            sample_chat_messages, competencies
        )

        assert result["achieved_competency_ids"] == []

    @pytest.mark.asyncio
    async def test_generate_scenario_evaluation_failed(
        self, summary_service, mock_text_generation_service, sample_chat_messages
    ):
        """Test scenario evaluation when LLM invocation fails."""
        competencies = [{"id": "comp-1", "competency": "Test"}]

        mock_text_generation_service.generate_scenario_evaluation.side_effect = (
            LLMInvocationFailedException("LLM error")
        )

        with pytest.raises(CounselorTrainingAnalysisFailedException):
            await summary_service.generate_scenario_evaluation(
                sample_chat_messages, competencies
            )
