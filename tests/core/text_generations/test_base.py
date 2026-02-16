"""Tests for BaseTextGenerationService."""

from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

import pytest

from app.core.text_generations.base import BaseTextGenerationService
from app.schemas.common import ChatMessage
from app.schemas.conversation import IdentifyResponse
from app.schemas.summary import DynamicSummaryNoteResponse, SummaryNoteAndTagsResponse


class ConcreteTextGenerationService(BaseTextGenerationService[MagicMock]):
    """Concrete implementation of BaseTextGenerationService for testing."""

    async def generate_nudge(
        self, conversation: str, chat_history: str, suggestion: str, **kwargs
    ) -> str:
        """Concrete implementation of generate_nudge."""
        return "test nudge"

    async def generate_summary_notes(
        self, chat_history: List[ChatMessage], keys: Optional[List[str]] = None
    ) -> Union[SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse]:
        """Concrete implementation of generate_summary_notes."""
        if keys:
            return DynamicSummaryNoteResponse(fields={"test": "value"})
        return SummaryNoteAndTagsResponse(
            session_summary="test summary", tags=[], call_quality=3
        )

    async def enhance_content(self, content: str, **kwargs) -> str:
        """Concrete implementation of enhance_content."""
        return "enhanced content"

    async def identify_user(self, chat_history: List[ChatMessage]) -> IdentifyResponse:
        """Concrete implementation of identify_user."""
        return IdentifyResponse(speaker0="client", speaker1="counselor")

    async def get_tag_positivity_ratings(self, tags: List[str]) -> List[Dict]:
        """Concrete implementation of get_tag_positivity_ratings."""
        return [{"tag": tag, "rating": 3} for tag in tags]

    async def analyze_counselor_messages(
        self, chat_history: List[ChatMessage]
    ) -> Dict[str, int]:
        """Concrete implementation of analyze_counselor_messages."""
        return {"reflective": 1, "open_ended": 2, "back_channel": 3}

    async def generate_simulation_summary(
        self,
        chat_history: List[ChatMessage],
        need_memory: bool = False,
        previous_memory: Optional[str] = None,
        memory_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Concrete implementation of generate_simulation_summary."""
        result: Dict[str, Any] = {
            "improvements": ["improvement1"],
            "positives": ["positive1"],
        }
        if need_memory:
            result["session_glimpse"] = "Test glimpse"
            result["cumulative_memory"] = "Test memory"
        return result

    async def generate_scenario_evaluation(
        self,
        chat_history: List[ChatMessage],
        need_memory: bool = False,
        previous_memory: Optional[str] = None,
        memory_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Concrete implementation of generate_scenario_evaluation."""
        result: Dict[str, Any] = {
            "improvements": ["improvement1"],
            "positives": ["positive1"],
            "skill_coverage": [
                {"category": "Learning", "percentage": 50},
                {"category": "Support", "percentage": 50},
                {"category": "Standards", "percentage": 50},
            ],
        }
        if need_memory:
            result["session_glimpse"] = "Test glimpse"
            result["cumulative_memory"] = "Test memory"
        return result


class TestBaseTextGenerationService:
    """Test cases for BaseTextGenerationService."""

    @pytest.fixture
    def mock_model(self):
        """Mock model for testing."""
        return MagicMock()

    @pytest.fixture
    def text_generation_service(self, mock_model):
        """Create BaseTextGenerationService instance for testing."""
        return ConcreteTextGenerationService(mock_model)

    @pytest.fixture
    def sample_chat_messages(self):
        """Sample chat messages for testing."""
        return [
            ChatMessage(
                id="msg-1", role="client", content="Hello", start_time=None, end_time=None
            ),
            ChatMessage(
                id="msg-2",
                role="counselor",
                content="Hi there",
                start_time=None,
                end_time=None,
            ),
        ]

    def test_init(self, mock_model):
        """Test BaseTextGenerationService initialization."""
        service = ConcreteTextGenerationService(mock_model)
        assert service.model == mock_model

    @pytest.mark.asyncio
    async def test_generate_nudge(self, text_generation_service):
        """Test generate_nudge method."""
        conversation = "test conversation"
        chat_history = "test history"
        suggestion = "test suggestion"

        result = await text_generation_service.generate_nudge(
            conversation, chat_history, suggestion
        )

        assert result == "test nudge"

    @pytest.mark.asyncio
    async def test_generate_nudge_with_kwargs(self, text_generation_service):
        """Test generate_nudge method with additional kwargs."""
        conversation = "test conversation"
        chat_history = "test history"
        suggestion = "test suggestion"
        extra_param = "extra"

        result = await text_generation_service.generate_nudge(
            conversation, chat_history, suggestion, extra_param=extra_param
        )

        assert result == "test nudge"

    @pytest.mark.asyncio
    async def test_generate_summary_notes_without_keys(
        self, text_generation_service, sample_chat_messages
    ):
        """Test generate_summary_notes method without keys."""
        result = await text_generation_service.generate_summary_notes(
            sample_chat_messages
        )

        assert isinstance(result, SummaryNoteAndTagsResponse)
        assert result.session_summary == "test summary"
        assert result.tags == []
        assert result.call_quality == 3

    @pytest.mark.asyncio
    async def test_generate_summary_notes_with_keys(
        self, text_generation_service, sample_chat_messages
    ):
        """Test generate_summary_notes method with keys."""
        keys = ["test_key"]
        result = await text_generation_service.generate_summary_notes(
            sample_chat_messages, keys
        )

        assert isinstance(result, DynamicSummaryNoteResponse)
        assert result.fields == {"test": "value"}

    @pytest.mark.asyncio
    async def test_enhance_content(self, text_generation_service):
        """Test enhance_content method."""
        content = "original content"
        result = await text_generation_service.enhance_content(content)

        assert result == "enhanced content"

    @pytest.mark.asyncio
    async def test_enhance_content_with_kwargs(self, text_generation_service):
        """Test enhance_content method with additional kwargs."""
        content = "original content"
        extra_param = "extra"

        result = await text_generation_service.enhance_content(
            content, extra_param=extra_param
        )

        assert result == "enhanced content"

    @pytest.mark.asyncio
    async def test_identify_user(self, text_generation_service, sample_chat_messages):
        """Test identify_user method."""
        result = await text_generation_service.identify_user(sample_chat_messages)

        assert isinstance(result, IdentifyResponse)
        assert result.speaker0 == "client"
        assert result.speaker1 == "counselor"

    @pytest.mark.asyncio
    async def test_get_tag_positivity_ratings(self, text_generation_service):
        """Test get_tag_positivity_ratings method."""
        tags = ["anxiety", "depression"]
        result = await text_generation_service.get_tag_positivity_ratings(tags)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["tag"] == "anxiety"
        assert result[0]["rating"] == 3
        assert result[1]["tag"] == "depression"
        assert result[1]["rating"] == 3

    @pytest.mark.asyncio
    async def test_analyze_counselor_messages(
        self, text_generation_service, sample_chat_messages
    ):
        """Test analyze_counselor_messages method."""
        result = await text_generation_service.analyze_counselor_messages(
            sample_chat_messages
        )

        assert isinstance(result, dict)
        assert result["reflective"] == 1
        assert result["open_ended"] == 2
        assert result["back_channel"] == 3

    @pytest.mark.asyncio
    async def test_generate_simulation_summary(
        self, text_generation_service, sample_chat_messages
    ):
        """Test generate_simulation_summary method."""
        result = await text_generation_service.generate_simulation_summary(
            sample_chat_messages
        )

        assert isinstance(result, dict)
        assert "improvements" in result
        assert "positives" in result
        assert result["improvements"] == ["improvement1"]
        assert result["positives"] == ["positive1"]

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_with_kwargs(
        self, text_generation_service, sample_chat_messages
    ):
        """Test generate_simulation_summary method with additional kwargs."""
        result = await text_generation_service.generate_simulation_summary(
            sample_chat_messages, extra_param="extra"
        )

        assert isinstance(result, dict)
        assert "improvements" in result
        assert "positives" in result

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError when not implemented."""

        class IncompleteTextGenerationService(BaseTextGenerationService[MagicMock]):
            """Incomplete implementation missing some abstract methods."""

            async def generate_nudge(
                self, conversation: str, chat_history: str, suggestion: str, **kwargs
            ) -> str:
                return "test"

            async def generate_summary_notes(
                self, chat_history: List[ChatMessage], keys: Optional[List[str]] = None
            ):
                return SummaryNoteAndTagsResponse(
                    session_summary="test", tags=[], call_quality=3
                )

            # Missing other abstract methods

        # This should work since we're not instantiating the incomplete class
        # The abstract methods will be enforced when trying to instantiate
        with pytest.raises(TypeError):
            IncompleteTextGenerationService(MagicMock())
