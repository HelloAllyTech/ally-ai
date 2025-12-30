"""Tests for OpenAITextGenerationService."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from httpx import Request, Response

from app.core.text_generations.openai_text_generation_service import (
    OpenAITextGenerationService,
    split_text_by_length,
)
from app.core.text_generations.structured_output_models import (
    CounselorMessageAnalysis,
    SimulationAnalysis,
    StructuredDiarization,
    StructuredIdentifyUsers,
    StructuredSummaryNote,
    StructuredTag,
)
from app.exceptions.custom_exceptions import (
    ContentEnhancementFailedException,
    IdentifyUserFailedException,
    LLMInvocationFailedException,
    NudgeGenerationFailedException,
    SummaryNoteFailedException,
)
from app.schemas.common import ChatMessage
from app.schemas.conversation import IdentifyResponse, Nudge
from app.schemas.summary import ContentEnhance, DynamicSummaryNoteResponse, Tag


class TestSplitTextByLength:
    """Test cases for split_text_by_length function."""

    def test_split_short_text(self):
        """Test splitting short text that doesn't need chunking."""
        text = "[00:00:01] Speaker 0: Hello\n[00:00:03] Speaker 1: Hi there"
        result = split_text_by_length(text, max_words=100)

        assert len(result) == 1
        assert result[0] == text

    def test_split_long_text(self):
        """Test splitting long text into multiple chunks."""
        # Create a long text with many lines
        lines = []
        for i in range(100):
            lines.append(f"[00:00:{i:02d}] Speaker {i % 2}: This is message number {i}")
        text = "\n".join(lines)

        result = split_text_by_length(text, max_words=50)

        assert len(result) > 1
        # Verify all chunks are within word limit (with some tolerance for overlap)
        for chunk in result:
            word_count = len(chunk.split())
            assert word_count <= 80  # Allow some tolerance for overlap

    def test_split_with_overlap(self):
        """Test that chunks have overlap between them."""
        lines = []
        for i in range(20):
            lines.append(f"[00:00:{i:02d}] Speaker {i % 2}: Message {i}")
        text = "\n".join(lines)

        result = split_text_by_length(text, max_words=30)

        if len(result) > 1:
            # Check that there's some overlap between consecutive chunks
            first_chunk_lines = result[0].split("\n")
            second_chunk_lines = result[1].split("\n")

            # Should have some common lines due to overlap
            assert len(set(first_chunk_lines) & set(second_chunk_lines)) > 0

    def test_empty_text(self):
        """Test splitting empty text."""
        result = split_text_by_length("", max_words=10)
        assert result == []

    def test_single_line_text(self):
        """Test splitting single line text."""
        text = "[00:00:01] Speaker 0: This is a single line message"
        result = split_text_by_length(text, max_words=10)
        assert len(result) == 1
        assert result[0] == text


class TestOpenAITextGenerationService:
    """Test cases for OpenAITextGenerationService."""

    @pytest.fixture
    def mock_client(self):
        """Mock OpenAI client."""
        return MagicMock()

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        return AsyncMock()

    @pytest.fixture
    def text_generation_service(self, mock_client, mock_embedding_service):
        """Create OpenAITextGenerationService instance with mocked dependencies."""
        with patch(
            "app.core.text_generations.openai_text_generation_service.settings"
        ) as mock_settings:
            mock_settings.LLM.MAX_CONCURRENT_LLM_CALLS = 10
            return OpenAITextGenerationService(mock_client, mock_embedding_service)

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
    async def test_invoke_llm_success(self, text_generation_service, mock_client):
        """Test successful LLM invocation."""
        # Setup mocks
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_client.ainvoke = AsyncMock(return_value=mock_response)
        text_generation_service.model = mock_client

        # Execute
        result = await text_generation_service._invoke_llm("Test prompt")

        # Assert
        assert result == "Test response"
        mock_client.ainvoke.assert_called_once_with("Test prompt")

    @pytest.mark.asyncio
    async def test_invoke_llm_with_structured_output(
        self, text_generation_service, mock_client
    ):
        """Test LLM invocation with structured output."""
        # Setup mocks
        mock_structured_response = MagicMock()
        mock_client.with_structured_output.return_value = mock_client
        mock_client.ainvoke = AsyncMock(return_value=mock_structured_response)
        text_generation_service.model = mock_client

        # Execute
        result = await text_generation_service._invoke_llm(
            "Test prompt", output_class=StructuredSummaryNote
        )

        # Assert
        assert result == mock_structured_response
        mock_client.with_structured_output.assert_called_once_with(
            StructuredSummaryNote
        )

    @pytest.mark.asyncio
    async def test_invoke_llm_rate_limit_error(
        self, text_generation_service, mock_client
    ):
        """Test LLM invocation with rate limit error."""
        # Setup mocks
        req = Request("POST", "https://api.openai.com/v1/chat/completions")
        body = {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}
        resp = Response(429, request=req, json=body)

        mock_client.ainvoke = AsyncMock(
            side_effect=openai.RateLimitError(
                message=body["error"]["message"],
                response=resp,
                body=body,
            )
        )
        text_generation_service.model = mock_client

        # Execute and assert
        with pytest.raises(LLMInvocationFailedException) as exc_info:
            await text_generation_service._invoke_llm("Test prompt")

        assert "rate limit exceeded" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invoke_llm_connection_error(
        self, text_generation_service, mock_client
    ):
        """Test LLM invocation with connection error."""
        # Setup mocks
        req = Request("POST", "https://api.openai.com/v1/chat/completions")
        mock_client.ainvoke = AsyncMock(
            side_effect=openai.APIConnectionError(
                request=req,
                message="Connection error",
            )
        )
        text_generation_service.model = mock_client

        # Execute and assert
        with pytest.raises(LLMInvocationFailedException) as exc_info:
            await text_generation_service._invoke_llm("Test prompt")

        # Service wraps APIConnectionError with a generic message
        assert "openai api error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_nudge_success(self, text_generation_service):
        """Test successful nudge generation."""
        # Setup mocks
        mock_nudge = Nudge(nudge="Test nudge")
        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_nudge
        ):
            # Execute
            result = await text_generation_service.generate_nudge(
                "conversation", "chat_history", "suggestion"
            )

            # Assert
            assert result == "Test nudge"

    @pytest.mark.asyncio
    async def test_generate_nudge_failed(self, text_generation_service):
        """Test nudge generation failure."""
        # Setup mocks
        with patch.object(
            text_generation_service,
            "_invoke_llm",
            side_effect=LLMInvocationFailedException("LLM error"),
        ):
            # Execute and assert
            with pytest.raises(NudgeGenerationFailedException):
                await text_generation_service.generate_nudge(
                    "conversation", "chat_history", "suggestion"
                )

    def test_get_key_descriptions(self, text_generation_service):
        """_get_key_descriptions should include descriptions for StructuredSummaryNote fields."""  # noqa: E501
        desc = text_generation_service._get_key_descriptions(
            ["tags", "call_quality", "unknown_field"]
        )
        # Should mention known fields and ignore unknown ones
        assert "- tags:" in desc
        assert "- call_quality:" in desc

    def test_extract_tool_fields(self, text_generation_service):
        """_extract_tool_fields should parse tool_calls and return fields dict."""

        class R:
            pass

        r = R()
        r.additional_kwargs = {
            "tool_calls": [
                {
                    "function": {
                        "name": "generate_dynamic_summary",
                        "arguments": json.dumps({"fields": {"session_summary": "S"}}),
                    }
                }
            ]
        }
        fields = text_generation_service._extract_tool_fields(r)
        assert fields == {"session_summary": "S"}

    def test_chat_history_to_str(self, text_generation_service, sample_chat_messages):
        """_chat_history_to_str should format role-prefixed lines with newlines."""
        s = text_generation_service._chat_history_to_str(sample_chat_messages)
        assert "counselor:" in s and "client:" in s and "\n" in s

    @pytest.mark.asyncio
    async def test_generate_dynamic_summary_without_key_descriptions_returns_precomputed_only(  # noqa: E501
        self, text_generation_service, sample_chat_messages
    ):
        """When no key descriptions, dynamic summary should return only precomputed metrics."""  # noqa: E501
        with (
            patch.object(
                text_generation_service,
                "_calculate_metrics",
                return_value={"affirmations": 3},
            ),
            patch.object(
                text_generation_service, "_get_key_descriptions", return_value=""
            ),
        ):
            res = await text_generation_service.generate_summary_notes(
                sample_chat_messages, keys=["nonexistent_key"]
            )
        assert isinstance(res, DynamicSummaryNoteResponse)
        assert res.fields == {"affirmations": 3}

    @pytest.mark.asyncio
    async def test_generate_dynamic_summary_merges_tool_fields_and_precomputed(
        self, text_generation_service, sample_chat_messages
    ):
        """Dynamic summary merges tool output fields with precomputed metrics."""
        with (
            patch.object(
                text_generation_service,
                "_calculate_metrics",
                return_value={"affirmations": 2},
            ),
            patch.object(
                text_generation_service,
                "_get_key_descriptions",
                return_value="- session_summary: desc\n- tags: desc",
            ),
        ):
            # Mock model.bind_tools().ainvoke() to return a response with tool_calls
            tool_response = MagicMock()
            tool_response.additional_kwargs = {
                "tool_calls": [
                    {
                        "function": {
                            "name": "generate_dynamic_summary",
                            "arguments": json.dumps(
                                {
                                    "fields": {
                                        "session_summary": "X",
                                        "tags": [{"tag": "t", "positivity_rating": 1}],
                                    }
                                }
                            ),
                        }
                    }
                ]
            }

            mock_model = MagicMock()
            mock_model.ainvoke = AsyncMock(return_value=tool_response)
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_model

            # Inject mocked llm model
            text_generation_service.model = mock_llm

            res = await text_generation_service.generate_summary_notes(
                sample_chat_messages, keys=["session_summary", "tags"]
            )

            assert isinstance(res, DynamicSummaryNoteResponse)
            assert res.fields["session_summary"] == "X"
            assert res.fields["affirmations"] == 2

    @pytest.mark.asyncio
    async def test_generate_structured_summary_applies_metrics(
        self, text_generation_service, sample_chat_messages
    ):
        """Structured summary should have metrics merged onto the response object."""
        # Create a mock response object that allows setting attributes
        mock_response = MagicMock()
        mock_response.affirmations = 0

        # Mock the converter to return our mock response
        with (
            patch(
                "app.core.text_generations.openai_text_generation_service.structured_output_model_to_rest",  # noqa: E501
                return_value=mock_response,
            ),
            patch.object(
                text_generation_service,
                "_invoke_llm",
                return_value=StructuredSummaryNote(
                    tags=[StructuredTag(tag="a", positivity_rating=2)], call_quality=80
                ),
            ),
            patch.object(
                text_generation_service,
                "_calculate_metrics",
                return_value={"avg_client_utterance_duration": 1.2},
            ),
        ):
            res = await text_generation_service.generate_summary_notes(
                sample_chat_messages
            )

        # Verify metrics were applied to the response
        assert res == mock_response
        assert hasattr(res, "avg_client_utterance_duration")
        assert res.avg_client_utterance_duration == 1.2

    @pytest.mark.asyncio
    async def test_calculate_metrics_with_keys_subset_and_counselor_analysis_once(  # noqa: E501
        self, text_generation_service, sample_chat_messages, monkeypatch
    ):
        """_calculate_metrics should compute only requested keys and call counselor analysis once."""  # noqa: E501
        # Patch simple metric function to avoid heavy work
        monkeypatch.setattr(
            "app.core.text_generations.openai_text_generation_service.count_affirmations",  # noqa: E501
            lambda chat_history: 5,
        )

        # Spy on analyze_counselor_messages
        with patch.object(
            text_generation_service,
            "analyze_counselor_messages",
            return_value={
                "reflective_questions_asked": 1,
                "open_ended_questions_asked": 2,
                "back_channel_cues": 3,
            },
        ) as spy:
            keys = ["affirmations", "reflective_questions_asked", "back_channel_cues"]
            out = await text_generation_service._calculate_metrics(
                sample_chat_messages, "x", keys
            )

        # Only requested keys present
        assert set(out.keys()) == {
            "affirmations",
            "reflective_questions_asked",
            "back_channel_cues",
        }
        assert out["affirmations"] == 5
        assert out["reflective_questions_asked"] == 1
        assert out["back_channel_cues"] == 3
        spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_summary_notes_structured_success(
        self, text_generation_service, sample_chat_messages
    ):
        """Test successful structured summary generation."""
        # Setup mocks

        mock_structured_response = StructuredSummaryNote(
            session_summary="Test summary",
            tags=[StructuredTag(tag="anxiety", positivity_rating=2)],
            call_quality=85,
        )
        with patch.object(
            text_generation_service,
            "_generate_structured_summary",
            return_value=mock_structured_response,
        ) as mock_generate:
            # Execute
            result = await text_generation_service.generate_summary_notes(
                sample_chat_messages
            )

            # Assert
            assert result == mock_structured_response
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_summary_notes_dynamic_success(
        self, text_generation_service, sample_chat_messages
    ):
        """Test successful dynamic summary generation."""
        # Setup mocks
        keys = ["session_summary", "tags"]
        mock_dynamic_response = DynamicSummaryNoteResponse(
            fields={
                "session_summary": "Test summary",
                "tags": [{"tag": "anxiety", "positivity_rating": 2}],
            }
        )
        with patch.object(
            text_generation_service,
            "_generate_dynamic_summary",
            return_value=mock_dynamic_response,
        ) as mock_generate:
            # Execute
            result = await text_generation_service.generate_summary_notes(
                sample_chat_messages, keys
            )

            # Assert
            assert result == mock_dynamic_response
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_summary_notes_failed(
        self, text_generation_service, sample_chat_messages
    ):
        """Test summary generation failure."""
        # Setup mocks
        with patch.object(
            text_generation_service,
            "_generate_structured_summary",
            side_effect=Exception("Generation error"),
        ):
            # Execute and assert
            with pytest.raises(SummaryNoteFailedException):
                await text_generation_service.generate_summary_notes(
                    sample_chat_messages
                )

    @pytest.mark.asyncio
    async def test_enhance_content_success(self, text_generation_service):
        """Test successful content enhancement."""
        # Setup mocks
        mock_enhanced = ContentEnhance(enhanced_content="Enhanced content")
        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_enhanced
        ):
            # Execute
            result = await text_generation_service.enhance_content("Original content")

            # Assert
            assert result == "Enhanced content"

    @pytest.mark.asyncio
    async def test_enhance_content_failed(self, text_generation_service):
        """Test content enhancement failure."""
        # Setup mocks
        with patch.object(
            text_generation_service,
            "_invoke_llm",
            side_effect=LLMInvocationFailedException("LLM error"),
        ):
            # Execute and assert
            with pytest.raises(ContentEnhancementFailedException):
                await text_generation_service.enhance_content("Original content")

    @pytest.mark.asyncio
    async def test_identify_user_success(
        self, text_generation_service, sample_chat_messages
    ):
        """Test successful user identification."""
        # Setup mocks
        mock_identify_response = StructuredIdentifyUsers(
            speaker0="client", speaker1="counselor"
        )
        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_identify_response
        ):
            # Execute
            result = await text_generation_service.identify_user(sample_chat_messages)

            # Assert
            expected = IdentifyResponse(speaker0="client", speaker1="counselor")
            assert result == expected

    @pytest.mark.asyncio
    async def test_identify_user_failed(
        self, text_generation_service, sample_chat_messages
    ):
        """Test user identification failure."""
        # Setup mocks
        with patch.object(
            text_generation_service,
            "_invoke_llm",
            side_effect=LLMInvocationFailedException("LLM error"),
        ):
            # Execute and assert
            with pytest.raises(IdentifyUserFailedException):
                await text_generation_service.identify_user(sample_chat_messages)

    @pytest.mark.asyncio
    async def test_get_tag_positivity_ratings_success(self, text_generation_service):
        """Test successful tag positivity ratings retrieval."""
        # Setup mocks
        tags = ["anxiety", "depression"]
        mock_tag_list = MagicMock()
        mock_tag_list.tags = [
            Tag(tag="anxiety", positivity_rating=2),
            Tag(tag="depression", positivity_rating=1),
        ]
        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_tag_list
        ):
            # Execute
            result = await text_generation_service.get_tag_positivity_ratings(tags)

            # Assert
            expected = [
                {"tag": "anxiety", "positivity_rating": 2},
                {"tag": "depression", "positivity_rating": 1},
            ]
            assert result == expected

    @pytest.mark.asyncio
    async def test_get_tag_positivity_ratings_failed(self, text_generation_service):
        """Test tag positivity ratings failure."""
        # Setup mocks
        tags = ["anxiety"]
        with patch.object(
            text_generation_service,
            "_invoke_llm",
            side_effect=LLMInvocationFailedException("LLM error"),
        ):
            # Execute and assert
            with pytest.raises(Exception):
                await text_generation_service.get_tag_positivity_ratings(tags)

    @pytest.mark.asyncio
    async def test_diarize_from_transcription_single_chunk(
        self, text_generation_service
    ):
        """Test diarization with single chunk."""
        # Setup mocks
        transcription = "[00:00:01] Speaker 0: Hello\n[00:00:03] Speaker 1: Hi there"
        mock_diarization = StructuredDiarization(messages=[])
        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_diarization
        ):
            # Execute
            result = await text_generation_service.diarize_from_transcription(
                transcription
            )

            # Assert
            assert result == mock_diarization

    @pytest.mark.asyncio
    async def test_diarize_from_transcription_multiple_chunks(
        self, text_generation_service
    ):
        """Test diarization with multiple chunks."""
        # Create a very long transcription that will definitely be split into multiple
        # chunks
        lines = []
        for i in range(1000):  # Much larger to ensure multiple chunks
            lines.append(
                f"[00:00:{i:02d}] Speaker {i % 2}: This is a very long message "
                f"number {i} with lots of words to ensure it exceeds the word limit"
            )
        transcription = "\n".join(lines)

        # Setup mocks
        mock_diarization = StructuredDiarization(messages=[])
        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_diarization
        ):
            # Execute
            result = await text_generation_service.diarize_from_transcription(
                transcription
            )

            # Assert
            assert result == mock_diarization
            # Should be called multiple times for multiple chunks

    @pytest.mark.asyncio
    async def test_diarize_single_chunk_failure_raises(self, text_generation_service):
        """Single-chunk diarization should wrap errors in LLMInvocationFailedException."""  # noqa: E501
        transcription = "[00:00:01] Speaker 0: Hello"  # single small chunk
        with patch.object(
            text_generation_service, "_invoke_llm", side_effect=Exception("boom")
        ):
            with pytest.raises(LLMInvocationFailedException):
                await text_generation_service.diarize_from_transcription(transcription)

    @pytest.mark.asyncio
    async def test_diarize_multi_chunk_partial_failure_raises(
        self, text_generation_service
    ):
        """If any chunk fails, diarization should raise LLMInvocationFailedException."""
        # Create 2 small chunks by limiting max_words during split via patch
        text = "\n".join(
            [
                "[00:00:01] Speaker 0: one two three four five",
                "[00:00:02] Speaker 1: six seven eight nine ten",
                "[00:00:03] Speaker 0: eleven twelve thirteen fourteen fifteen",
                "[00:00:04] Speaker 1: sixteen seventeen eighteen nineteen twenty",
            ]
        )

        # Force split into two chunks
        with patch(
            "app.core.text_generations.openai_text_generation_service.MAX_WORDS_PER_CHUNK",  # noqa: E501
            10,
        ):
            # First chunk succeeds, second fails
            ok = StructuredDiarization(messages=[])
            with patch.object(
                text_generation_service,
                "_invoke_llm",
                side_effect=[ok, Exception("fail")],
            ):
                with pytest.raises(LLMInvocationFailedException):
                    await text_generation_service.diarize_from_transcription(text)

    @pytest.mark.asyncio
    async def test_analyze_counselor_messages_success(
        self, text_generation_service, sample_chat_messages
    ):
        """Test successful counselor message analysis."""
        # Setup mocks
        mock_analysis = CounselorMessageAnalysis(
            reflective=["How does that make you feel?"],
            open_ended=["Can you tell me more?"],
            back_channel=["I see", "Uh-huh"],
        )
        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_analysis
        ):
            # Execute
            result = await text_generation_service.analyze_counselor_messages(
                sample_chat_messages
            )

            # Assert
            expected = {
                "reflective_questions_asked": 2,
                "open_ended_questions_asked": 2,
                "back_channel_cues": 4,
            }
            assert result == expected

    @pytest.mark.asyncio
    async def test_analyze_counselor_messages_no_counselor_messages(
        self, text_generation_service
    ):
        """Test counselor message analysis with no counselor messages."""
        # Setup - only client messages
        client_messages = [
            ChatMessage(role="client", content="I'm feeling anxious."),
            ChatMessage(role="client", content="I need help."),
        ]

        # Execute
        result = await text_generation_service.analyze_counselor_messages(
            client_messages
        )

        # Assert
        expected = {
            "reflective_questions_asked": 0,
            "open_ended_questions_asked": 0,
            "back_channel_cues": 0,
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_analyze_counselor_messages_handles_per_message_error(
        self, text_generation_service, sample_chat_messages
    ):
        """Per-message analysis errors should be handled and counted as zeros."""
        # Only counselor messages are analyzed; make sure at least one exists
        with patch.object(
            text_generation_service,
            "_invoke_llm",
            side_effect=Exception("analysis error"),
        ):
            result = await text_generation_service.analyze_counselor_messages(
                sample_chat_messages
            )
        assert result == {
            "reflective_questions_asked": 0,
            "open_ended_questions_asked": 0,
            "back_channel_cues": 0,
        }

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_success(
        self, text_generation_service, sample_chat_messages
    ):
        """Test successful simulation summary generation."""
        # Setup mocks
        goal = "Improve counseling skills"
        mock_simulation = SimulationAnalysis(
            improvements=["Ask more open-ended questions"],
            positives=["Good rapport building"],
        )
        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_simulation
        ):
            # Execute
            result = await text_generation_service.generate_simulation_summary(
                sample_chat_messages, goal
            )

            # Assert
            expected = {
                "improvements": ["Ask more open-ended questions"],
                "positives": ["Good rapport building"],
            }
            assert result == expected

    @pytest.mark.asyncio
    async def test_generate_simulation_summary_failed(
        self, text_generation_service, sample_chat_messages
    ):
        """Test simulation summary generation failure."""
        # Setup mocks
        goal = "Improve counseling skills"
        with patch.object(
            text_generation_service,
            "_invoke_llm",
            side_effect=LLMInvocationFailedException("LLM error"),
        ):
            # Execute and assert
            with pytest.raises(LLMInvocationFailedException):
                await text_generation_service.generate_simulation_summary(
                    sample_chat_messages, goal
                )
