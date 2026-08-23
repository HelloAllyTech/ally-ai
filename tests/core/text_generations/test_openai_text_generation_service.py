"""Tests for OpenAITextGenerationService."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from httpx import Request, Response

from app.core.text_generations.openai_text_generation_service import (
    OpenAITextGenerationService,
    _build_scenario_behaviours_section,
    _build_supervisor_note_section,
    _format_live_notes,
    _language_directive,
    _wants_translated_feedback,
    split_text_by_length,
)
from app.core.text_generations.structured_output_models import (
    AreasOfGrowth,
    CounselorMessageAnalysis,
    EmotionalMovementItemOutput,
    MessageTagItemOutput,
    MessageTagLabelEnum,
    MessageTagOutput,
    ScenarioEvaluation,
    ScenarioEvaluationWithMemory,
    SkillCategoryEnum,
    SkillCoverageItemOutput,
    StructuredDiarization,
    StructuredIdentifyUsers,
    StructuredSummaryNote,
    StructuredTag,
    SupervisorMemoryUpdate,
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
        # Called once with the prompt; an optional usage-callback `config`
        # kwarg may also be present, so assert on the positional arg only.
        mock_client.ainvoke.assert_called_once()
        assert mock_client.ainvoke.call_args.args[0] == "Test prompt"

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
    async def test_invoke_llm_reports_usage_against_override_model(
        self, text_generation_service
    ):
        """A per-prompt llm_override's usage must be attributed to that
        override model, not to the service's default model."""
        default_model = MagicMock()
        default_model.model_name = "gpt-default"
        text_generation_service.model = default_model

        override_model = MagicMock()
        override_model.model_name = "gpt-override"
        mock_response = MagicMock()
        mock_response.content = "Test response"
        override_model.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "app.core.llm_usage.emitter.emit_llm_usage"
        ) as mock_emit:
            await text_generation_service._invoke_llm(
                "Test prompt", llm_override=override_model, task="nudge"
            )

        mock_emit.assert_called_once()
        assert mock_emit.call_args.kwargs["model"] == "gpt-override"

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
        # Keep the retry path fast in tests (no real backoff sleeps).
        text_generation_service._LLM_BACKOFF_BASE_SECONDS = 0
        text_generation_service._LLM_BACKOFF_JITTER_SECONDS = 0

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
        # Keep the retry path fast in tests (no real backoff sleeps).
        text_generation_service._LLM_BACKOFF_BASE_SECONDS = 0
        text_generation_service._LLM_BACKOFF_JITTER_SECONDS = 0

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
            session_summary=["Test summary"],
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
            ChatMessage(id="msg-1", role="client", content="I'm feeling anxious."),
            ChatMessage(id="msg-2", role="client", content="I need help."),
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
    async def test_generate_scenario_evaluation_success(
        self, text_generation_service, sample_chat_messages
    ):
        """Test successful scenario evaluation generation (need_memory=False).

        The LLM receives short IDs (m1, m2, m3) but the output should
        contain the original UUIDs (msg-1, msg-2, msg-3).
        """
        # LLM mock uses short IDs (what the LLM sees in the prompt)
        mock_evaluation = ScenarioEvaluation(
            areas_of_growth=[
                AreasOfGrowth(
                    improvement="Ask more open-ended questions",
                    recommendation=(
                        "Try using 'what' and 'how' questions instead "
                        "of closed yes/no questions"
                    ),
                )
            ],
            positives=["Good rapport building"],
            message_tags=[
                MessageTagItemOutput(
                    id="m1",
                    tags=[MessageTagOutput(label=MessageTagLabelEnum.STEADY_PACING)],
                ),
                MessageTagItemOutput(
                    id="m3",
                    tags=[MessageTagOutput(label=MessageTagLabelEnum.PARAPHRASING)],
                ),
            ],
            emotional_movement=[
                EmotionalMovementItemOutput(message_id="m2", level=-2),
            ],
            skill_coverage=[
                SkillCoverageItemOutput(
                    category=SkillCategoryEnum.LISTENING_ENGAGEMENT, percentage=60
                ),
                SkillCoverageItemOutput(
                    category=SkillCategoryEnum.EMOTIONAL_ATTUNEMENT, percentage=90
                ),
                SkillCoverageItemOutput(
                    category=SkillCategoryEnum.SUPPORTIVE_ENGAGEMENT, percentage=40
                ),
            ],
            supervisor_note=(
                "You held steady pacing when you asked about the evenings "
                "[[msg:m1]], which gave the client room to open up. Try "
                "paraphrasing back what you heard before moving to your next "
                "question. Reply any time you want to talk through a moment."
            ),
            memory_update=SupervisorMemoryUpdate(
                focus_areas=["Reflective listening"],
                trajectory="Steadily building comfort with open-ended questions.",
                next_time="Paraphrase the client's last statement before asking.",
            ),
        )

        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_evaluation
        ):
            result = await text_generation_service.generate_scenario_evaluation(
                sample_chat_messages
            )

            # Check both new and deprecated fields
            assert len(result["areas_of_growth"]) == 1
            assert (
                result["areas_of_growth"][0]["improvement"]
                == "Ask more open-ended questions"
            )
            assert result["areas_of_growth"][0]["recommendation"] == (
                "Try using 'what' and 'how' questions instead "
                "of closed yes/no questions"
            )
            # Backward compatibility - improvements should contain just
            # the improvement strings
            assert result["improvements"] == ["Ask more open-ended questions"]
            assert result["positives"] == ["Good rapport building"]
            # IDs should be remapped back to original UUIDs
            assert len(result["message_tags"]) == 2
            assert result["message_tags"][0]["id"] == "msg-1"
            assert result["message_tags"][1]["id"] == "msg-3"
            assert len(result["emotional_movement"]) == 1
            assert result["emotional_movement"][0]["message_id"] == "msg-2"
            assert result["emotional_movement"][0]["level"] == -2
            assert len(result["skill_coverage"]) == 3
            assert result["skill_coverage"][0] == {
                "category": "Listening Engagement",
                "percentage": 60,
            }
            assert result["skill_coverage"][1] == {
                "category": "Emotional Attunement",
                "percentage": 90,
            }
            assert result["skill_coverage"][2] == {
                "category": "Supportive Engagement",
                "percentage": 40,
            }
            assert "session_glimpse" not in result
            assert "cumulative_memory" not in result

    @pytest.mark.asyncio
    async def test_generate_scenario_evaluation_with_memory_success(
        self, text_generation_service, sample_chat_messages
    ):
        """Test scenario evaluation with need_memory=True returns all fields in a
        single LLM call.
        """
        # LLM mock uses short IDs
        mock_response = ScenarioEvaluationWithMemory(
            areas_of_growth=[
                AreasOfGrowth(
                    improvement="Improve reflective listening",
                    recommendation=(
                        "Practice paraphrasing client statements before "
                        "asking new questions"
                    ),
                )
            ],
            positives=["Strong empathy demonstration"],
            message_tags=[
                MessageTagItemOutput(
                    id="m1",
                    tags=[MessageTagOutput(label=MessageTagLabelEnum.USE_OF_SILENCE)],
                ),
            ],
            emotional_movement=[
                EmotionalMovementItemOutput(message_id="m2", level=-1),
            ],
            skill_coverage=[
                SkillCoverageItemOutput(
                    category=SkillCategoryEnum.LISTENING_ENGAGEMENT, percentage=70
                ),
                SkillCoverageItemOutput(
                    category=SkillCategoryEnum.EMOTIONAL_ATTUNEMENT, percentage=85
                ),
                SkillCoverageItemOutput(
                    category=SkillCategoryEnum.SUPPORTIVE_ENGAGEMENT, percentage=55
                ),
            ],
            supervisor_note=(
                "Naming the anxiety out loud [[msg:m1]] was well judged - it gave "
                "the client permission to slow down. Next time, sit with the "
                "silence a beat longer before offering a reflection. Reply "
                "whenever you'd like to unpack any part of this session."
            ),
            memory_update=SupervisorMemoryUpdate(
                focus_areas=["Use of silence"],
                trajectory="Building on last session's work on pacing.",
                next_time="Let a silence sit for a count of three before speaking.",
            ),
            session_glimpse="Brief session overview",
            cumulative_memory="Comprehensive memory narrative",
        )

        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_response
        ):
            result = await text_generation_service.generate_scenario_evaluation(
                sample_chat_messages,
                need_memory=True,
                previous_memory="Previous context",
                memory_prompt="Custom instructions",
            )

            # Check both new and deprecated fields
            assert len(result["areas_of_growth"]) == 1
            assert (
                result["areas_of_growth"][0]["improvement"]
                == "Improve reflective listening"
            )
            assert (
                result["areas_of_growth"][0]["recommendation"]
                == "Practice paraphrasing client statements before asking new questions"
            )
            assert result["improvements"] == ["Improve reflective listening"]
            assert result["positives"] == ["Strong empathy demonstration"]
            assert len(result["message_tags"]) == 1
            assert result["message_tags"][0]["id"] == "msg-1"
            assert len(result["emotional_movement"]) == 1
            assert result["emotional_movement"][0]["message_id"] == "msg-2"
            assert len(result["skill_coverage"]) == 3
            assert result["skill_coverage"][0]["category"] == "Listening Engagement"
            assert result["skill_coverage"][1]["category"] == "Emotional Attunement"
            assert result["skill_coverage"][2]["category"] == "Supportive Engagement"
            assert result["session_glimpse"] == "Brief session overview"
            assert result["cumulative_memory"] == "Comprehensive memory narrative"

    @pytest.mark.asyncio
    async def test_areas_of_growth_to_api_response_conversion(
        self, text_generation_service, sample_chat_messages
    ):
        """Test business logic for converting AreasOfGrowth objects to
        API response format with backward compatibility for improvements
        field.
        """
        # Mock LLM response with multiple AreasOfGrowth objects
        mock_evaluation = ScenarioEvaluation(
            areas_of_growth=[
                AreasOfGrowth(
                    improvement="Ask more open-ended questions",
                    recommendation=(
                        "Use 'what' and 'how' questions instead of yes/no questions"
                    ),
                ),
                AreasOfGrowth(
                    improvement="Practice reflective listening",
                    recommendation="Paraphrase client statements before responding",
                ),
                AreasOfGrowth(
                    improvement="Improve emotional validation",
                    recommendation="Acknowledge feelings before offering solutions",
                ),
            ],
            positives=["Good rapport building"],
            message_tags=[],
            emotional_movement=[],
            skill_coverage=[],
            supervisor_note=(
                "Good rapport building came through clearly in this session. "
                "Working on open-ended questions and reflective listening will "
                "help you go deeper next time. Reply if you want to talk "
                "through any of this."
            ),
            memory_update=SupervisorMemoryUpdate(
                focus_areas=["Open-ended questions", "Reflective listening"],
                trajectory="Early sessions show solid rapport-building instincts.",
                next_time="Ask one 'what' or 'how' question before summarising.",
            ),
        )

        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_evaluation
        ):
            result = await text_generation_service.generate_scenario_evaluation(
                sample_chat_messages
            )

            # Test business logic: areas_of_growth should be converted
            # to dict format
            assert len(result["areas_of_growth"]) == 3
            assert result["areas_of_growth"][0] == {
                "improvement": "Ask more open-ended questions",
                "recommendation": (
                    "Use 'what' and 'how' questions instead of yes/no questions"
                ),
            }
            assert result["areas_of_growth"][1] == {
                "improvement": "Practice reflective listening",
                "recommendation": "Paraphrase client statements before responding",
            }
            assert result["areas_of_growth"][2] == {
                "improvement": "Improve emotional validation",
                "recommendation": "Acknowledge feelings before offering solutions",
            }

            # Test backward compatibility: improvements should contain
            # only improvement strings
            assert len(result["improvements"]) == 3
            assert result["improvements"] == [
                "Ask more open-ended questions",
                "Practice reflective listening",
                "Improve emotional validation",
            ]

            # Verify order is maintained between areas_of_growth and improvements
            for i, area in enumerate(result["areas_of_growth"]):
                assert area["improvement"] == result["improvements"][i]

    @pytest.mark.asyncio
    async def test_generate_scenario_evaluation_filters_hallucinated_data(
        self, text_generation_service, sample_chat_messages
    ):
        """Test that hallucinated short IDs are filtered and valid ones remapped
        to UUIDs.
        """
        # LLM uses short IDs; m99 is hallucinated (doesn't exist)
        mock_evaluation = ScenarioEvaluation(
            areas_of_growth=[
                AreasOfGrowth(
                    improvement="Improve X", recommendation="Try doing X better"
                )
            ],
            positives=["Good Y"],
            message_tags=[
                # m1 = counselor → should be kept and remapped to msg-1
                MessageTagItemOutput(
                    id="m1",
                    tags=[MessageTagOutput(label=MessageTagLabelEnum.STEADY_PACING)],
                ),
                # m2 = client → should be filtered out (tags are for counselor only)
                MessageTagItemOutput(
                    id="m2",
                    tags=[MessageTagOutput(label=MessageTagLabelEnum.PARAPHRASING)],
                ),
                # m3 = counselor → should be kept and remapped to msg-3
                MessageTagItemOutput(
                    id="m3",
                    tags=[
                        MessageTagOutput(label=MessageTagLabelEnum.REINFORCE_AUTONOMY)
                    ],
                ),
                # m99 = hallucinated → should be filtered out
                MessageTagItemOutput(
                    id="m99",
                    tags=[MessageTagOutput(label=MessageTagLabelEnum.STEADY_PACING)],
                ),
            ],
            emotional_movement=[
                # m1 = counselor → should be filtered out
                EmotionalMovementItemOutput(message_id="m1", level=0),
                # m2 = client → should be kept and remapped to msg-2
                EmotionalMovementItemOutput(message_id="m2", level=-3),
                # m3 = counselor → should be filtered out
                EmotionalMovementItemOutput(message_id="m3", level=2),
                # m99 = hallucinated → should be filtered out
                EmotionalMovementItemOutput(message_id="m99", level=5),
            ],
            skill_coverage=[
                SkillCoverageItemOutput(
                    category=SkillCategoryEnum.LISTENING_ENGAGEMENT, percentage=50
                ),
                SkillCoverageItemOutput(
                    category=SkillCategoryEnum.EMOTIONAL_ATTUNEMENT, percentage=75
                ),
                SkillCoverageItemOutput(
                    category=SkillCategoryEnum.SUPPORTIVE_ENGAGEMENT, percentage=30
                ),
            ],
            # Mirrors the mix above: m1 is a real transcript message, m99 is
            # hallucinated and must be dropped rather than remapped.
            supervisor_note=(
                "The steady pacing when you checked in [[msg:m1]] helped keep "
                "things grounded. There's also a moment worth revisiting "
                "[[msg:m99]] where staying with the silence a little longer "
                "might have opened things up further. Reply if you'd like to "
                "talk through it."
            ),
            memory_update=SupervisorMemoryUpdate(
                focus_areas=["Steady pacing"],
                trajectory="Consistent, grounded pacing across sessions so far.",
                next_time="Let a silence sit for a few extra seconds before speaking.",
            ),
        )

        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_evaluation
        ):
            result = await text_generation_service.generate_scenario_evaluation(
                sample_chat_messages
            )

            # Only counselor messages should remain, remapped to original UUIDs
            assert len(result["message_tags"]) == 2
            tag_ids = [t["id"] for t in result["message_tags"]]
            assert "msg-1" in tag_ids
            assert "msg-3" in tag_ids
            assert "msg-2" not in tag_ids
            assert "m99" not in tag_ids

            # Only client message should remain, remapped to original UUID
            assert len(result["emotional_movement"]) == 1
            assert result["emotional_movement"][0]["message_id"] == "msg-2"
            assert result["emotional_movement"][0]["level"] == -3

    @pytest.mark.asyncio
    async def test_generate_scenario_evaluation_failed(
        self, text_generation_service, sample_chat_messages
    ):
        """Test scenario evaluation generation failure."""
        with patch.object(
            text_generation_service,
            "_invoke_llm",
            side_effect=LLMInvocationFailedException("LLM error"),
        ):
            with pytest.raises(LLMInvocationFailedException):
                await text_generation_service.generate_scenario_evaluation(
                    sample_chat_messages
                )


def _make_scenario_evaluation(supervisor_note=None, memory_update=None):
    """Build a minimal valid ScenarioEvaluation for prompt-inspection tests.

    Accepts overrides for supervisor_note/memory_update so callers testing
    those fields specifically (e.g. anchor remapping) don't need to
    duplicate the whole builder.
    """
    return ScenarioEvaluation(
        areas_of_growth=[
            AreasOfGrowth(
                improvement="Ask more open-ended questions",
                recommendation="Use 'what' and 'how' questions",
            )
        ],
        positives=["Good rapport building"],
        message_tags=[],
        emotional_movement=[],
        skill_coverage=[
            SkillCoverageItemOutput(
                category=SkillCategoryEnum.LISTENING_ENGAGEMENT, percentage=60
            ),
        ],
        supervisor_note=supervisor_note
        or (
            "Good rapport building came through in this session. Try asking "
            "more open-ended questions next time. Reply if you'd like to talk "
            "through any of it."
        ),
        memory_update=memory_update
        or SupervisorMemoryUpdate(
            focus_areas=["Open-ended questions"],
            trajectory="Off to a steady start with rapport-building.",
            next_time="Ask one open-ended question before summarising.",
        ),
    )


class TestWantsTranslatedFeedback:
    """Unit tests for the _wants_translated_feedback helper."""

    @pytest.mark.parametrize(
        "language_code,expected",
        [
            (None, False),
            ("", False),
            ("en", False),
            ("EN", False),
            ("en-US", False),
            ("en_US", False),
            ("hi", True),
            ("HI", True),
            ("hi-IN", True),
            ("hi_IN", True),
            ("mr", True),
            ("ta", True),
            ("kn", True),
        ],
    )
    def test_wants_translated_feedback(self, language_code, expected):
        assert _wants_translated_feedback(language_code) is expected


class TestLanguageDirective:
    """Unit tests for the _language_directive helper."""

    def test_directive_includes_code_and_known_name(self):
        directive = _language_directive("hi")
        assert "OUTPUT LANGUAGE" in directive
        assert "'hi'" in directive
        assert "Hindi" in directive

    def test_directive_handles_bcp47_and_known_name(self):
        directive = _language_directive("mr-IN")
        assert "'mr-IN'" in directive
        assert "Marathi" in directive

    def test_directive_falls_back_to_raw_code_for_unknown(self):
        directive = _language_directive("xx")
        assert "'xx'" in directive

    def test_directive_names_supervisor_note(self):
        """The output-language directive must also cover supervisor_note, so
        the debrief note is translated along with the rest of the feedback.
        """
        directive = _language_directive("hi")
        assert "supervisor_note" in directive


class TestBuildSupervisorNoteSection:
    """Unit tests for the _build_supervisor_note_section helper."""

    def test_lay_worker_type_pulls_lay_register(self):
        section = _build_supervisor_note_section(worker_type="LAY")
        assert "not clinically trained" in section

    def test_early_professional_worker_type_pulls_its_register(self):
        section = _build_supervisor_note_section(worker_type="EARLY_PROFESSIONAL")
        assert "clinically trained but early in practice" in section

    def test_experienced_professional_worker_type_pulls_its_register(self):
        section = _build_supervisor_note_section(worker_type="EXPERIENCED_PROFESSIONAL")
        assert "seasoned practitioner" in section

    def test_unknown_worker_type_falls_back_to_lay_register(self):
        section = _build_supervisor_note_section(worker_type="SOME_MADE_UP_TYPE")
        assert "not clinically trained" in section

    def test_none_worker_type_falls_back_to_lay_register(self):
        section = _build_supervisor_note_section(worker_type=None)
        assert "not clinically trained" in section

    def test_worker_type_is_case_insensitive(self):
        section = _build_supervisor_note_section(worker_type="lay")
        assert "not clinically trained" in section

    def test_missing_learner_name_uses_there_placeholder(self):
        section = _build_supervisor_note_section(learner_name=None)
        assert "is: there" in section

    def test_blank_learner_name_uses_there_placeholder(self):
        section = _build_supervisor_note_section(learner_name="   ")
        assert "is: there" in section

    def test_learner_name_is_interpolated(self):
        section = _build_supervisor_note_section(learner_name="Priya")
        assert "is: Priya" in section

    def test_missing_supervisor_memory_uses_default_text(self):
        section = _build_supervisor_note_section(supervisor_memory=None)
        assert "No previous sessions with this learner yet." in section

    def test_supervisor_memory_is_interpolated(self):
        section = _build_supervisor_note_section(
            supervisor_memory="Worked on pacing last time."
        )
        assert "Worked on pacing last time." in section

    def test_missing_live_notes_states_none_were_given(self):
        # Must read as a fact, not a blank: otherwise the note starts
        # referring to advice it never gave.
        section = _build_supervisor_note_section(live_notes=None)
        assert "You gave no live notes during this session." in section

    def test_live_notes_are_rendered_as_a_list(self):
        section = _build_supervisor_note_section(
            live_notes=["Slow down here.", "She just named a fear."]
        )
        assert "- Slow down here." in section
        assert "- She just named a fear." in section

    def test_blank_live_notes_fall_back_to_the_none_given_text(self):
        section = _build_supervisor_note_section(live_notes=["  ", ""])
        assert "You gave no live notes during this session." in section


class TestFormatLiveNotes:
    """Unit tests for the _format_live_notes helper."""

    def test_empty_and_none_are_equivalent(self):
        assert _format_live_notes(None) == _format_live_notes([])

    def test_notes_keep_their_order(self):
        assert _format_live_notes(["first", "second"]) == "- first\n- second"

    def test_whitespace_is_trimmed(self):
        assert _format_live_notes(["  padded  "]) == "- padded"


class TestBuildScenarioBehavioursSection:
    """Unit tests for the _build_scenario_behaviours_section helper."""

    def test_no_behaviours_returns_empty_section(self):
        section = _build_scenario_behaviours_section()
        assert section == ""

    def test_empty_lists_return_empty_section(self):
        section = _build_scenario_behaviours_section(
            helpful_behaviours=[], unhelpful_behaviours=[]
        )
        assert section == ""

    def test_helpful_behaviours_are_bulleted(self):
        section = _build_scenario_behaviours_section(
            helpful_behaviours=["Reflects feelings before offering advice"]
        )
        assert "- Reflects feelings before offering advice" in section
        assert "Helpful behaviours for this scenario:" in section

    def test_unhelpful_behaviours_are_bulleted(self):
        section = _build_scenario_behaviours_section(
            unhelpful_behaviours=["Interrupts before the client finishes a thought"]
        )
        assert "- Interrupts before the client finishes a thought" in section
        assert "Unhelpful behaviours for this scenario:" in section

    def test_only_one_list_present_omits_the_other_heading(self):
        section = _build_scenario_behaviours_section(
            helpful_behaviours=["Uses silence well"]
        )
        assert "Helpful behaviours for this scenario:" in section
        assert "Unhelpful behaviours for this scenario:" not in section

    def test_never_moves_skill_coverage(self):
        section = _build_scenario_behaviours_section(
            helpful_behaviours=["Uses silence well"]
        )
        assert "skill_coverage" in section


class TestScenarioEvaluationLanguageDirective:
    """Tests that language_code threads into the prompt sent to the LLM."""

    @pytest.fixture
    def text_generation_service(self):
        with patch(
            "app.core.text_generations.openai_text_generation_service.settings"
        ) as mock_settings:
            mock_settings.LLM.MAX_CONCURRENT_LLM_CALLS = 10
            return OpenAITextGenerationService(MagicMock(), AsyncMock())

    @pytest.fixture
    def sample_chat_messages(self):
        return [
            ChatMessage(
                id="msg-1", role="counselor", content="How are you feeling today?"
            ),
            ChatMessage(
                id="msg-2", role="client", content="I'm feeling anxious about work."
            ),
        ]

    async def _capture_prompt(
        self, service, sample_chat_messages, **kwargs
    ) -> str:
        """Run generate_scenario_evaluation and return the prompt sent to the LLM."""
        with patch.object(
            service, "_invoke_llm", return_value=_make_scenario_evaluation()
        ) as mock_invoke:
            await service.generate_scenario_evaluation(
                sample_chat_messages, **kwargs
            )
            assert mock_invoke.await_count == 1
            return mock_invoke.await_args.args[0]

    @pytest.mark.asyncio
    async def test_non_english_appends_directive(
        self, text_generation_service, sample_chat_messages
    ):
        prompt = await self._capture_prompt(
            text_generation_service, sample_chat_messages, language_code="hi"
        )
        assert "OUTPUT LANGUAGE" in prompt
        assert "'hi'" in prompt
        assert "Hindi" in prompt

    @pytest.mark.asyncio
    async def test_non_english_directive_matches_baseline_plus_directive(
        self, text_generation_service, sample_chat_messages
    ):
        baseline = await self._capture_prompt(
            text_generation_service, sample_chat_messages, language_code=None
        )
        translated = await self._capture_prompt(
            text_generation_service, sample_chat_messages, language_code="hi"
        )
        assert translated == baseline + _language_directive("hi")

    @pytest.mark.asyncio
    async def test_none_leaves_prompt_unchanged(
        self, text_generation_service, sample_chat_messages
    ):
        prompt = await self._capture_prompt(
            text_generation_service, sample_chat_messages, language_code=None
        )
        assert "OUTPUT LANGUAGE" not in prompt

    @pytest.mark.asyncio
    async def test_default_leaves_prompt_unchanged(
        self, text_generation_service, sample_chat_messages
    ):
        prompt = await self._capture_prompt(
            text_generation_service, sample_chat_messages
        )
        assert "OUTPUT LANGUAGE" not in prompt

    @pytest.mark.asyncio
    @pytest.mark.parametrize("language_code", ["en", "en-US", "en_US"])
    async def test_english_leaves_prompt_unchanged(
        self, text_generation_service, sample_chat_messages, language_code
    ):
        baseline = await self._capture_prompt(
            text_generation_service, sample_chat_messages, language_code=None
        )
        prompt = await self._capture_prompt(
            text_generation_service,
            sample_chat_messages,
            language_code=language_code,
        )
        assert "OUTPUT LANGUAGE" not in prompt
        assert prompt == baseline


class TestGenerateScenarioEvaluationSupervisorNote:
    """Tests that worker_type/learner_name/supervisor_memory thread into the
    prompt sent to the LLM, and that the returned supervisor_note/memory_update
    are correctly post-processed.
    """

    @pytest.fixture
    def text_generation_service(self):
        with patch(
            "app.core.text_generations.openai_text_generation_service.settings"
        ) as mock_settings:
            mock_settings.LLM.MAX_CONCURRENT_LLM_CALLS = 10
            return OpenAITextGenerationService(MagicMock(), AsyncMock())

    @pytest.fixture
    def sample_chat_messages(self):
        return [
            ChatMessage(
                id="msg-1", role="counselor", content="How are you feeling today?"
            ),
            ChatMessage(
                id="msg-2", role="client", content="I'm feeling anxious about work."
            ),
        ]

    @pytest.mark.asyncio
    async def test_worker_context_threads_into_prompt(
        self, text_generation_service, sample_chat_messages
    ):
        with patch.object(
            text_generation_service,
            "_invoke_llm",
            return_value=_make_scenario_evaluation(),
        ) as mock_invoke:
            await text_generation_service.generate_scenario_evaluation(
                sample_chat_messages,
                worker_type="EXPERIENCED_PROFESSIONAL",
                learner_name="Priya",
                supervisor_memory="Worked on pacing last time.",
            )
            prompt = mock_invoke.await_args.args[0]

        assert "seasoned practitioner" in prompt
        assert "is: Priya" in prompt
        assert "Worked on pacing last time." in prompt

    @pytest.mark.asyncio
    async def test_missing_worker_context_falls_back_to_defaults_in_prompt(
        self, text_generation_service, sample_chat_messages
    ):
        with patch.object(
            text_generation_service,
            "_invoke_llm",
            return_value=_make_scenario_evaluation(),
        ) as mock_invoke:
            await text_generation_service.generate_scenario_evaluation(
                sample_chat_messages
            )
            prompt = mock_invoke.await_args.args[0]

        assert "not clinically trained" in prompt
        assert "is: there" in prompt
        assert "No previous sessions with this learner yet." in prompt

    @pytest.mark.asyncio
    async def test_supervisor_note_anchors_remapped_to_uuids_in_result(
        self, text_generation_service, sample_chat_messages
    ):
        mock_evaluation = _make_scenario_evaluation(
            supervisor_note=(
                "You held steady pacing when you checked in [[msg:m1]]. "
                "Reply any time you want to talk through this session."
            )
        )
        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_evaluation
        ):
            result = await text_generation_service.generate_scenario_evaluation(
                sample_chat_messages
            )

        assert "[[msg:msg-1]]" in result["supervisor_note"]
        assert "[[msg:m1]]" not in result["supervisor_note"]

    @pytest.mark.asyncio
    async def test_memory_update_is_plain_dict_with_three_keys(
        self, text_generation_service, sample_chat_messages
    ):
        mock_evaluation = _make_scenario_evaluation(
            memory_update=SupervisorMemoryUpdate(
                focus_areas=["Use of silence", "Open-ended questions"],
                trajectory="Building confidence across sessions.",
                next_time="Let a silence sit for three seconds before speaking.",
            )
        )
        with patch.object(
            text_generation_service, "_invoke_llm", return_value=mock_evaluation
        ):
            result = await text_generation_service.generate_scenario_evaluation(
                sample_chat_messages
            )

        assert result["memory_update"] == {
            "focus_areas": ["Use of silence", "Open-ended questions"],
            "trajectory": "Building confidence across sessions.",
            "next_time": "Let a silence sit for three seconds before speaking.",
        }


class TestCondenseForSummary:
    """Tests for the summary-input length guard (_condense_for_summary)."""

    @pytest.fixture
    def text_generation_service(self):
        with patch(
            "app.core.text_generations.openai_text_generation_service.settings"
        ) as mock_settings:
            mock_settings.LLM.MAX_CONCURRENT_LLM_CALLS = 10
            return OpenAITextGenerationService(MagicMock(), AsyncMock())

    @pytest.mark.asyncio
    async def test_short_transcript_passes_through_untouched(
        self, text_generation_service
    ):
        text_generation_service._invoke_llm = AsyncMock(return_value="CONDENSED")
        transcript = "CLIENT: hello\nCOUNSELOR: hi there"

        result = await text_generation_service._condense_for_summary(transcript)

        assert result == transcript
        text_generation_service._invoke_llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_long_transcript_is_condensed(self, text_generation_service):
        text_generation_service._invoke_llm = AsyncMock(return_value="CONDENSED")
        # > MAX_SUMMARY_INPUT_WORDS (8000); many lines so it splits into
        # multiple chunks, each condensed via an LLM call.
        transcript = "\n".join("CLIENT: word word word word word" for _ in range(2000))

        result = await text_generation_service._condense_for_summary(transcript)

        assert "CONDENSED" in result
        assert len(result.split()) < len(transcript.split())
        assert text_generation_service._invoke_llm.await_count >= 2

    @pytest.mark.asyncio
    async def test_condense_chunk_failure_falls_back_to_raw(
        self, text_generation_service
    ):
        # If a chunk's condensation fails, it must fall back to (truncated) raw
        # text rather than failing the whole summary.
        text_generation_service._invoke_llm = AsyncMock(
            side_effect=Exception("llm down")
        )
        transcript = "\n".join("CLIENT: alpha beta gamma delta" for _ in range(2000))

        result = await text_generation_service._condense_for_summary(transcript)

        assert isinstance(result, str)
        assert "alpha" in result  # raw content preserved on fallback
        text_generation_service._invoke_llm.assert_awaited()
