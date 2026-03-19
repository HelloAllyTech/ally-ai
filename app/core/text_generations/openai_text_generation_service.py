import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Type, Union, cast

import openai
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import create_model

from app.core.config import settings
from app.core.embeddings.base import BaseEmbeddingService
from app.core.phi_events import PHIEvents
from app.core.phi_logger import PHILogEvent, phi_logger
from app.core.text_generations.base import BaseTextGenerationService
from app.core.text_generations.structured_output_models import (
    CounselorMessageAnalysis,
    ScenarioEvaluation,
    ScenarioEvaluationWithMemory,
    StructuredDiarization,
    StructuredIdentifyUsers,
    StructuredSummaryNote,
)
from app.exceptions.custom_exceptions import (
    ContentEnhancementFailedException,
    IdentifyUserFailedException,
    LLMInvocationFailedException,
    NudgeGenerationFailedException,
    SummaryNoteFailedException,
)
from app.prompts.resolver import load_and_format, load_template
from app.schemas.common import ChatMessage
from app.schemas.conversation import IdentifyResponse, Nudge
from app.schemas.summary import (
    ContentEnhance,
    DynamicSummaryNoteResponse,
    SummaryNoteAndTagsResponse,
    Tag,
)
from app.utils.affirmation_counter import count_affirmations
from app.utils.client_positivity_lift_calculator import calculate_client_positivity_lift
from app.utils.common import (
    build_id_mapping,
    filter_emotional_movement,
    filter_message_tags,
)
from app.utils.counselor_interruption_calculator import (
    calculate_counselor_interruptions,
)
from app.utils.language_detector import detect_languages
from app.utils.logger import get_logger
from app.utils.rate_limiter import rate_limiter
from app.utils.reflective_listening_calculator import calculate_reflective_listening
from app.utils.silence_calculator import calculate_silence_by_counselor
from app.utils.structured_model_converter import structured_output_model_to_rest
from app.utils.utterance_duration_calculator import (
    calculate_avg_client_utterance_duration,
)

# Static prompt constants are bypassed for dynamic runtime loading with overrides.


logger = get_logger(__name__)

# Constants for chunking
MAX_WORDS_PER_CHUNK = (
    2000  # Conservative limit based on word count (roughly equivalent to 3000 tokens)
)
CHUNK_OVERLAP_WORDS = (
    300  # Overlap between chunks to maintain context (roughly equivalent to 500 tokens)
)


@tool
def generate_dynamic_summary(
    fields: dict[str, Union[str, int]]
) -> dict[str, Union[str, int]]:
    """Generate a dynamic summary with the given fields."""
    # Create a temporary StructuredSummaryNote instance for validation
    try:
        # Convert values to appropriate types based on StructuredSummaryNote fields
        validated_fields = {}
        for key, value in fields.items():
            field = StructuredSummaryNote.model_fields.get(key)
            if field:
                if field.annotation == int:
                    validated_fields[key] = int(value)
                elif field.annotation == List[str]:
                    validated_fields[key] = (
                        value.split(",") if isinstance(value, str) else value
                    )
                else:
                    validated_fields[key] = value
            else:
                validated_fields[key] = value

        # Validate using StructuredSummaryNote
        _ = StructuredSummaryNote(**validated_fields)
        return validated_fields
    except Exception as e:
        logger.error(f"Validation failed: {type(e).__name__}")
        return {}


def split_text_by_length(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> List[str]:
    """
    Split transcription text into chunks while preserving timing information.

    This function intelligently splits transcription text to avoid exceeding word limits
    while maintaining context and preserving timestamp information. It uses word
    counting
    for simpler and more predictable chunk sizing.

    Word Counting Strategy:
    - Uses simple word splitting for fast and predictable chunk sizing
    - Processes text line-by-line to maintain natural boundaries
    - Preserves timestamp information and speaker identification
    - Adds overlap between chunks to maintain conversation context

    Chunking Strategy:
    1. Split text into lines (preserving timestamps and speaker info)
    2. Group lines until approaching max_words limit
    3. Create overlap between chunks using timestamp-containing lines
    4. Ensure no chunk exceeds the word limit

    Args:
        text (str): Raw transcription text with timestamps and speaker information
        max_words (int): Maximum words per chunk (default: MAX_WORDS_PER_CHUNK)

    Returns:
        List[str]: List of text chunks, each within the word limit

    Example:
        Input: "[00:00:01] Speaker 0: Hello\n[00:00:03] Speaker 1: Hi there"
        Output: ["[00:00:01] Speaker 0: Hello\n[00:00:03] Speaker 1: Hi there"]

    Word Count Examples:
        - "[00:00:01] Speaker 0: Hello" → 4 words
        - "[00:00:05] Speaker 1: How are you today?" → 7 words
    """
    lines = text.splitlines()
    chunks = []
    current_chunk = []
    current_word_count = 0

    def count_words(text: str) -> int:
        """
        Count words in text using simple whitespace splitting.

        This provides fast and predictable word counts for chunk sizing.

        Args:
            text (str): Text to count words for

        Returns:
            int: Word count
        """
        return len(text.split())

    # Process each line individually to maintain natural boundaries
    for line in lines:
        line_word_count = count_words(line)

        # If adding this line would exceed the limit and we have content, start a
        # new chunk
        if current_word_count + line_word_count > max_words and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(line)
        current_word_count += line_word_count

    # Add the last chunk if it has content
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    # Add overlap between chunks to maintain conversation context
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            # Add some lines from the end of current chunk to the beginning of next
            # chunk
            current_lines = chunks[i].splitlines()
            next_lines = chunks[i + 1].splitlines()

            # Take last few lines from current chunk (if they contain timestamps)
            # This helps maintain context between chunks
            overlap_lines = []
            for line in reversed(current_lines[-3:]):  # Last 3 lines
                if any(char.isdigit() for char in line):  # Likely contains timestamp
                    overlap_lines.insert(0, line)
                if count_words("\n".join(overlap_lines)) > CHUNK_OVERLAP_WORDS:
                    break

            if overlap_lines:
                chunks[i + 1] = "\n".join(overlap_lines + next_lines)

    return chunks


class OpenAITextGenerationService(BaseTextGenerationService[ChatOpenAI]):
    """Text generation service using OpenAI models."""

    def __init__(
        self, client: ChatOpenAI, embedding_service: BaseEmbeddingService
    ) -> None:
        """
        Initialize the OpenAITextGenerator with a client and embedding service.

        Parameters:
            client (ChatOpenAI): The OpenAI chat client to use.
            embedding_service (BaseEmbeddingService): The embedding service
            for reflective listening calculation.
        """
        # Store the embedding service
        self.embedding_service = embedding_service
        # Initialize semaphore for rate limiting at class level
        self.semaphore = asyncio.Semaphore(settings.LLM.MAX_CONCURRENT_LLM_CALLS)

        # Initialize the base class with the client
        super().__init__(client)

    async def _invoke_llm(
        self,
        messages: List[BaseMessage],
        output_class: Optional[Type] = None,
        **kwargs,
    ) -> Union[Any, str]:
        """
        Invoke the LLM asynchronously with optional structured output.

        Parameters:
            messages (List[BaseMessage] | str):
                - A list of messages (`List[BaseMessage]`) for conversational input.
                - A raw string (`str`) if a single message is provided.
            output_class (Optional[type[T]]):
                - An optional Pydantic model or class to structure the output.
                - If provided, the response will be parsed into an instance of this
                  class.
            **kwargs:
                - Additional keyword arguments.

        Returns:
            T | str:
                - If `output_class` is provided, returns an instance of
                `output_class` (T).
                - Otherwise, returns the raw response content as a string.
        """

        await rate_limiter.acquire()

        async with self.semaphore:
            # Use the model from base class
            llm = self.model

            # Bind structured output class with the llm if passed by user
            if output_class:
                llm = llm.with_structured_output(output_class)

            try:
                response = await llm.ainvoke(messages)
            except openai.RateLimitError as e:
                logger.exception(f"Error: {type(e).__name__}")
                raise LLMInvocationFailedException(
                    "OpenAI API rate limit exceeded. Please try again later."
                ) from e

            except openai.APIConnectionError as e:
                logger.exception(f"Error: {type(e).__name__}")
                raise LLMInvocationFailedException(
                    "OpenAI API error. Please try again later."
                ) from e

            return response if output_class else response.content

    async def generate_nudge(
        self,
        conversation: str,
        chat_history: str,
        suggestion: str,
        prompts: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate a nudge using the OpenAI language model.

        This method formats a prompt with the provided conversation context, chat
        history, and suggestion, then invokes the language model to generate a nudge.
        The response is cast to a NudgeOutput object from which the nudge text is
        extracted and returned.

        Parameters:
            conversation (str): The current conversation context.
            chat_history (str): A string representation of the chat history to
            provide context.
            suggestion (str): A suggestion or prompt to guide the nudge generation.
            **kwargs: Additional keyword arguments to be passed to the underlying
            language model invocation.

        Returns:
            str: The generated nudge extracted from the language model's response.

        Raises:
            NudgeGenerationFailedException: If the OpenAI API rate
            limit is exceeded or an API connection error occurs.
        """
        logger.info("Generating nudge using OpenAI")
        try:
            template = load_template("nudge/nudge", prompts=prompts)
            response = cast(
                Nudge,
                await self._invoke_llm(
                    template.format(
                        conversation=conversation,
                        chat_history=chat_history,
                        suggestion=suggestion,
                    ),
                    Nudge,
                    **kwargs,
                ),
            )

        except LLMInvocationFailedException as e:
            raise NudgeGenerationFailedException("Failed to invoke LLM.") from e

        logger.info("Nudge generated successfully")

        return response.nudge

    async def generate_summary_notes(
        self,
        chat_history: List[ChatMessage],
        keys: Optional[List[str]] = None,
        chat_id: Optional[str] = None,
        prompts: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Union[SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse]:
        start_time = time.time()

        try:
            logger.info("Generating summary notes using OpenAI")
            chat_history_str = self._chat_history_to_str(chat_history)

            if keys:
                result = await self._generate_dynamic_summary(
                    chat_history, chat_history_str, keys, prompts=prompts, **kwargs
                )
            else:
                result = await self._generate_structured_summary(
                    chat_history, chat_history_str, prompts=prompts, **kwargs
                )

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Log successful completion
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=chat_id,
                    audit_id=None,  # Will be set by external service
                    details={
                        "message": "Summary generation completed using OpenAI",
                        "component": "OpenAITextGenerationService",
                        "processing_time_ms": processing_time_ms,
                        "result_type": type(result).__name__,
                    },
                )
            )

            return result

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Log error
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=chat_id,
                    audit_id=None,  # Will be set by external service
                    details={
                        "error": f"Failed to generate summary: {type(e).__name__}",
                        "component": "OpenAITextGenerationService",
                        "method": "generate_summary_notes",
                        "exception_type": type(e).__name__,
                        "chat_history_count": len(chat_history),
                        "processing_time_ms": processing_time_ms,
                        "summary_type": "dynamic" if keys else "structured",
                        "keys_provided": keys is not None,
                    },
                )
            )

            logger.exception(f"Failed to generate summary: {type(e).__name__}")
            raise SummaryNoteFailedException("Failed to generate summary") from e

    async def _generate_dynamic_summary(
        self,
        chat_history,
        chat_history_str,
        keys,
        prompts: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        precomputed = await self._calculate_metrics(
            chat_history, chat_history_str, keys, prompts=prompts
        )

        key_descriptions = self._get_key_descriptions(keys)
        if not key_descriptions:
            return DynamicSummaryNoteResponse(fields=precomputed)

        template = load_template("summary/dynamic_summary", prompts=prompts)
        prompt = template.format(
            chat_history=chat_history_str, key_descriptions=key_descriptions
        )
        model = self.model.bind_tools([generate_dynamic_summary])
        response = await model.ainvoke(prompt)

        tool_fields = self._extract_tool_fields(response)
        merged = {**tool_fields, **precomputed}
        return DynamicSummaryNoteResponse(fields=merged)

    async def _generate_structured_summary(
        self,
        chat_history,
        chat_history_str,
        prompts: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Optimized structured summary with parallel processing."""
        # Start LLM call and metric calculation in parallel
        template = load_template("summary/summary", prompts=prompts)
        llm_task = self._invoke_llm(
            template.format(chat_history=chat_history_str),
            StructuredSummaryNote,
            **kwargs,
        )

        metrics_task = self._calculate_metrics(
            chat_history, chat_history_str, prompts=prompts
        )

        # Run both in parallel
        response, metrics = await asyncio.gather(llm_task, metrics_task)

        response = structured_output_model_to_rest(response)

        # Add metrics to response
        for key, value in metrics.items():
            setattr(response, key, value)

        return response

    def _chat_history_to_str(self, chat_history: List[ChatMessage]) -> str:
        return "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])

    async def _calculate_metrics(
        self,
        chat_history,
        chat_history_str,
        keys: Optional[List[str]] = None,
        prompts: Optional[Dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Calculate metrics:
        - If keys=None → calculate all available metrics
        - If keys provided → calculate only those
        """

        # Simple metrics
        simple_dispatch = {
            "languages": (lambda: detect_languages(chat_history_str)),
            "reflective_listening": (
                lambda: calculate_reflective_listening(
                    chat_history, self.embedding_service
                )
            ),
            "avg_client_utterance_duration": (
                lambda: calculate_avg_client_utterance_duration(chat_history)
            ),
            "silence_by_counselor": (
                lambda: calculate_silence_by_counselor(chat_history)
            ),
            "client_positivity_lift": (
                lambda: calculate_client_positivity_lift(chat_history)
            ),
            "counselor_interruptions": (
                lambda: calculate_counselor_interruptions(chat_history)
            ),
            "affirmations": (lambda: count_affirmations(chat_history)),
        }

        counselor_analysis_keys = {
            "reflective_questions_asked",
            "open_ended_questions_asked",
            "back_channel_cues",
        }

        # Determine which keys to process
        keys_to_process = (
            list(simple_dispatch.keys()) + list(counselor_analysis_keys)
            if keys is None
            else keys
        )
        results = {}
        counselor_analysis = None

        for key in keys_to_process:
            if key in simple_dispatch:
                fn = simple_dispatch[key]
                value = fn()
                results[key] = await value if asyncio.iscoroutine(value) else value

            elif key in counselor_analysis_keys:
                if counselor_analysis is None:
                    counselor_analysis = await self.analyze_counselor_messages(
                        chat_history, prompts=prompts
                    )
                results[key] = counselor_analysis.get(key, 0)

        return results

    def _extract_tool_fields(self, response) -> dict[str, Any]:
        if (
            hasattr(response, "additional_kwargs")
            and "tool_calls" in response.additional_kwargs
        ):
            tool_call = response.additional_kwargs["tool_calls"][0]
            if tool_call["function"]["name"] == "generate_dynamic_summary":
                fields = json.loads(tool_call["function"]["arguments"])
                return fields.get("fields", {})
        return {}

    def _get_key_descriptions(self, keys: list[str]) -> str:
        descriptions = []
        for key in keys:
            field = StructuredSummaryNote.model_fields.get(key)
            if field and field.description:
                descriptions.append(f"- {key}: {field.description}")
        return "\n".join(descriptions)

    async def enhance_content(
        self,
        content: str,
        prompts: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> str:
        """
        Enhance the content by generating a summary and tags.

        Parameters:
            content (str): The content to enhance.
            **kwargs: Additional keyword arguments to be passed to the
            underlying language model invocation.

        Returns:
            str: The enhanced content.

        Raises:
            ContentEnhancementFailedException: If the content enhancement fails.
        """
        logger.info("Enhancing content using OpenAI")
        try:
            template = load_template("notes/content_enhance", prompts=prompts)
            response = cast(
                ContentEnhance,
                await self._invoke_llm(
                    template.format(content=content),
                    ContentEnhance,
                    **kwargs,
                ),
            )

        except LLMInvocationFailedException as e:
            raise ContentEnhancementFailedException("Failed to invoke LLM.") from e

        logger.info("Content enhanced successfully")
        return response.enhanced_content

    async def identify_user(
        self,
        chat_history: List[ChatMessage],
        prompts: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> IdentifyResponse:
        """
        Identify whether speaker0 and speaker1 are client or counselor based on
        chat history.

        This method analyzes the chat history to determine the roles of both speakers.
        The chat history should be a list of ChatMessage objects with role and content.

        Args:
            chat_history (List[ChatMessage]): List of messages with role and content
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            IdentifyResponse: Object containing the identified roles for
            speaker0 and speaker1
        """
        logger.info("Identifying users using OpenAI")

        # Format chat history for the prompt
        formatted_conversations = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in chat_history]
        )

        try:
            template = load_template("user/identify_user", prompts=prompts)
            response = cast(
                StructuredIdentifyUsers,
                await self._invoke_llm(
                    template.format(conversations=formatted_conversations),
                    StructuredIdentifyUsers,
                    **kwargs,
                ),
            )
        except LLMInvocationFailedException as e:
            raise IdentifyUserFailedException("Failed to invoke LLM.") from e

        logger.info("Users identified successfully")
        return IdentifyResponse(speaker0=response.speaker0, speaker1=response.speaker1)

    async def get_tag_positivity_ratings(
        self,
        tags: List[str],
        prompts: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Get positivity ratings for a list of tags.

        Parameters:
            tags (List[str]): List of tags to get positivity ratings for.
            **kwargs: Additional keyword arguments to be passed to the
            underlying language model invocation.

        Returns:
            List[Dict]: List of tags with their positivity ratings.

        Raises:
            Exception: If the positivity rating generation fails.
        """
        logger.info("Getting positivity ratings for tags using OpenAI")

        # Format tags for the prompt
        formatted_tags = "\n".join(tags)

        try:
            # Using a list of Tag objects as the structured output
            TagList = create_model("TagList", tags=(List[Tag], ...))

            template = load_template("tags/positivity_rating", prompts=prompts)
            response = cast(
                TagList,
                await self._invoke_llm(
                    template.format(tags=formatted_tags),
                    TagList,
                    **kwargs,
                ),
            )

            # Convert to list of dictionaries
            return [
                {"tag": tag.tag, "positivity_rating": tag.positivity_rating}
                for tag in response.tags
            ]

        except LLMInvocationFailedException as e:
            logger.exception(f"Failed to get positivity ratings: {type(e).__name__}")
            raise Exception("Failed to get positivity ratings for tags.") from e

        except Exception as e:
            logger.exception(
                f"Unexpected error getting positivity ratings: {type(e).__name__}"
            )
            raise Exception(
                "An unexpected error occurred while getting positivity ratings."
            ) from e

    async def diarize_from_transcription(
        self, transcription: str, prompts: Optional[Dict[str, str]] = None, **kwargs
    ) -> StructuredDiarization:
        """
        Diarize a raw transcription string into structured messages with speaker roles.

        This method takes a raw transcription text with timestamps and uses
        OpenAI's structured output
        to parse it into an array of messages with role, content,
        start_time and end_time fields.
        For large transcriptions, it splits the text into chunks
        and processes them in parallel.

        Chunking Strategy:
        - Uses split_text_by_length() to intelligently split large transcriptions
        - Each chunk is limited to MAX_WORDS_PER_CHUNK (2000 words) to avoid OpenAI
        limits
        - Chunks are processed in parallel using asyncio.gather() for efficiency
        - Results are combined in original order to maintain conversation flow
        - Overlap between chunks preserves context and timing information

        Processing Flow:
        1. Split transcription into manageable chunks using accurate word counting
        2. If single chunk: process directly
        3. If multiple chunks: process all chunks in parallel
        4. Combine results maintaining original order
        5. Return structured diarization with all messages

        Token Management:
        - Conservative chunk size (2000 words) to avoid hitting OpenAI's 16,384
        limit
        - Overlap mechanism ensures context preservation between chunks

        Parameters:
            transcription (str): Raw transcription text with timestamps
            from audio
            **kwargs: Additional keyword arguments to be passed to the
            underlying language model invocation.

        Returns:
            StructuredDiarization: Object containing array of messages with
            role, content, start_time and end_time

        Raises:
            LLMInvocationFailedException: If the OpenAI API call fails or chunk
            processing fails

        Example:
            Input: "[00:00:01] Speaker 0: Hello\n[00:00:03] Speaker 1: Hi there"
            Output: StructuredDiarization with messages containing:
                - role: "Speaker 0", content: "Hello", start_time: "00:00:01",
                end_time: "00:00:02"
                - role: "Speaker 1", content: "Hi there", start_time: "00:00:03",
                end_time: "00:00:05"
        """
        logger.info("Diarizing transcription using OpenAI structured output")

        # Split transcription into manageable chunks using accurate word counting
        chunks = split_text_by_length(transcription, MAX_WORDS_PER_CHUNK)
        logger.info(f"Split transcription into {len(chunks)} chunks for processing")

        if len(chunks) == 1:
            # Single chunk - process directly
            logger.info("Processing single chunk")
            try:
                template = load_template("audio/diarization", prompts=prompts)
                result = await self._invoke_llm(
                    template.format(transcription=chunks[0]),
                    StructuredDiarization,
                    **kwargs,
                )
                logger.info("Diarization completed successfully")
                return result
            except Exception as e:
                logger.error(f"Failed to diarize single chunk: {type(e).__name__}")
                raise LLMInvocationFailedException(
                    "Failed to diarize transcription."
                ) from e

        # Multiple chunks - process in parallel for efficiency
        logger.info(f"Processing {len(chunks)} chunks in parallel")

        async def process_chunk(
            chunk_text: str, index: int, prompts: Optional[Dict[str, str]] = None
        ):
            """
            Process a single chunk of transcription.

            This function is called for each chunk in parallel to maximize efficiency.
            Each chunk is processed independently and returns a StructuredDiarization
            object.

            Args:
                chunk_text (str): The text chunk to process
                index (int): Index of the chunk for logging purposes

            Returns:
                StructuredDiarization: Diarization result for this chunk

            Raises:
                LLMInvocationFailedException: If the chunk processing fails
            """
            try:
                logger.debug(
                    f"Processing chunk {index + 1}/{len(chunks)} "
                    f"(length: {len(chunk_text)} chars)"
                )
                template = load_template("audio/diarization", prompts=prompts)
                result = await self._invoke_llm(
                    template.format(transcription=chunk_text),
                    StructuredDiarization,
                    **kwargs,
                )
                logger.debug(
                    f"Chunk {index + 1} completed with {len(result.messages)} messages"
                )
                return result
            except Exception as e:
                logger.error(f"Chunk {index + 1} failed to diarize: {type(e).__name__}")
                raise LLMInvocationFailedException(f"Chunk {index + 1} failed") from e

        try:
            # Process all chunks in parallel using asyncio.gather
            # This maximizes efficiency by processing all chunks simultaneously
            results: List[StructuredDiarization] = await asyncio.gather(
                *[
                    process_chunk(chunk, i, prompts=prompts)
                    for i, chunk in enumerate(chunks)
                ],
                return_exceptions=True,
            )

            # Check for any exceptions in chunk processing
            failed_chunks = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_chunks.append(i + 1)
                    logger.error(f"Chunk {i + 1} failed: {type(result).__name__}")

            if failed_chunks:
                raise LLMInvocationFailedException(
                    f"Failed to process chunks: {failed_chunks}"
                )

            # Combine messages from all chunks in original order
            # This maintains the conversation flow and timing sequence
            combined_messages = []
            total_messages = 0

            for i, result in enumerate(results):
                if isinstance(result, StructuredDiarization):
                    chunk_messages = result.messages
                    combined_messages.extend(chunk_messages)
                    total_messages += len(chunk_messages)
                    logger.debug(
                        f"Added {len(chunk_messages)} messages from chunk {i + 1}"
                    )

            logger.info(
                f"All chunks diarized and merged successfully. "
                f"Total messages: {total_messages}"
            )
            return StructuredDiarization(messages=combined_messages)

        except Exception as e:
            logger.error(f"Failed during parallel diarization: {type(e).__name__}")
            raise LLMInvocationFailedException(
                "Failed to diarize transcription."
            ) from e

    async def analyze_counselor_messages(
        self, chat_history: List[ChatMessage], prompts: Optional[Dict[str, str]] = None
    ) -> Dict[str, int]:
        """
        Analyze counselor messages to extract counts of:
          - Reflective questions
          - Open-ended questions
          - Back-channel cues

        Runs per-message analysis in parallel (asyncio.gather).
        """

        # Extract counselor messages
        counselor_messages = [
            msg.content for msg in chat_history if msg.role.lower() == "counselor"
        ]

        if not counselor_messages:
            return {
                "reflective_questions_asked": 0,
                "open_ended_questions_asked": 0,
                "back_channel_cues": 0,
            }

        async def analyze_single_message(message: str, index: int) -> Dict[str, int]:
            """Analyze one counselor message and return counts as dict."""

            try:
                template = load_template("analysis/counselor_analysis", prompts=prompts)
                prompt = template.format(message=message)
                response = await self._invoke_llm(
                    prompt, output_class=CounselorMessageAnalysis
                )

                return {
                    "reflective_questions_asked": len(response.reflective),
                    "open_ended_questions_asked": len(response.open_ended),
                    "back_channel_cues": len(response.back_channel),
                }

            except Exception as e:
                logger.warning(
                    f"Failed to analyze message {index + 1}: {type(e).__name__}"
                )
                return {
                    "reflective_questions_asked": 0,
                    "open_ended_questions_asked": 0,
                    "back_channel_cues": 0,
                }

        # Run all analyses in parallel
        tasks = [
            analyze_single_message(msg, i) for i, msg in enumerate(counselor_messages)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        totals = {
            "reflective_questions_asked": 0,
            "open_ended_questions_asked": 0,
            "back_channel_cues": 0,
        }

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Task {i + 1} failed with exception: {type(result).__name__}"
                )
                continue
            totals["reflective_questions_asked"] += result["reflective_questions_asked"]
            totals["open_ended_questions_asked"] += result["open_ended_questions_asked"]
            totals["back_channel_cues"] += result["back_channel_cues"]

        return totals

    async def generate_scenario_evaluation(
        self,
        chat_history: List[ChatMessage],
        need_memory: bool = False,
        previous_memory: Optional[str] = None,
        memory_prompt: Optional[str] = None,
        prompts: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate scenario evaluation.

        Uses a single LLM call. Returns improvements, positives, message_tags,
        emotional_movement, and skill_coverage.
        When need_memory is True, also returns session_glimpse and cumulative_memory.

        Parameters:
            chat_history (List[ChatMessage]): List of chat messages/exchanges
            need_memory (bool): Whether to also generate memory fields
            previous_memory (Optional[str]): Previous memory to build upon
            memory_prompt (Optional[str]): Custom instructions for memory generation
            **kwargs: Additional arguments for LLM invocation

        Returns:
            Dict[str, Any]: Dictionary with improvements, positives,
                message_tags, emotional_movement, and skill_coverage.
                When need_memory=True, also includes session_glimpse and
                cumulative_memory.

        Raises:
            LLMInvocationFailedException: If LLM invocation fails
        """
        logger.info("Generating scenario evaluation using OpenAI")

        # Map UUIDs to compact keys to reduce token usage
        uuid_to_key, key_to_uuid = build_id_mapping(chat_history)

        # Convert chat history to string format using keys
        chat_history_str = "\n".join(
            [
                f"[ID: {uuid_to_key[msg.id]}] ({msg.role}): {msg.content}"
                for msg in chat_history
            ]
        )

        # Track valid keys per role for post-processing validation
        counselor_keys: set[str] = set()
        client_keys: list[str] = []

        for msg in chat_history:
            key = uuid_to_key[msg.id]
            if msg.role and msg.role.lower() in ("client", "assistant"):
                client_keys.append(key)
            else:
                counselor_keys.add(key)

        try:
            # Select prompt and response model based on memory requirement
            if need_memory:
                custom_prompt_section = (
                    f"Additional Instructions:\n{memory_prompt}"
                    if memory_prompt
                    else ""
                )
                formatted_prompt = load_and_format(
                    "scenario/scenario_evaluation_with_memory",
                    prompts=prompts,
                    chat_history=chat_history_str,
                    previous_summary=(
                        previous_memory or "No previous summary available."
                    ),
                    custom_prompt_section=custom_prompt_section,
                    MESSAGE_TAG_PROMPT_TEXT=load_template(
                        "shared/message_tags", prompts=prompts
                    ),
                    SKILL_COVERAGE_DESCRIPTIONS=load_template(
                        "shared/skill_coverage", prompts=prompts
                    ),
                )
                response_model = ScenarioEvaluationWithMemory
            else:
                formatted_prompt = load_and_format(
                    "scenario/scenario_evaluation",
                    prompts=prompts,
                    chat_history=chat_history_str,
                    MESSAGE_TAG_PROMPT_TEXT=load_template(
                        "shared/message_tags", prompts=prompts
                    ),
                    SKILL_COVERAGE_DESCRIPTIONS=load_template(
                        "shared/skill_coverage", prompts=prompts
                    ),
                )
                response_model = ScenarioEvaluation

            response = cast(
                response_model,
                await self._invoke_llm(
                    formatted_prompt,
                    response_model,
                    **kwargs,
                ),
            )

            logger.info("Scenario evaluation generated successfully")

            # Build result: validate keys and remap back to UUIDs
            # Convert areas_of_growth to dict format and populate
            # deprecated improvements
            areas_of_growth_list = [
                {
                    "improvement": item.improvement,
                    "recommendation": item.recommendation,
                }
                for item in response.areas_of_growth
            ]

            # For backward compatibility: extract just the improvement strings
            improvements_list = [item.improvement for item in response.areas_of_growth]

            result: Dict[str, Any] = {
                "areas_of_growth": areas_of_growth_list,
                # Deprecated, for backward compatibility
                "improvements": improvements_list,
                "positives": response.positives,
                "message_tags": filter_message_tags(
                    response.message_tags,
                    counselor_keys,
                    key_to_uuid,
                ),
                "emotional_movement": filter_emotional_movement(
                    response.emotional_movement,
                    client_keys,
                    key_to_uuid,
                ),
                "skill_coverage": [
                    {
                        "category": item.category.value,
                        "percentage": int(item.percentage),
                    }
                    for item in response.skill_coverage
                ],
            }

            if need_memory:
                result["session_glimpse"] = response.session_glimpse
                result["cumulative_memory"] = response.cumulative_memory

            return result

        except LLMInvocationFailedException as e:
            logger.exception("Failed to generate scenario evaluation")
            raise LLMInvocationFailedException(
                "Failed to generate scenario evaluation"
            ) from e
