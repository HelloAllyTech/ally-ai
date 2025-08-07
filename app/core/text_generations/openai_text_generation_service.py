from typing import Type, Optional, List, cast, Dict, Union, TypeVar
import json
import asyncio

import openai
import tiktoken
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.core.text_generations.base import BaseTextGenerationService
from app.core.text_generations.structured_output_models import (
    StructuredSummaryNote,
    StructuredIdentifyUsers,
    StructuredDiarization,
)
from app.core.embeddings.base import BaseEmbeddingService
from app.exceptions.custom_exceptions import (
    LLMInvocationFailedException,
    SummaryNoteFailedException,
    NudgeGenerationFailedException,
    ContentEnhancementFailedException,
    IdentifyUserFailedException
)
from app.schemas.conversation import Nudge, IdentifyResponse
from app.schemas.summary import ContentEnhance, Tag
from app.schemas.common import ChatMessage
from app.schemas.summary import DynamicSummaryNoteResponse, SummaryNoteAndTagsResponse
from pydantic import create_model
from app.utils.affirmation_counter import count_affirmations
from app.utils.client_positivity_lift_calculator import calculate_client_positivity_lift
from app.utils.language_detector import detect_languages
from app.utils.reflective_listening_calculator import calculate_reflective_listening
from app.utils.utterance_duration_calculator import calculate_avg_client_utterance_duration
from app.utils.silence_calculator import calculate_silence_by_counselor
from app.utils.counselor_interruption_calculator import calculate_counselor_interruptions
from app.utils.structured_model_converter import structured_output_model_to_rest
from app.core.text_generations.prompts import (
    SUMMARY_PROMPT,
    DYNAMIC_SUMMARY_PROMPT,
    NUDGE_PROMPT,
    CONTENT_ENHANCE_PROMPT,
    IDENTIFY_USER_PROMPT,
    TAG_POSITIVITY_RATING_PROMPT,
    DIARIZATION_PROMPT
)
from app.utils.logger import get_logger
from app.core.constants import TextGenerationConstants

logger = get_logger(__name__)

# Constants for chunking
MAX_TOKENS_PER_CHUNK = 3000  # Conservative limit to avoid token issues
CHUNK_OVERLAP = 500  # Overlap between chunks to maintain context

@tool
def generate_dynamic_summary(fields: dict[str, Union[str, int]]) -> dict[str, Union[str, int]]:
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
                    validated_fields[key] = value.split(',') if isinstance(value, str) else value
                else:
                    validated_fields[key] = value
            else:
                validated_fields[key] = value

        # Validate using StructuredSummaryNote
        _ = StructuredSummaryNote(**validated_fields)
        return validated_fields
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return {}

def split_text_by_length(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> List[str]:
    """
    Split transcription text into chunks while preserving timing information.
    
    This function intelligently splits transcription text to avoid OpenAI's token limits
    while maintaining context and preserving timestamp information. It uses tiktoken
    for accurate token counting to ensure chunks are properly sized.
    
    Token Counting Strategy:
    - Uses OpenAI's tiktoken library for accurate token estimation
    - Processes text line-by-line to maintain natural boundaries
    - Preserves timestamp information and speaker identification
    - Adds overlap between chunks to maintain conversation context
    
    Chunking Strategy:
    1. Split text into lines (preserving timestamps and speaker info)
    2. Group lines until approaching max_tokens limit
    3. Create overlap between chunks using timestamp-containing lines
    4. Ensure no chunk exceeds the token limit
    
    Args:
        text (str): Raw transcription text with timestamps and speaker information
        max_tokens (int): Maximum tokens per chunk (default: MAX_TOKENS_PER_CHUNK)
    
    Returns:
        List[str]: List of text chunks, each within the token limit
        
    Example:
        Input: "[00:00:01] Speaker 0: Hello\n[00:00:03] Speaker 1: Hi there"
        Output: ["[00:00:01] Speaker 0: Hello\n[00:00:03] Speaker 1: Hi there"]
        
    Token Estimation Examples:
        - "[00:00:01] Speaker 0: Hello" → ~15 tokens (accurate with tiktoken)
        - "[00:00:05] Speaker 1: How are you today?" → ~12 tokens
    """
    lines = text.splitlines()
    chunks = []
    current_chunk = []
    current_len = 0
    
    # Use tiktoken for accurate token counting
    encoding = tiktoken.encoding_for_model(TextGenerationConstants.DEFAULT_MODEL)
    
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count using OpenAI's tiktoken library.
        
        This provides accurate token counts that match OpenAI's actual tokenization,
        ensuring we never exceed token limits and create optimally sized chunks.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Accurate token count
        """
        return len(encoding.encode(text))
    
    # Process each line individually to maintain natural boundaries
    for line in lines:
        line_tokens = estimate_tokens(line)
        
        # If adding this line would exceed the limit and we have content, start a new chunk
        if current_len + line_tokens > max_tokens and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_len = 0
        
        current_chunk.append(line)
        current_len += line_tokens
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    # Add overlap between chunks to maintain conversation context
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            # Add some lines from the end of current chunk to the beginning of next chunk
            current_lines = chunks[i].splitlines()
            next_lines = chunks[i + 1].splitlines()
            
            # Take last few lines from current chunk (if they contain timestamps)
            # This helps maintain context between chunks
            overlap_lines = []
            for line in reversed(current_lines[-3:]):  # Last 3 lines
                if any(char.isdigit() for char in line):  # Likely contains timestamp
                    overlap_lines.insert(0, line)
                if estimate_tokens('\n'.join(overlap_lines)) > CHUNK_OVERLAP:
                    break
            
            if overlap_lines:
                chunks[i + 1] = '\n'.join(overlap_lines + next_lines)
    
    return chunks


class OpenAITextGenerationService(BaseTextGenerationService[ChatOpenAI]):
    """Text generation service using OpenAI models."""

    def __init__(self, client: ChatOpenAI, embedding_service: BaseEmbeddingService) -> None:
        """
        Initialize the OpenAITextGenerator with a client and embedding service.

        Parameters:
            client (ChatOpenAI): The OpenAI chat client to use.
            embedding_service (BaseEmbeddingService): The embedding service for reflective listening calculation.
        """
        # Store the embedding service
        self.embedding_service = embedding_service

        # Initialize the base class with the client
        super().__init__(client)

    async def _invoke_llm[T](
            self,
            messages: List[BaseMessage] | str,
            output_class: Optional[Type[T]] = None,
            **kwargs,
    ) -> T | str:
        """
        Invoke the LLM asynchronously with optional structured output.

        Parameters:
            messages (List[BaseMessage] | str):
                - A list of messages (`List[BaseMessage]`) for conversational input.
                - A raw string (`str`) if a single message is provided.
            output_class (Optional[type[T]]):
                - An optional Pydantic model or class to structure the output.
                - If provided, the response will be parsed into an instance of this class.
            **kwargs:
                - Additional keyword arguments.

        Returns:
            T | str:
                - If `output_class` is provided, returns an instance of `output_class` (T).
                - Otherwise, returns the raw response content as a string.
        """
        # Use the model from base class
        llm = self.model

        # Bind structured output class with the llm if passed by user
        if output_class:
            llm = llm.with_structured_output(output_class)

        try:
            response = await llm.ainvoke(messages)
        except openai.RateLimitError as e:
            logger.exception(str(e))
            raise LLMInvocationFailedException("OpenAI API rate limit exceeded. Please try again later.") from e

        except openai.APIConnectionError as e:
            logger.exception(str(e))
            raise LLMInvocationFailedException("OpenAI API error. Please try again later.") from e

        return response if output_class else response.content

    async def generate_nudge(self, conversation: str, chat_history: str, suggestion: str, **kwargs) -> str:
        """
        Generate a nudge using the OpenAI language model.

        This method formats a prompt with the provided conversation context, chat history,
        and suggestion, then invokes the language model to generate a nudge. The response is
        cast to a NudgeOutput object from which the nudge text is extracted and returned.

        Parameters:
            conversation (str): The current conversation context.
            chat_history (str): A string representation of the chat history to provide context.
            suggestion (str): A suggestion or prompt to guide the nudge generation.
            **kwargs: Additional keyword arguments to be passed to the underlying language model invocation.

        Returns:
            str: The generated nudge extracted from the language model's response.

        Raises:
            NudgeGenerationFailedException: If the OpenAI API rate limit is exceeded or an API connection error occurs.
        """
        logger.info("Generating nudge using OpenAI")
        try:
            response = cast(
                Nudge,
                await self._invoke_llm(
                    NUDGE_PROMPT.format(conversation=conversation, chat_history=chat_history, suggestion=suggestion),
                    Nudge,
                    **kwargs))

        except LLMInvocationFailedException as e:
            raise NudgeGenerationFailedException("Failed to invoke LLM.") from e

        logger.info("Nudge generated successfully")

        return response.nudge

    async def generate_summary_notes(
            self,
            chat_history: List[ChatMessage],
            keys: Optional[List[str]] = None,
            **kwargs
    ) -> Union[SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse]:
        """
        Generate summary notes from chat history.

        Parameters:
            chat_history (List[ChatMessage]): The chat history to summarize as a list of ChatMessage objects
            keys (Optional[List[str]]): Optional list of keys to generate. If provided, returns a DynamicSummaryNoteResponse
                with only the requested fields. If None, returns a SummaryNoteAndTagsResponse with all predefined fields.
            **kwargs: Additional keyword arguments

        Returns:
            Union[SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse]: The generated summary notes

        Raises:
            SummaryNoteFailedException: If the summary generation fails
        """
        logger.info("Generating summary notes using OpenAI")
        try:
            # Convert chat history to string
            chat_history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])

            # Count affirmations
            affirmation_count = count_affirmations(chat_history)

            # Get the embedding service
            embedding_service = self.embedding_service

            if keys:
                languages = None
                reflective_listening_score = None
                avg_client_utterance_duration = None
                silence_by_counselor = None
                client_positivity_lift = None
                counselor_interruptions = None
                logger.info("Generating dynamic summary")
                # Get field descriptions from StructuredSummaryNote
                key_descriptions = []
                for key in keys:
                    if key == "languages":
                        languages = detect_languages(chat_history_str)
                    elif key == "affirmations" and affirmation_count > 0:
                        # If affirmations is requested and we have a count, we'll add it later
                        continue
                    elif key == "reflective_listening":
                        # Calculate reflective listening if requested
                        reflective_listening_score = await calculate_reflective_listening(chat_history,
                                                                                          embedding_service)
                        continue
                    elif key == "avg_client_utterance_duration":
                        avg_client_utterance_duration = calculate_avg_client_utterance_duration(chat_history)
                        continue
                    elif key == "silence_by_counselor":
                        silence_by_counselor = calculate_silence_by_counselor(chat_history)
                        continue
                    elif key == "client_positivity_lift":
                        client_positivity_lift = calculate_client_positivity_lift(chat_history)
                        continue
                    elif key == "counselor_interruptions":
                        counselor_interruptions = calculate_counselor_interruptions(chat_history)
                        continue
                    else:
                        # Get the field from StructuredSummaryNote if it exists
                        field = StructuredSummaryNote.model_fields.get(key)
                        if field:
                            description = field.description
                            key_descriptions.append(f"- {key}: {description}")
                        else:
                            logger.info(f"Key not found in StructuredSummaryNote: {key}")

                if key_descriptions:
                    key_descriptions_text = "\n".join(key_descriptions)
                    prompt = DYNAMIC_SUMMARY_PROMPT.format(
                        chat_history=chat_history_str,
                        key_descriptions=key_descriptions_text
                    )

                    # Get the model and bind the tool
                    model = self.model
                    model = model.bind_tools([generate_dynamic_summary])

                    # Generate the response
                    response = await model.ainvoke(prompt)

                    # Extract the tool call result
                    if hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
                        tool_call = response.additional_kwargs['tool_calls'][0]
                        if tool_call['function']['name'] == 'generate_dynamic_summary':
                            fields = json.loads(tool_call['function']['arguments'])
                            fields_dict = fields.get('fields', {})

                            # Add languages to fields if it was requested in keys
                            if languages and 'languages' in keys:
                                fields_dict['languages'] = [lang.model_dump() for lang in languages]

                            # Add affirmations count if requested
                            if 'affirmations' in keys and affirmation_count > 0:
                                fields_dict['affirmations'] = affirmation_count

                            # Add reflective listening if requested
                            if reflective_listening_score is not None:
                                fields_dict['reflective_listening'] = reflective_listening_score

                            # Add avg_client_utterance_duration if requested
                            if avg_client_utterance_duration is not None:
                                fields_dict['avg_client_utterance_duration'] = avg_client_utterance_duration

                            # Add silence_by_counselor if requested
                            if 'silence_by_counselor' in keys:
                                fields_dict['silence_by_counselor'] = silence_by_counselor
                                
                            # Add client_positivity_lift if requested
                            if 'client_positivity_lift' in keys:
                                fields_dict['client_positivity_lift'] = client_positivity_lift

                            # Add counselor_interruptions if requested
                            if 'counselor_interruptions' in keys:
                                fields_dict['counselor_interruptions'] = counselor_interruptions

                            logger.info("Note generated successfully")
                            return DynamicSummaryNoteResponse(fields=fields_dict)
                else:
                    # If no tool call result but languages was requested, return just languages
                    fields_dict = {}

                    if languages and 'languages' in keys:
                        logger.info("Adding languages field")
                        fields_dict['languages'] = [lang.model_dump() for lang in languages]

                    # Add affirmations count if requested
                    if 'affirmations' in keys:
                        logger.info("Adding affirmations field")
                        fields_dict['affirmations'] = affirmation_count

                    # Add reflective listening if requested
                    if 'reflective_listening' in keys:
                        logger.info("Adding reflective_listening field")
                        reflective_listening_score = await calculate_reflective_listening(chat_history,
                                                                                          embedding_service)
                        fields_dict['reflective_listening'] = reflective_listening_score

                    # Add avg_client_utterance_duration if requested
                    if 'avg_client_utterance_duration' in keys:
                        logger.info("Adding avg_client_utterance_duration field")
                        avg_client_utterance_duration = calculate_avg_client_utterance_duration(chat_history)
                        fields_dict['avg_client_utterance_duration'] = avg_client_utterance_duration

                    # Add silence_by_counselor if requested
                    if 'silence_by_counselor' in keys:
                        logger.info("Adding silence_by_counselor field")
                        silence_by_counselor = calculate_silence_by_counselor(chat_history)
                        fields_dict['silence_by_counselor'] = silence_by_counselor
                        
                    # Add client_positivity_lift if requested
                    if 'client_positivity_lift' in keys:
                        logger.info("Adding client_positivity_lift field")
                        client_positivity_lift = calculate_client_positivity_lift(chat_history)
                        fields_dict['client_positivity_lift'] = client_positivity_lift

                    # Add counselor_interruptions if requested
                    if 'counselor_interruptions' in keys:
                        logger.info("Adding counselor_interruptions field")
                        counselor_interruptions = calculate_counselor_interruptions(chat_history)
                        fields_dict['counselor_interruptions'] = counselor_interruptions

                    if fields_dict:
                        logger.info("Returning fields")
                        return DynamicSummaryNoteResponse(fields=fields_dict)

                    logger.info("No fields found in the response")
                    return DynamicSummaryNoteResponse(fields={})
            else:
                # Handle structured summary without keys
                prompt = SUMMARY_PROMPT.format(chat_history=chat_history_str)

                logger.info("Generating structured summary")
                # Generate structured summary
                response = cast(
                    StructuredSummaryNote,
                    await self._invoke_llm(
                                prompt,
                        StructuredSummaryNote,
                                **kwargs
                            )
                        )
                
                logger.info("Note generated successfully")

                response.affirmations = affirmation_count

                # Convert the summary to a response using the appropriate converter
                response = structured_output_model_to_rest(response)
                languages = detect_languages(chat_history_str)

                response.languages = languages
                response.affirmations = affirmation_count

                # Calculate reflective listening using the embedding service
                reflective_listening = await calculate_reflective_listening(chat_history, embedding_service)
                response.reflective_listening = reflective_listening

                # Calculate avg_client_utterance_duration
                avg_client_utterance_duration = calculate_avg_client_utterance_duration(chat_history)
                response.avg_client_utterance_duration = avg_client_utterance_duration

                # Calculate silence_by_counselor
                silence_by_counselor = calculate_silence_by_counselor(chat_history)
                response.silence_by_counselor = silence_by_counselor
                
                # Calculate client_positivity_lift
                client_positivity_lift = calculate_client_positivity_lift(chat_history)
                response.client_positivity_lift = client_positivity_lift

                # Calculate counselor_interruptions
                counselor_interruptions = calculate_counselor_interruptions(chat_history)
                response.counselor_interruptions = counselor_interruptions

                return response

        except LLMInvocationFailedException as e:
            logger.error(f"Failed to invoke LLM: {str(e)}")
            raise SummaryNoteFailedException("Failed to invoke LLM.") from e
        except Exception as e:
            logger.exception(f"Failed to generate summary: {str(e)}")
            raise SummaryNoteFailedException(f"Failed to generate summary: {str(e)}") from e

    async def enhance_content(self, content: str, **kwargs) -> str:
        """
        Enhance the content by generating a summary and tags.

        Parameters:
            content (str): The content to enhance.
            **kwargs: Additional keyword arguments to be passed to the underlying language model invocation.

        Returns:
            str: The enhanced content.

        Raises:
            ContentEnhancementFailedException: If the content enhancement fails.
        """
        logger.info("Enhancing content using OpenAI")
        try:
            response = cast(
                ContentEnhance,
                await self._invoke_llm(
                    CONTENT_ENHANCE_PROMPT.format(content=content),
                    ContentEnhance,
                    **kwargs))

        except LLMInvocationFailedException as e:
            raise ContentEnhancementFailedException("Failed to invoke LLM.") from e

        logger.info("Content enhanced successfully")
        return response.enhanced_content
    
    async def identify_user(self, chat_history: List[ChatMessage], **kwargs) -> IdentifyResponse:
        """
        Identify whether speaker0 and speaker1 are client or counselor based on chat history.

        This method analyzes the chat history to determine the roles of both speakers.
        The chat history should be a list of ChatMessage objects with role and content.

        Args:
            chat_history (List[ChatMessage]): List of messages with role and content
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            IdentifyResponse: Object containing the identified roles for speaker0 and speaker1
        """
        logger.info("Identifying users using OpenAI")

        # Format chat history for the prompt
        formatted_conversations = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])

        try:
            response = cast(
                StructuredIdentifyUsers,
                await self._invoke_llm(
                    IDENTIFY_USER_PROMPT.format(conversations=formatted_conversations),
                    StructuredIdentifyUsers,
                    **kwargs
                )
            )
        except LLMInvocationFailedException as e:
            raise IdentifyUserFailedException("Failed to invoke LLM.") from e

        logger.info("Users identified successfully")
        return IdentifyResponse(speaker0=response.speaker0, speaker1=response.speaker1)

    async def get_tag_positivity_ratings(self, tags: List[str], **kwargs) -> List[Dict]:
        """
        Get positivity ratings for a list of tags.

        Parameters:
            tags (List[str]): List of tags to get positivity ratings for.
            **kwargs: Additional keyword arguments to be passed to the underlying language model invocation.

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
            TagList = create_model('TagList', tags=(List[Tag], ...))

            response = cast(
                TagList,
                await self._invoke_llm(
                    TAG_POSITIVITY_RATING_PROMPT.format(tags=formatted_tags),
                    TagList,
                    **kwargs
                )
            )

            # Convert to list of dictionaries
            return [{"tag": tag.tag, "positivity_rating": tag.positivity_rating} for tag in response.tags]

        except LLMInvocationFailedException as e:
            logger.exception(f"Failed to get positivity ratings: {str(e)}")
            raise Exception("Failed to get positivity ratings for tags.") from e

        except Exception as e:
            logger.exception(f"Unexpected error getting positivity ratings: {str(e)}")
            raise Exception("An unexpected error occurred while getting positivity ratings.") from e

    async def diarize_from_transcription(self, transcription: str, **kwargs) -> StructuredDiarization:
        """
        Diarize a raw transcription string into structured messages with speaker roles.
        
        This method takes a raw transcription text with timestamps and uses OpenAI's structured output
        to parse it into an array of messages with role, content, start_time and end_time fields.
        For large transcriptions, it splits the text into chunks and processes them in parallel.
        
        Chunking Strategy:
        - Uses split_text_by_length() to intelligently split large transcriptions
        - Each chunk is limited to MAX_TOKENS_PER_CHUNK (3000 tokens) to avoid OpenAI limits
        - Chunks are processed in parallel using asyncio.gather() for efficiency
        - Results are combined in original order to maintain conversation flow
        - Overlap between chunks preserves context and timing information
        
        Processing Flow:
        1. Split transcription into manageable chunks using accurate token counting
        2. If single chunk: process directly
        3. If multiple chunks: process all chunks in parallel
        4. Combine results maintaining original order
        5. Return structured diarization with all messages
        
        Token Management:
        - Conservative chunk size (3000 tokens) to avoid hitting OpenAI's 16,384 limit
        - Overlap mechanism ensures context preservation between chunks
        
        Parameters:
            transcription (str): Raw transcription text with timestamps from audio
            **kwargs: Additional keyword arguments to be passed to the underlying language model invocation.
            
        Returns:
            StructuredDiarization: Object containing array of messages with role, content, start_time and end_time
            
        Raises:
            LLMInvocationFailedException: If the OpenAI API call fails or chunk processing fails
            
        Example:
            Input: "[00:00:01] Speaker 0: Hello\n[00:00:03] Speaker 1: Hi there"
            Output: StructuredDiarization with messages containing:
                - role: "Speaker 0", content: "Hello", start_time: "00:00:01", end_time: "00:00:02"
                - role: "Speaker 1", content: "Hi there", start_time: "00:00:03", end_time: "00:00:05"
        """
        logger.info("Diarizing transcription using OpenAI structured output")
        
        # Split transcription into manageable chunks using accurate token counting
        chunks = split_text_by_length(transcription, MAX_TOKENS_PER_CHUNK)
        logger.info(f"Split transcription into {len(chunks)} chunks for processing")
        
        if len(chunks) == 1:
            # Single chunk - process directly
            logger.info("Processing single chunk")
            try:
                result = await self._invoke_llm(
                    DIARIZATION_PROMPT.format(transcription=chunks[0]),
                    StructuredDiarization,
                    **kwargs
                )
                logger.info("Diarization completed successfully")
                return result
            except Exception as e:
                logger.error(f"Failed to diarize single chunk: {str(e)}")
                raise LLMInvocationFailedException("Failed to diarize transcription.") from e
        
        # Multiple chunks - process in parallel for efficiency
        logger.info(f"Processing {len(chunks)} chunks in parallel")
        
        async def process_chunk(chunk_text: str, index: int):
            """
            Process a single chunk of transcription.
            
            This function is called for each chunk in parallel to maximize efficiency.
            Each chunk is processed independently and returns a StructuredDiarization object.
            
            Args:
                chunk_text (str): The text chunk to process
                index (int): Index of the chunk for logging purposes
                
            Returns:
                StructuredDiarization: Diarization result for this chunk
                
            Raises:
                LLMInvocationFailedException: If the chunk processing fails
            """
            try:
                logger.debug(f"Processing chunk {index + 1}/{len(chunks)} (length: {len(chunk_text)} chars)")
                result = await self._invoke_llm(
                    DIARIZATION_PROMPT.format(transcription=chunk_text),
                    StructuredDiarization,
                    **kwargs
                )
                logger.debug(f"Chunk {index + 1} completed with {len(result.messages)} messages")
                return result
            except Exception as e:
                logger.error(f"Chunk {index + 1} failed to diarize: {str(e)}")
                raise LLMInvocationFailedException(f"Chunk {index + 1} failed") from e

        try:
            # Process all chunks in parallel using asyncio.gather
            # This maximizes efficiency by processing all chunks simultaneously
            results: List[StructuredDiarization] = await asyncio.gather(
                *[process_chunk(chunk, i) for i, chunk in enumerate(chunks)],
                return_exceptions=True
            )
            
            # Check for any exceptions in chunk processing
            failed_chunks = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_chunks.append(i + 1)
                    logger.error(f"Chunk {i + 1} failed: {str(result)}")
            
            if failed_chunks:
                raise LLMInvocationFailedException(f"Failed to process chunks: {failed_chunks}")

            # Combine messages from all chunks in original order
            # This maintains the conversation flow and timing sequence
            combined_messages = []
            total_messages = 0
            
            for i, result in enumerate(results):
                if isinstance(result, StructuredDiarization):
                    chunk_messages = result.messages
                    combined_messages.extend(chunk_messages)
                    total_messages += len(chunk_messages)
                    logger.debug(f"Added {len(chunk_messages)} messages from chunk {i + 1}")

            logger.info(f"All chunks diarized and merged successfully. Total messages: {total_messages}")
            return StructuredDiarization(messages=combined_messages)

        except Exception as e:
            logger.error(f"Failed during parallel diarization: {str(e)}")
            raise LLMInvocationFailedException("Failed to diarize transcription.") from e
