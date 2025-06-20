from typing import Type, Optional, List, cast, Dict, Union
import json

import openai
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.embeddings.base import BaseEmbeddingService
from app.core.text_generations.base import BaseTextGenerationService
from app.core.text_generations.prompts import NUDGE_PROMPT, SUMMARY_PROMPT, DYNAMIC_SUMMARY_PROMPT, \
    CONTENT_ENHANCE_PROMPT, IDENTIFY_USER_PROMPT, TAG_POSITIVITY_RATING_PROMPT
from app.core.text_generations.structured_output_models import StructuredSummaryNote, StructuredIdentifyUsers
from app.exceptions.custom_exceptions import (
    NudgeGenerationFailedException,
    LLMInvocationFailedException,
    SummaryNoteFailedException,
    ContentEnhancementFailedException,
    IdentifyUserFailedException
)
from app.schemas.conversation import Nudge, IdentifyResponse
from app.schemas.summary import ContentEnhance, Tag
from app.utils.logger import get_logger
from app.schemas.common import ChatMessage
from app.utils.structured_model_converter import structured_output_model_to_rest
from app.schemas.summary import DynamicSummaryNoteResponse, SummaryNoteAndTagsResponse
from pydantic import create_model
from app.utils.language_detector import detect_languages
from app.utils.affirmation_counter import count_affirmations
from app.utils.reflective_listening_calculator import calculate_reflective_listening

logger = get_logger(__name__)


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


class OpenAITextGenerationService(BaseTextGenerationService[ChatOpenAI]):
    """Text generation service using OpenAI models."""

    def __init__(self, model_name: str, embedding_service: BaseEmbeddingService, *model_names: str) -> None:
        """
        Initialize the OpenAITextGenerator with one or more model names and embedding service.

        Parameters:
            model_name (str): The primary model name (mandatory).
            embedding_service (BaseEmbeddingService): The embedding service for reflective listening calculation.
            *model_names (str): Additional optional model names.
        """
        # Store the embedding service
        self.embedding_service = embedding_service
        
        # Combine the primary model name with any additional ones
        all_model_names = (model_name,) + model_names

        # Create a dictionary to cache model instances keyed by model name
        models = {
            name: ChatOpenAI(
                model_name=name,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_organization=settings.OPENAI_ORGANIZATION_ID
            ) for name in all_model_names
        }

        # Initialize the base class with the model instances and the default model name
        super().__init__(models, model_name)

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
                - Additional keyword arguments (e.g., `model_name` to select a specific model).

        Returns:
            T | str:
                - If `output_class` is provided, returns an instance of `output_class` (T).
                - Otherwise, returns the raw response content as a string.
        """
        llm = self.models[kwargs.get("model_name", self.default_model_name)]

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
        logger.info("Generating note summary using OpenAI")
        try:
            # Count affirmations directly from ChatMessage objects
            affirmation_count = count_affirmations(chat_history)
            
            # Use the injected embedding service for reflective listening calculation
            embedding_service = self.embedding_service

            # Convert ChatMessage objects to string format for LLM prompt
            chat_history_str = '\n'.join(f'{msg.role}: {msg.content}' for msg in chat_history)

            if keys:
                languages = None
                reflective_listening_score = None
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
                        reflective_listening_score = await calculate_reflective_listening(chat_history, embedding_service)
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
                    model = self.models[kwargs.get("model_name", self.default_model_name)]
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

                            logger.info("Note generated successfully")
                            return DynamicSummaryNoteResponse(fields=fields_dict)
                else:
                    # If no tool call result but languages was requested, return just languages
                    fields_dict = {}

                    if languages and 'languages' in keys:
                        logger.info("Adding languages field")
                        fields_dict['languages'] = [lang.model_dump() for lang in languages]

                    # Add affirmations count if requested
                    if 'affirmations' in keys :
                        logger.info("Adding affirmations field")
                        fields_dict['affirmations'] = affirmation_count

                    # Add reflective listening if requested
                    if 'reflective_listening' in keys:
                        logger.info("Adding reflective_listening field")
                        reflective_listening_score = await calculate_reflective_listening(chat_history, embedding_service)
                        fields_dict['reflective_listening'] = reflective_listening_score

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
