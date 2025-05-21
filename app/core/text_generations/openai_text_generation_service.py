from typing import Type, Optional, List, cast, Dict

import openai
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.text_generations.base import BaseTextGenerationService
from app.core.text_generations.prompts import NUDGE_PROMPT, SUMMARY_PROMPT, CONTENT_ENHANCE_PROMPT, IDENTIFY_USER_PROMPT, TAG_POSITIVITY_RATING_PROMPT
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
from pydantic import create_model

logger = get_logger(__name__)


class OpenAITextGenerationService(BaseTextGenerationService[ChatOpenAI]):
    """Text generation service using OpenAI models."""

    def __init__(self, model_name: str, *model_names: str) -> None:
        """
        Initialize the OpenAITextGenerator with one or more model names.

        Parameters:
            model_name (str): The primary model name (mandatory).
            *model_names (str): Additional optional model names.
        """
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

    async def generate_summary_notes(self, chat_history: str, **kwargs) -> StructuredSummaryNote:
        """
        Generate notes for the chat history using the OpenAI language model.

        Parameters:
            chat_history (str): Chat history to summarize.
            **kwargs: Additional keyword arguments to be passed to the underlying language model invocation.

        Returns:
            StructuredSummaryNote: Object with different fields of the summary.

        Raises:
            SummaryNoteFailedException: If the note generation fails.
        """
        logger.info("Generating note summary using OpenAI")
        try:
            response = cast(
                StructuredSummaryNote,
                await self._invoke_llm(
                    SUMMARY_PROMPT.format(chat_history=chat_history),
                    StructuredSummaryNote,
                    **kwargs))

        except LLMInvocationFailedException as e:
            raise SummaryNoteFailedException("Failed to invoke LLM.") from e

        logger.info("Note generated successfully")
        return response

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
    
    async def identify_user(self, latest_message: str, chat_history: List[ChatMessage], **kwargs) -> IdentifyResponse:
        """
        Identify whether speaker0 and speaker1 are client or counselor based on chat history.

        This method analyzes the chat history and latest message to determine the roles of both speakers.
        The latest message should be in the format "role: content" (e.g., "speaker0: I'm feeling anxious").
        The chat history should be a list of ChatMessage objects with role and content.

        Args:
            latest_message (str): The latest message in the format "role: content"
            chat_history (List[ChatMessage]): List of previous messages with role and content
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            IdentifyResponse: Object containing the identified roles for speaker0 and speaker1
        """
        logger.info("Identifying users using OpenAI")
        
        # Format chat history for the prompt
        formatted_conversations = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
        if latest_message:
            formatted_conversations += f"\n{latest_message}"
        
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
