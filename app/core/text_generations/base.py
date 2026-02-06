from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from app.schemas.common import ChatMessage
from app.schemas.conversation import IdentifyResponse
from app.schemas.summary import DynamicSummaryNoteResponse, SummaryNoteAndTagsResponse


class BaseTextGenerationService[ModelT](ABC):
    def __init__(self, model: ModelT) -> None:
        """
        Initialize the base text generation service with a model.

        Parameters:
            model (ModelT): The model to use for text generation.
        """
        self.model = model

    @abstractmethod
    async def generate_nudge(
        self, conversation: str, chat_history: str, suggestion: str, **kwargs
    ) -> str:
        """
        Generate a nudge based on the conversation.

        Parameters:
            conversation (str): The conversation to generate a nudge for.
            chat_history (str): The chat history to consider.
            suggestion (str): The suggestion to base the nudge on.
            **kwargs: Additional keyword arguments to be passed

        Returns:
            str: The generated nudge.

        Raises:
            NudgeGenerationFailedException: If the nudge generation fails.
        """
        pass

    @abstractmethod
    async def generate_summary_notes(
        self, chat_history: List[ChatMessage], keys: Optional[List[str]] = None
    ) -> Union[SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse]:
        """
        Generate summary notes from chat history.

        Parameters:
            chat_history (List[ChatMessage]): The chat history to summarize as a
                list of ChatMessage objects
            keys (Optional[List[str]]): Optional list of keys to generate. If
                provided, returns a DynamicSummaryNoteResponse with only the
                requested fields. If None, returns a SummaryNoteAndTagsResponse
                with all predefined fields.

        Returns:
            Union[SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse]:
            The generated summary notes
        """
        pass

    @abstractmethod
    async def enhance_content(self, content: str, **kwargs) -> str:
        """
        Enhance the content.

        Parameters:
            content (str): The content to enhance.
            **kwargs: Additional keyword arguments to be passed to the underlying
            language model invocation.

        Returns:
            str: The enhanced content.

        Raises:
            ContentEnhancementFailedException: If the content enhancement fails.
        """
        pass

    @abstractmethod
    async def identify_user(self, chat_history: List[ChatMessage]) -> IdentifyResponse:
        """
        Identify the users who did the conversation from the conversation history.
        """
        pass

    @abstractmethod
    async def get_tag_positivity_ratings(self, tags: List[str]) -> List[Dict]:
        """
        Get positivity ratings for a list of tags.

        Parameters:
            tags (List[str]): List of tags to get positivity ratings for.

        Returns:
            List[Dict]: List of tags with their positivity ratings.

        Raises:
            Exception: If the positivity rating generation fails.
        """
        pass

    @abstractmethod
    async def analyze_counselor_messages(
        self, chat_history: List[ChatMessage]
    ) -> Dict[str, int]:
        """
        Analyze a single counselor message asynchronously.

        Args:
            message (str): The message text that needs to be analyzed.
            index (int): The position of the
            message in the list, useful for tracking or debugging.

        Returns:
            Any: The result of the analysis. This could be an integer score,
                 a dictionary with structured results, or raise an exception
                 if the analysis fails.
        """
        pass

    @abstractmethod
    async def generate_simulation_summary(
        self,
        chat_history: List[ChatMessage],
        need_memory: bool = False,
        previous_memory: Optional[str] = None,
        memory_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate simulation summary analyzing chat history, with optional memory.

        Uses a single LLM call. When need_memory is False, returns only improvements
        and positives. When need_memory is True, returns all four fields in one call.

        Parameters:
            chat_history (List[ChatMessage]): List of chat messages/exchanges
            need_memory (bool): Whether to also generate memory fields
            previous_memory (Optional[str]): Previous memory to build upon
                (when need_memory=True)
            memory_prompt (Optional[str]): Custom instructions for memory generation
                (when need_memory=True)
            **kwargs: Additional arguments for LLM invocation

        Returns:
            Dict[str, Any]: Dictionary with 'improvements' and 'positives' arrays.
                When need_memory=True, also includes 'session_glimpse' and
                'cumulative_memory'.

        Raises:
            LLMInvocationFailedException: If LLM invocation fails
        """
        pass
