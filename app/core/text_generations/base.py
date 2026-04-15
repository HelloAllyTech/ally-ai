from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic

from app.schemas.common import ChatMessage
from app.schemas.conversation import IdentifyResponse
from app.schemas.summary import DynamicSummaryNoteResponse, SummaryNoteAndTagsResponse


ModelT = TypeVar("ModelT")


class BaseTextGenerationService(Generic[ModelT], ABC):
    def __init__(self, model: ModelT) -> None:
        """
        Initialize the base text generation service with a model.

        Parameters:
            model (ModelT): The model to use for text generation.
        """
        self.model = model

    @abstractmethod
    async def generate_nudge(
        self,
        conversation: str,
        chat_history: str,
        suggestion: str,
        prompts: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Generate a nudge based on the conversation.

        Parameters:
            conversation (str): The conversation to generate a nudge for.
            chat_history (str): The chat history to consider.
            suggestion (str): The suggestion to base the nudge on.
            prompts (Optional[Dict[str, Any]]): Optional prompt overrides.
            **kwargs: Additional keyword arguments to be passed

        Returns:
            str: The generated nudge.

        Raises:
            NudgeGenerationFailedException: If the nudge generation fails.
        """
        pass

    @abstractmethod
    async def generate_summary_notes(
        self,
        chat_history: List[ChatMessage],
        keys: Optional[List[str]] = None,
        prompts: Optional[Dict[str, Any]] = None,
        session_mode: Optional[str] = None,
        **kwargs,
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
            prompts (Optional[Dict[str, Any]]): Optional prompt overrides.
            session_mode (Optional[str]): SCRIBE vs DICTATION; selects summary prompt.
            **kwargs: Additional keyword arguments to be passed.

        Returns:
            Union[SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse]:
            The generated summary notes
        """
        pass

    @abstractmethod
    async def enhance_content(
        self, content: str, prompts: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """
        Enhance the content.

        Parameters:
            content (str): The content to enhance.
            prompts (Optional[Dict[str, Any]]): Optional prompt overrides.
            **kwargs: Additional keyword arguments to be passed to the underlying
            language model invocation.

        Returns:
            str: The enhanced content.

        Raises:
            ContentEnhancementFailedException: If the content enhancement fails.
        """
        pass

    @abstractmethod
    async def identify_user(
        self,
        chat_history: List[ChatMessage],
        prompts: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> IdentifyResponse:
        """
        Identify the users who did the conversation from the conversation history.

        Parameters:
            chat_history (List[ChatMessage]): The chat history to identify from.
            prompts (Optional[Dict[str, Any]]): Optional prompt overrides.
            **kwargs: Additional keyword arguments to be passed.

        Returns:
            IdentifyResponse: The identified user roles.
        """
        pass

    @abstractmethod
    async def get_tag_positivity_ratings(
        self,
        tags: List[str],
        prompts: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Get positivity ratings for a list of tags.

        Parameters:
            tags (List[str]): List of tags to get positivity ratings for.
            prompts (Optional[Dict[str, Any]]): Optional prompt overrides.
            **kwargs: Additional keyword arguments to be passed.

        Returns:
            List[Dict]: List of tags with their positivity ratings.

        Raises:
            Exception: If the positivity rating generation fails.
        """
        pass

    @abstractmethod
    async def analyze_counselor_messages(
        self,
        chat_history: List[ChatMessage],
        prompts: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, int]:
        """
        Analyze counselor messages from chat history to extract counts of
        reflective questions, open-ended questions, and back-channel cues.

        Parameters:
            chat_history (List[ChatMessage]): The chat history to analyze.
            prompts (Optional[Dict[str, Any]]): Optional prompt overrides.
            **kwargs: Additional keyword arguments to be passed.

        Returns:
            Dict[str, int]: A dictionary containing counts for different metrics.
        """
        pass

    @abstractmethod
    async def generate_scenario_evaluation(
        self,
        chat_history: List[ChatMessage],
        need_memory: bool = False,
        previous_memory: Optional[str] = None,
        memory_prompt: Optional[str] = None,
        prompts: Optional[Dict[str, Any]] = None,
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
                (when need_memory=True)
            memory_prompt (Optional[str]): Custom instructions for memory generation
                (when need_memory=True)
            prompts (Optional[Dict[str, Any]]): Optional prompt overrides.
            **kwargs: Additional arguments for LLM invocation

        Returns:
            Dict[str, Any]: Dictionary with 'improvements', 'positives', 'message_tags',
                'emotional_movement', and 'skill_coverage'. When need_memory=True, also
                includes 'session_glimpse' and 'cumulative_memory'.

        Raises:
            LLMInvocationFailedException: If LLM invocation fails
        """
        pass
