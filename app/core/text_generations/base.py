from abc import ABC, abstractmethod
from typing import Dict, List

from app.core.text_generations.structured_output_models import StructuredSummaryNote
from app.schemas.conversation import IdentifyResponse
from app.schemas.common import ChatMessage


class BaseTextGenerationService[ModelT](ABC):
    def __init__(self, models: Dict[str, ModelT], default_model_name: str) -> None:
        if default_model_name not in models:
            raise ValueError(f"Default model '{default_model_name}' not found in provided models.")

        self.models = models
        self.default_model_name = default_model_name

    @abstractmethod
    async def generate_nudge(self, conversation: str, chat_history: str, suggestion: str, **kwargs) -> str:
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
    async def generate_summary_notes(self, chat_history: str) -> StructuredSummaryNote:
        """
        Generate notes for the chat history.

        Parameters:
            chat_history (str): The chat history to summarize.

        Returns:
            StructuredSummaryNote: The summary object.

        Raises:
            SummaryNoteFailedException: If the note generation fails.
        """
        pass

    @abstractmethod
    async def enhance_content(self, content: str, **kwargs) -> str:
        """
        Enhance the content.

        Parameters:
            content (str): The content to enhance.
            **kwargs: Additional keyword arguments to be passed to the underlying language model invocation.

        Returns:
            str: The enhanced content.

        Raises:
            ContentEnhancementFailedException: If the content enhancement fails.
        """
        pass

    @abstractmethod
    async def identify_user(self, latest_message: str, chat_history: List[ChatMessage]) -> IdentifyResponse:
        """
        Identify the users who did the conversation from the conversation history.
        """
        pass
