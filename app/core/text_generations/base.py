from abc import ABC, abstractmethod
from typing import Dict


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
