from typing import Any, Dict, List, Optional, Tuple

from app.core.text_generations.base import BaseTextGenerationService
from app.core.vector_db.base import VectorDB
from app.exceptions.custom_exceptions import (
    ConversationAnalysisFailedException,
    ConversationIdentifyFailedException,
    IdentifyUserFailedException,
    NudgeGenerationFailedException,
    VectorDBFetchFailedException,
)
from app.schemas.common import ChatMessage
from app.schemas.conversation import IdentifyResponse
from app.utils.common import convert_chat_messages_to_string
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationService:
    def __init__(
        self, text_generation_service: BaseTextGenerationService, vector_db: VectorDB
    ) -> None:
        self.text_generation_service = text_generation_service
        self.vector_db = vector_db

    async def analyze(
        self,
        latest_message: str,
        chat_history: List[ChatMessage],
        force_nudge: bool = False,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Analyzes the latest message in the chat history and generates a nudge if
        necessary.

        Parameters:
            latest_message (str): The latest message in the chat history.
            chat_history (List[ChatMessage]): The full chat history.
            force_nudge (bool): Whether to always generate a nudge.

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple where the first element is
            the current conversation stage (or None/"Unknown" if not available) and
            the second element is the generated nudge (if any).

        Raises:
            ConversationAnalysisFailedException: If fetching relevant conversations
            from the vector database
                or generating the nudge fails.
        """
        try:
            # Retrieves the most relevant conversations
            relevant_conversations = await self.vector_db.fetch_relevant_conversations(
                latest_message, top_k=1
            )

        except VectorDBFetchFailedException as e:
            raise ConversationAnalysisFailedException(
                "Failed to fetch relevant conversations. Please try again later."
            ) from e

        if relevant_conversations:
            objects = getattr(
                relevant_conversations, "objects", [None]
            )  # Safely get 'objects' if it exists

            stage = None
            generated_nudge = None

            if top_nudge := (objects or [None])[0]:
                # Safely access the 'properties' dict
                properties = getattr(top_nudge, "properties", {})
                nudge = properties.get("nudge")
                nudge_conversation = properties.get("conversation")
                stage = properties.get("stage", None)

                generated_nudge = None
                messages = convert_chat_messages_to_string(chat_history)

                # Generate nudge only if it is forced or a nudge exists in similar
                # conversation
                if nudge or force_nudge:
                    try:
                        generated_nudge = (
                            await self.text_generation_service.generate_nudge(
                                nudge_conversation,
                                messages,
                                nudge,
                                prompts=prompts,
                            )
                        )

                    except NudgeGenerationFailedException as e:
                        raise ConversationAnalysisFailedException(
                            "Failed to generate nudge. Please try again later."
                        ) from e

        else:
            # TODO: Handle case when no relevant conversation is found

            generated_nudge = None
            stage = "Unknown"

        return stage, generated_nudge

    async def identify(
        self, chat_history: List[ChatMessage], prompts: Optional[Dict[str, Any]] = None
    ) -> IdentifyResponse:
        """
        Identifies the users who did the conversation from the conversation history.
        """
        try:
            return await self.text_generation_service.identify_user(
                chat_history, prompts=prompts
            )

        except IdentifyUserFailedException as e:
            raise ConversationIdentifyFailedException(
                "Failed to identify users. Please try again later."
            ) from e
