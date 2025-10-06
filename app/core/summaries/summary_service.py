from typing import List, Optional, Union

from app.core.text_generations.base import BaseTextGenerationService
from app.exceptions.custom_exceptions import (
    ContentEnhancementFailedException,
    CounselorTrainingAnalysisFailedException,
    LLMInvocationFailedException,
    SummarizationFailedException,
    SummaryNoteFailedException,
)
from app.schemas.common import ChatMessage
from app.schemas.summary import (
    DynamicSummaryNoteResponse,
    SummaryNoteAndTagsResponse,
    Tag,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SummaryService:
    def __init__(self, text_generation_service: BaseTextGenerationService) -> None:
        self.text_generation_service = text_generation_service

    async def generate_summary_and_tags(
        self, chat_history: List[ChatMessage], keys: Optional[List[str]] = None
    ) -> Union[SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse]:
        """
        Generate a summary and tags from the given chat history.

        Parameters:
            chat_history (List[ChatMessage]): The chat history to summarize
            as a list of ChatMessage objects.
            keys (Optional[List[str]]): The keys to include in the summary.

        Returns:
            Union[SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse]:
                - If keys are provided, returns a DynamicSummaryNoteResponse
                - Otherwise, returns a SummaryNoteAndTagsResponse

        Raises:
            Exception: If summary generation fails.
        """
        try:
            # Generate the summary note
            return await self.text_generation_service.generate_summary_notes(
                chat_history, keys
            )
        except SummaryNoteFailedException as e:
            logger.error(f"Failed to generate summary: {type(e).__name__}")
            raise SummarizationFailedException(
                "Failed to generate the summary. Please try again later."
            ) from e

    async def enhance_content(self, content: str) -> str:
        """
        Enhance the given content using the text generation service.

        Parameters:
            content (str): The content to enhance.

        Returns:
            str: The enhanced content.

        Raises:
            ContentEnhancementFailedException: If content enhancement fails.
        """
        try:
            enhanced_content = await self.text_generation_service.enhance_content(
                content
            )
        except ContentEnhancementFailedException as e:
            raise SummarizationFailedException(
                "Failed to enhance content. Please try again later."
            ) from e

        return enhanced_content

    async def get_tag_positivity_ratings(self, tags: list[str]) -> list[Tag]:
        """
        Get positivity ratings for a list of tags.

        Parameters:
            tags (list[str]): List of tags to get positivity ratings for.

        Returns:
            list[Tag]: List of tags with their positivity ratings.

        Raises:
            SummarizationFailedException: If the positivity rating generation fails.
        """
        try:
            tag_ratings = await self.text_generation_service.get_tag_positivity_ratings(
                tags
            )

            # Convert the list of dictionaries to a list of Tag objects
            return [
                Tag(tag=item["tag"], positivity_rating=item["positivity_rating"])
                for item in tag_ratings
            ]

        except Exception as e:
            raise SummarizationFailedException(
                "Failed to get positivity ratings for tags. Please try again later."
            ) from e

    async def generate_simulation_summary(
        self, chat_history: List[ChatMessage], goal: str
    ):
        """
        Generate counselor training simulation analysis from chat history and goal.

        Parameters:
            chat_history (List[str]): The conversation between AI client and counselor
            as a list of string messages.
            goal (str): The specific training objective to evaluate against.

        Returns:
            Dict[str, List[str]]: Dictionary containing:
                - "improvements": Array of areas needing development
                - "positives": Array of demonstrated strengths

        Raises:
            CounselorTrainingAnalysisFailedException: If analysis generation fails.
        """
        try:
            # Generate the counselor training analysis
            return await self.text_generation_service.generate_simulation_summary(
                chat_history, goal
            )
        except LLMInvocationFailedException as e:
            logger.error(f"Failed to generate counselor training analysis: {str(e)}")
            raise CounselorTrainingAnalysisFailedException(
                "Failed to generate counselor training analysis. "
                "Please try again later."
            ) from e
