from app.core.text_generations.base import BaseTextGenerationService
from app.exceptions.custom_exceptions import (
    SummarizationFailedException,
    SummaryNoteFailedException,
    ContentEnhancementFailedException,
)
from app.schemas.summary import SummaryNoteAndTagsResponse, Tag
from app.utils.logger import get_logger
from app.utils.structured_model_converter import structured_output_model_to_rest

logger = get_logger(__name__)


class SummaryService:
    def __init__(self, text_generation_service: BaseTextGenerationService) -> None:
        self.text_generation_service = text_generation_service

    async def generate_summary_and_tags(self, chat_history: str) -> SummaryNoteAndTagsResponse:
        """
        Generate a summary and associated tags from the provided chat history.

        Parameters:
            chat_history (str): A string representing the chat history to be summarized and analyzed.

        Returns:
            SummaryNoteAndTagsResponse: An object that contains the summary note along with a list of tags derived from the chat history.

        Raises:
            SummarizationFailedException: If the summary generation or tagging process fails.
            NotImplementedError: If conversion for the type of `sop_model` is not implemented.
        """
        try:
            summary = await self.text_generation_service.generate_summary_notes(chat_history)
        except SummaryNoteFailedException as e:
            raise SummarizationFailedException("Failed to generate the summary. Please try again later.") from e

        return structured_output_model_to_rest(summary)

    async def enhance_content(self, content: str) -> str:
        """
        Enhances the content

        Parameters:
            content (str): The content to enhance.

        Returns:
            str: The enhanced content.

        Raises:
            ContentEnhancementFailedException: If content enhancement fails.
        """
        try:
            enhanced_content = await self.text_generation_service.enhance_content(content)
        except ContentEnhancementFailedException as e:
            raise SummarizationFailedException("Failed to enhance content. Please try again later.") from e

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
            tag_ratings = await self.text_generation_service.get_tag_positivity_ratings(tags)
            
            # Convert the list of dictionaries to a list of Tag objects
            return [Tag(tag=item["tag"], positivity_rating=item["positivity_rating"]) for item in tag_ratings]
            
        except Exception as e:
            raise SummarizationFailedException("Failed to get positivity ratings for tags. Please try again later.") from e
