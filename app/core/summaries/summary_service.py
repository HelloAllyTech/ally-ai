from app.core.text_generations.base import BaseTextGenerationService
from app.exceptions.custom_exceptions import SummarizationFailedException, SummaryNoteFailedException
from app.schemas.summary import SummaryNote
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SummaryService:
    def __init__(self, text_generation_service: BaseTextGenerationService) -> None:
        self.text_generation_service = text_generation_service

    async def generate_summary_and_tags(self, chat_history: str) -> SummaryNote:
        """
        Generates a summary and tags for the given chat history.

        Parameters:
            chat_history (str): The chat history to summarize and tag.

        Returns:
            SummaryNote: The summary object.

        Raises:
            SummarizationFailedException: If the summary generation fails.
        """
        try:
            summary = await self.text_generation_service.generate_summary_notes(chat_history)
        except SummaryNoteFailedException as e:
            raise SummarizationFailedException("Failed to generate the summary. Please try again later.") from e

        return summary
