import time
from typing import List, Optional, Union

from app.core.phi_events import PHIEvents
from app.core.phi_logger import PHILogEvent, phi_logger
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
        self,
        chat_history: List[ChatMessage],
        chat_id: Optional[str] = None,
        keys: Optional[List[str]] = None,
    ) -> Union[SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse]:
        """
        Generate a summary and tags from the given chat history.

        Parameters:
            chat_history (List[ChatMessage]): The chat history to summarize
            chat_id (Optional[str]): The chat ID for PHI logging.
            as a list of ChatMessage objects.
            keys (Optional[List[str]]): The keys to include in the summary.

        Returns:
            Union[SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse]:
                - If keys are provided, returns a DynamicSummaryNoteResponse
                - Otherwise, returns a SummaryNoteAndTagsResponse

        Raises:
            Exception: If summary generation fails.
        """
        start_time = time.time()

        try:
            # Generate the summary note
            result = await self.text_generation_service.generate_summary_notes(
                chat_history, keys, chat_id=chat_id
            )

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Log successful completion
            await phi_logger.log(
                PHILogEvent(
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by caller
                    event_type=PHIEvents.DATA_MODIFIED,
                    details={
                        "message": "Summary generation completed",
                        "component": "SummaryService",
                        "processing_time_ms": processing_time_ms,
                        "summary_length": (
                            len(result.session_summary)
                            if hasattr(result, "session_summary")
                            and result.session_summary
                            else (
                                len(result.fields.get("session_summary", ""))
                                if hasattr(result, "fields")
                                and result.fields.get("session_summary")
                                else 0
                            )
                        ),
                        "tags_count": (
                            len(result.tags)
                            if hasattr(result, "tags") and result.tags
                            else (
                                len(result.fields.get("tags", []))
                                if hasattr(result, "fields")
                                and result.fields.get("tags")
                                else 0
                            )
                        ),
                    },
                )
            )

            return result

        except SummaryNoteFailedException as e:
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Log error
            await phi_logger.log(
                PHILogEvent(
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by caller
                    event_type=PHIEvents.SYSTEM_ERROR,
                    details={
                        "error": f"Failed to generate summary: {type(e).__name__}",
                        "component": "SummaryService",
                        "method": "generate_summary_and_tags",
                        "exception_type": type(e).__name__,
                        "chat_history_count": len(chat_history),
                        "processing_time_ms": processing_time_ms,
                        "summary_type": "dynamic" if keys else "structured",
                        "keys_provided": keys is not None,
                    },
                )
            )

            logger.error(f"Failed to generate summary: {type(e).__name__}")
            raise SummarizationFailedException(
                "Failed to generate the summary. Please try again later."
            ) from e

    async def enhance_content(self, content: str, chat_id: Optional[str] = None) -> str:
        """
        Enhance the given content using the text generation service.

        Parameters:
            content (str): The content to enhance.
            chat_id (Optional[str]): The chat ID for PHI logging.

        Returns:
            str: The enhanced content.

        Raises:
            ContentEnhancementFailedException: If content enhancement fails.
        """
        start_time = time.time()

        try:
            enhanced_content = await self.text_generation_service.enhance_content(
                content, chat_id=chat_id
            )

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Log successful completion
            await phi_logger.log(
                PHILogEvent(
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by caller
                    event_type=PHIEvents.DATA_MODIFIED,
                    details={
                        "message": "Content enhancement completed",
                        "component": "SummaryService",
                        "processing_time_ms": processing_time_ms,
                        "enhanced_content_length": len(enhanced_content),
                    },
                )
            )

            return enhanced_content

        except ContentEnhancementFailedException as e:
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Log error
            await phi_logger.log(
                PHILogEvent(
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by caller
                    event_type=PHIEvents.SYSTEM_ERROR,
                    details={
                        "error": f"Failed to enhance content: {type(e).__name__}",
                        "component": "SummaryService",
                        "method": "enhance_content",
                        "exception_type": type(e).__name__,
                        "content_length": len(content),
                        "processing_time_ms": processing_time_ms,
                    },
                )
            )

            raise SummarizationFailedException(
                "Failed to enhance content. Please try again later."
            ) from e

    async def get_tag_positivity_ratings(
        self, tags: list[str], chat_id: Optional[str] = None
    ) -> list[Tag]:
        """
        Get positivity ratings for a list of tags.

        Parameters:
            tags (list[str]): List of tags to get positivity ratings for.
            chat_id (Optional[str]): The chat ID for PHI logging.

        Returns:
            list[Tag]: List of tags with their positivity ratings.

        Raises:
            SummarizationFailedException: If the positivity rating generation fails.
        """
        start_time = time.time()

        try:
            tag_ratings = await self.text_generation_service.get_tag_positivity_ratings(
                tags, chat_id=chat_id
            )

            # Convert the list of dictionaries to a list of Tag objects
            result = [
                Tag(tag=item["tag"], positivity_rating=item["positivity_rating"])
                for item in tag_ratings
            ]

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Log successful completion
            await phi_logger.log(
                PHILogEvent(
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by caller
                    event_type=PHIEvents.DATA_MODIFIED,
                    details={
                        "message": "Tag positivity rating generation completed",
                        "component": "SummaryService",
                        "processing_time_ms": processing_time_ms,
                        "result_count": len(result),
                    },
                )
            )

            return result

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Log error
            await phi_logger.log(
                PHILogEvent(
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by caller
                    event_type=PHIEvents.SYSTEM_ERROR,
                    details={
                        "error": f"Failed to get positivity ratings for tags: {type(e).__name__}",  # noqa: E501
                        "component": "SummaryService",
                        "method": "get_tag_positivity_ratings",
                        "exception_type": type(e).__name__,
                        "tags_count": len(tags),
                        "processing_time_ms": processing_time_ms,
                    },
                )
            )

            raise SummarizationFailedException(
                "Failed to get positivity ratings for tags. Please try again later."
            ) from e

    async def generate_scenario_evaluation(
        self,
        chat_history: List[ChatMessage],
        need_memory: bool = False,
        previous_memory: Optional[str] = None,
        memory_prompt: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        """
        Generate scenario evaluation.

        Parameters:
            chat_history (List[ChatMessage]): The conversation between AI
                client and counselor.
            need_memory (bool): Whether to generate memory summary alongside
                evaluation.
            previous_memory (Optional[str]): Previous memory summary to build
                upon (when need_memory=True).
            memory_prompt (Optional[str]): Custom instructions for memory
                generation (when need_memory=True).
            chat_id (Optional[str]): The chat ID for PHI logging.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "improvements": Array of areas needing development
                - "positives": Array of demonstrated strengths
                - "message_tags": Per-message tags for counselor messages
                - "emotional_movement": Emotional trajectory of client messages
                - "skill_coverage": Skill coverage percentages across categories
                - "session_glimpse": Brief session overview (only if need_memory=True)
                - "cumulative_memory": Cumulative narrative (only if need_memory=True)

        Raises:
            CounselorTrainingAnalysisFailedException: If evaluation generation fails.
        """
        start_time = time.time()

        try:
            result = await self.text_generation_service.generate_scenario_evaluation(
                chat_history,
                need_memory=need_memory,
                previous_memory=previous_memory,
                memory_prompt=memory_prompt,
                chat_id=chat_id,
            )

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Log successful completion
            await phi_logger.log(
                PHILogEvent(
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by caller
                    event_type=PHIEvents.DATA_MODIFIED,
                    details={
                        "message": "Scenario evaluation completed",
                        "component": "SummaryService",
                        "processing_time_ms": processing_time_ms,
                        "memory_generated": need_memory,
                        "result_keys": (
                            list(result.keys()) if isinstance(result, dict) else []
                        ),
                    },
                )
            )

            return result

        except LLMInvocationFailedException as e:
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Log error
            await phi_logger.log(
                PHILogEvent(
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by caller
                    event_type=PHIEvents.SYSTEM_ERROR,
                    details={
                        "error": (
                            f"Failed to generate scenario evaluation: "
                            f"{type(e).__name__}"
                        ),
                        "component": "SummaryService",
                        "method": "generate_scenario_evaluation",
                        "exception_type": type(e).__name__,
                        "chat_history_count": len(chat_history),
                        "processing_time_ms": processing_time_ms,
                    },
                )
            )

            logger.error(f"Failed to generate scenario evaluation: {str(e)}")
            raise CounselorTrainingAnalysisFailedException(
                "Failed to generate scenario evaluation. " "Please try again later."
            ) from e
