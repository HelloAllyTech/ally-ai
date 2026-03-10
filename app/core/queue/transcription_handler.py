from typing import Any, Dict, List, Optional

from app.core.ally_core import AllyCoreService
from app.core.phi_events import PHIEvents
from app.core.phi_logger import PHILogEvent, phi_logger
from app.core.queue.message_models import TranscriptionResultMessage
from app.core.text_generations.openai_text_generation_service import (
    OpenAITextGenerationService,
)
from app.schemas.common import ChatMessage
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TranscriptionHandler:
    """
    Handler for processing transcription requests and sending responses via queues.
    """

    def __init__(
        self,
        ally_core_service: AllyCoreService = None,
        request_queue_url: str = None,
        text_generation_service: Optional[OpenAITextGenerationService] = None,
    ):
        """
        Initialize the transcription handler.

        Parameters:
            ally_core_service (AllyCoreService): The service to use for interaction with
            ally backend (core).
            text_generation_service (Optional[OpenAITextGenerationService]): The text
            generation service for diarization and summary.
        """
        self.ally_core_service = ally_core_service
        self.request_queue_url = request_queue_url
        self.text_generation_service = text_generation_service  # Use passed service

    async def process_request(self, message_data: Dict[str, Any]) -> None:
        """
        Process a transcription request message.

        Parameters:
            message_data (Dict[str, Any]): The message data from the queue.
        """
        try:
            chat_id = message_data.get("chat_id", "unknown")
            logger.info(f"Processing transcription request for chat_id: {chat_id}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=chat_id,
                    audit_id=None,  # Will be set by caller
                    details={
                        "message": f"Processing transcription request for chat_id: {chat_id}",
                        "chat_id": chat_id,
                        "component": "TranscriptionHandler",
                        "method": "process_request",
                        "request_queue_url": self.request_queue_url,
                    },
                )
            )

            # Parse the request message
            request = TranscriptionResultMessage(**message_data)

            # Process the transcription and get results
            success = await self._process_transcription(request)

            if success:
                logger.info(
                    f"Transcription processing completed successfully for chat_id: "
                    f"{chat_id}"
                )
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.DATA_MODIFIED,
                        chat_id=chat_id,
                        audit_id=None,  # Will be set by caller
                        details={
                            "message": f"Transcription processing completed successfully for chat_id: {chat_id}",
                            "chat_id": chat_id,
                            "component": "TranscriptionHandler",
                            "method": "process_request",
                            "status": "success",
                            "request_queue_url": self.request_queue_url,
                        },
                    )
                )
            else:
                logger.error(f"Transcription processing failed for chat_id: {chat_id}")
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=chat_id,
                        audit_id=None,  # Will be set by caller
                        details={
                            "error": f"Transcription processing failed for chat_id: {chat_id}",
                            "chat_id": chat_id,
                            "component": "TranscriptionHandler",
                            "method": "process_request",
                            "status": "failed",
                            "request_queue_url": self.request_queue_url,
                        },
                    )
                )
                # Send error response
                await self._send_error_response(chat_id, "Processing failed")

        except Exception as e:
            chat_id = message_data.get("chat_id", "unknown")
            logger.exception(
                f"Error processing transcription request for chat_id {chat_id}: "
                f"{type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=chat_id,
                    audit_id=None,
                    details={
                        "error": f"Error processing transcription request for chat_id {chat_id}: {type(e).__name__}",
                        "chat_id": chat_id,
                        "component": "TranscriptionHandler",
                        "method": "process_request",
                        "exception_type": type(e).__name__,
                        "request_queue_url": self.request_queue_url,
                    },
                )
            )
            # Send error response
            await self._send_error_response(chat_id, "Processing failed")

    async def _process_transcription(self, request: TranscriptionResultMessage) -> bool:
        """
        Process the transcription result from Lambda and do diarization + summary.

        Parameters:
            request (TranscriptionResultMessage): The transcription result
            message with segments_text.

        Returns:
            bool: True if processing was successful.
        """
        try:
            chat_id = request.chat_id
            segments_text = request.segments_text

            logger.info(f"Processing diarization and summary for chat_id: {chat_id}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by caller
                    details={
                        "message": f"Processing diarization and summary for chat_id: {chat_id}",
                        "chat_id": chat_id,
                        "component": "TranscriptionHandler",
                        "method": "_process_transcription",
                        "segments_text_length": (
                            len(segments_text) if segments_text else 0
                        ),
                        "request_queue_url": self.request_queue_url,
                    },
                )
            )

            # Do diarization
            diarization_result = (
                await self.text_generation_service.diarize_from_transcription(
                    transcription=segments_text
                )
            )

            # Convert to ChatMessage objects
            messages = [
                ChatMessage(
                    role=msg.role.upper(),  # Convert to uppercase for consistency
                    content=msg.content,
                    start_time=msg.start_time,
                    end_time=msg.end_time,
                )
                for msg in diarization_result.messages
            ]

            # Generate summary
            summary = await self._generate_summary(messages, chat_id)

            # Convert to data format
            transcription_data = [
                msg.model_dump() if hasattr(msg, "model_dump") else msg
                for msg in messages
            ]
            summary_data = (
                summary.model_dump() if hasattr(summary, "model_dump") else summary
            )

            # Send the results to ally core
            await self.send_combined_result_to_ally_core(
                chat_id, transcription_data, summary_data
            )

            logger.info(f"Diarization and summary completed for chat_id {chat_id}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=chat_id,
                    audit_id=None,
                    details={
                        "message": f"Diarization and summary completed for chat_id {chat_id}",
                        "chat_id": chat_id,
                        "component": "TranscriptionHandler",
                        "method": "_process_transcription",
                        "messages_count": len(messages),
                        "transcription_data_length": len(transcription_data),
                        "request_queue_url": self.request_queue_url,
                    },
                )
            )
            return True

        except Exception as e:
            chat_id = getattr(request, "chat_id", "unknown")
            logger.error(
                f"Error in diarization/summary for chat_id {chat_id}: "
                f"{type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=chat_id,
                    audit_id=None,
                    details={
                        "error": f"Error in diarization/summary for chat_id {chat_id}: {type(e).__name__}",
                        "chat_id": chat_id,
                        "component": "TranscriptionHandler",
                        "method": "_process_transcription",
                        "exception_type": type(e).__name__,
                        "request_queue_url": self.request_queue_url,
                    },
                )
            )
            return False

    async def _generate_summary(
        self, messages: List[ChatMessage], chat_id: int
    ) -> Dict[str, Any]:
        """
        Generate summary from diarized messages.

        Parameters:
            messages (List[ChatMessage]): The diarized messages.
            chat_id (int): The chat ID.

        Returns:
            Dict[str, Any]: The summary data.
        """
        try:
            # Generate summary using text generation service
            summary = await self.text_generation_service.generate_summary_notes(
                chat_history=messages, keys=None
            )

            return summary

        except Exception as e:
            logger.error(
                f"Error generating summary for chat_id {chat_id}: {type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by caller
                    details={
                        "error": f"Error generating summary for chat_id {chat_id}: {type(e).__name__}",
                        "chat_id": chat_id,
                        "component": "TranscriptionHandler",
                        "method": "_generate_summary",
                        "exception_type": type(e).__name__,
                        "messages_count": len(messages),
                        "request_queue_url": self.request_queue_url,
                    },
                )
            )
            # Send error response
            await self._send_error_response(chat_id, "Processing failed")
            raise Exception("Summary generation failed")

    async def send_combined_result_to_ally_core(
        self, chat_id: int, transcription: List[Dict[str, Any]], summary: Dict[str, Any]
    ) -> None:
        """
        Send transcription and summary results to ally core using its
        process-transcript API.

        Parameters:
            chat_id (int): The chat ID.
            transcription (List[Dict[str, Any]]): The transcription data.
            summary (Dict[str, Any]): The summary data.
        """
        try:
            # Create message with presigned URLs
            await self.ally_core_service.process_transcript(
                chat_id=chat_id,
                transcription=transcription,
                summary=summary,
            )

            logger.info(f"Sent transcription and summary to core for chat_id: {chat_id}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "message": f"Sent presigned URLs to queue for chat_id: {chat_id}",
                        "chat_id": chat_id,
                        "component": "TranscriptionHandler",
                        "method": "send_combined_result_to_ally_core",
                        "transcription_count": len(transcription),
                        "summary_keys": list(summary.keys()) if summary else [],
                    },
                )
            )

        except Exception as e:
            logger.error(
                f"Error sending combined result to queue for chat_id {chat_id}: "
                f"{type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "error": f"Error sending combined result to queue for chat_id {chat_id}: {type(e).__name__}",
                        "chat_id": chat_id,
                        "component": "TranscriptionHandler",
                        "method": "send_combined_result_to_ally_core",
                        "exception_type": type(e).__name__,
                        "transcription_count": len(transcription),
                        "summary_keys": list(summary.keys()) if summary else [],
                    },
                )
            )
            # Send error response
            await self._send_error_response(chat_id, "Processing failed")

    async def _send_error_response(self, chat_id: Any, error_message: str) -> None:
        """Send error response to the results queue."""
        try:
            await self.ally_core_service.process_transcript(
                chat_id=chat_id,
                error=error_message,
            )

            logger.info(f"Error response sent for chat_id: {chat_id}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "message": f"Error response sent for chat_id: {chat_id}",
                        "chat_id": chat_id,
                        "component": "TranscriptionHandler",
                        "method": "_send_error_response",
                        "error_message": error_message,
                    },
                )
            )

        except Exception as e:
            logger.error(
                f"Failed to send error response for chat_id {chat_id}: "
                f"{type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "error": f"Failed to send error response for chat_id {chat_id}: {type(e).__name__}",
                        "chat_id": chat_id,
                        "component": "TranscriptionHandler",
                        "method": "_send_error_response",
                        "exception_type": type(e).__name__,
                        "error_message": error_message,
                    },
                )
            )
