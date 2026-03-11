import asyncio
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import boto3
from app.core.ally_core import AllyCoreService
from app.core.config import settings
from app.core.transcriptions.core.message_models import (
    TranscribeAndSummarizeRequestMessage,
    TranscriptionResultMessage,
)
from app.core.text_generations.openai_text_generation_service import (
    OpenAITextGenerationService,
)
from app.core.transcriptions.services import (
    DeepgramTranscriptionService,
    OpenAITranscriptionService,
    SarvamTranscriptionService,
)
from app.schemas.common import ChatMessage
from app.core.transcriptions.utils.logger import get_logger
from app.core.transcriptions.utils.phi_events import PHIEvents
from app.core.transcriptions.utils.phi_logger import PHILogEvent, log_sync, phi_logger

logger = get_logger(__name__)

class TranscriptionProvider(str, Enum):
    """Enumeration of supported transcription providers."""

    OPENAI = "openai"
    DEEPGRAM = "deepgram"
    SARVAM = "sarvam"

class TranscriptionRequestHandler:
    """
    Handler for transcription processing.
    """

    def __init__(
        self,
        ally_core_service: AllyCoreService = None,
        text_generation_service: Optional[OpenAITextGenerationService] = None,
    ):
        """
        Initialize the transcription request worker.

        """
        logger.info("Initializing transcription services...")
        self.transcription_service = self.create_transcription_service()
        self.ally_core_service = ally_core_service
        self.text_generation_service = text_generation_service

        logger.info("Transcription services initialized successfully")

    def create_transcription_service(self, provider: str | None = None):
        """
        Create a transcription service based on the specified provider.

        Args:
            provider (str, optional): Provider to use ('openai', 'deepgram', 'sarvam').
                If None, will use settings.TRANSCRIPTION_PROVIDER.

        Returns:
            The transcription service instance

        Raises:
            ValueError: If provider is not supported or required API keys are missing
        """
        provider_str = provider
        if provider_str is None:
            provider_str = settings.TRANSCRIPTION.PROVIDER.lower()

        try:
            provider_enum = TranscriptionProvider[provider_str.upper()]
        except KeyError:
            raise ValueError(
                f"Unsupported transcription provider: {provider_str}. Supported providers: "
                "'openai', 'deepgram', 'sarvam'"
            )

        logger.info(f"Creating transcription service with provider: {provider_str}")

        # 3. Compare against Enum members
        if provider_enum == TranscriptionProvider.OPENAI:
            if not settings.OPENAI.API_KEY:
                raise ValueError(
                    "OPENAI__API_KEY is required in settings for OpenAI provider"
                )
            return OpenAITranscriptionService()

        elif provider_enum == TranscriptionProvider.DEEPGRAM:
            if not settings.DEEPGRAM.API_KEY:
                raise ValueError(
                    "DEEPGRAM__API_KEY is required in settings for Deepgram provider"
                )
            return DeepgramTranscriptionService()

        elif provider_enum == TranscriptionProvider.SARVAM:
            if not settings.SARVAM.API_KEY:
                raise ValueError(
                    "SARVAM__API_KEY is required in settings for Sarvam provider"
                )
            return SarvamTranscriptionService()


    async def process_transcription_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    
        """
        Process a single transcription request.

        Args:
            request_data: The request data from SQS message

        Returns:
            Dict containing the result or error information
        """
        try:
            # Parse request message
            request = TranscribeAndSummarizeRequestMessage(**request_data)
            chat_id = request.chat_id

            logger.info(f"Processing transcription request for chat_id: {chat_id}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=request.chat_id,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Processing transcription request for chat_id: {chat_id}",  # noqa: E501
                        "chat_id": chat_id,
                        "audio_url": request.audio_url,
                        "sample_rate": request.sample_rate,
                        "component": "LambdaHandler",
                        "method": "process_transcription_request",
                    },
                )
            )

            # Process transcription
            _, segments_text = await self.transcription_service.transcribe_audio_from_url(
                audio_url=request.audio_url,
                chat_id=request.chat_id,
                sample_rate=request.sample_rate,
            )

            # Create result message
            result_message = TranscriptionResultMessage(
                chat_id=chat_id,
                segments_text=segments_text,
                timestamp=int(time.time() * 1000),
            )

            await self._process_transcription(result_message)

            return {"status": "success", "chat_id": chat_id}

        except Exception as e:
            chat_id = request_data.get("chat_id", "unknown")
            logger.exception(
                f"Error processing transcription request: {chat_id} {type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=chat_id,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Error processing transcription request: {chat_id} {type(e).__name__}",  # noqa: E501
                        "chat_id": chat_id,
                        "exception_type": type(e).__name__,
                        "request_queue_url": settings.QUEUE.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL,  # noqa: E501
                        "result_queue_url": settings.QUEUE.TRANSCRIPTION_RESULTS_QUEUE_URL,
                        "component": "LambdaHandler",
                        "method": "process_transcription_request",
                    },
                )
            )
            return {
                "status": "error",
                "error": "Processing failed",
                "chat_id": chat_id,
            }

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
            print(request)
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
                    },
                )
            )
            return True

        except Exception as e:
            chat_id = getattr(request, "chat_id", "unknown")
            logger.error(
                f"Error in diarization/summary for chat_id {chat_id}: "
                f"{e}"
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
                    },
                )
            )
            return False

    async def _generate_summary(self, messages: List[ChatMessage], chat_id: int) -> Dict[str, Any]:
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
                    },
                )
            )
            # Send error response
            await self._send_error_response(chat_id, "Processing failed")
            raise Exception("Summary generation failed")

    async def send_combined_result_to_ally_core(self, chat_id: int, transcription: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
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