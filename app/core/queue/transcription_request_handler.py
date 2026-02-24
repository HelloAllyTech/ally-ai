import asyncio
import json
import time
from enum import Enum
from typing import Any, Dict

import boto3
from app.core.config import settings
from app.core.transcriptions.core.message_models import (
    TranscribeAndSummarizeRequestMessage,
    TranscriptionResultMessage,
)
from app.core.transcriptions.services import (
    DeepgramTranscriptionService,
    OpenAITranscriptionService,
    SarvamTranscriptionService,
)
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

    def __init__(self):
        """
        Initialize the transcription request worker.

        """
        logger.info("Initializing transcription services...")
        self.transcription_service = self.create_transcription_service()
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
            print(f"REQUEST DATA {request_data}")
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

            sqs_client = boto3.client("sqs")
            # Send the result message to the result queue
            await asyncio.to_thread(
                sqs_client.send_message,
                QueueUrl=settings.QUEUE.TRANSCRIPTION_RESULTS_QUEUE_URL,
                MessageBody=json.dumps(result_message.model_dump()),
            )

            logger.info(f"Successfully processed chat_id: {chat_id}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=chat_id,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Successfully processed chat_id: {chat_id}",
                        "chat_id": chat_id,
                        "segments_text_length": len(segments_text),
                        "timestamp": int(time.time() * 1000),
                        "request_queue_url": settings.QUEUE.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL,  # noqa: E501
                        "result_queue_url": settings.QUEUE.TRANSCRIPTION_RESULTS_QUEUE_URL,
                        "component": "LambdaHandler",
                        "method": "process_transcription_request",
                    },
                )
            )

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
