import asyncio
import json
import time
from enum import Enum
from typing import Any, Dict

import boto3
from core.config import settings
from core.message_models import (
    TranscribeAndSummarizeRequestMessage,
    TranscriptionResultMessage,
)
from services import (
    DeepgramTranscriptionService,
    OpenAITranscriptionService,
    SarvamTranscriptionService,
)
from utils.logger import get_logger
from utils.phi_events import PHIEvents
from utils.phi_logger import PHILogEvent, log_sync, phi_logger

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
            provider_str = settings.TRANSCRIPTION_PROVIDER.lower()

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
            if not settings.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is required in settings for OpenAI provider"
                )
            return OpenAITranscriptionService()

        elif provider_enum == TranscriptionProvider.DEEPGRAM:
            if not settings.DEEPGRAM_API_KEY:
                raise ValueError(
                    "DEEPGRAM_API_KEY is required in settings for Deepgram provider"
                )
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required in settings for summarization")
            return DeepgramTranscriptionService()

        elif provider_enum == TranscriptionProvider.SARVAM:
            if not getattr(settings, "SARVAM_API_KEY", None):
                raise ValueError(
                    "SARVAM_API_KEY is required in settings for Sarvam provider"
                )
            return SarvamTranscriptionService()


    async def process_transcription_request(self, request_data: Dict[str, Any], receipt_handle: str) -> Dict[str, Any]:
    
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
            _, segments_text = await transcription_service.transcribe_audio_from_url(
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
                QueueUrl=settings.TRANSCRIPTION_RESULTS_QUEUE_URL,
                MessageBody=json.dumps(result_message.model_dump()),
            )

            # Delete the message from the queue (best-effort in local testing)
            try:
                await asyncio.to_thread(
                    sqs_client.delete_message,
                    QueueUrl=settings.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL,
                    ReceiptHandle=receipt_handle,
                )

                logger.info(
                    f"Successfully deleted message from requests queue for chat_id: {chat_id}"
                )
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.DATA_DELETED,
                        chat_id=chat_id,
                        audit_id=None,  # Will be set by external service,
                        details={
                            "message": f"Successfully deleted message from requests queue for chat_id: {chat_id}",  # noqa: E501
                            "chat_id": chat_id,
                            "receipt_handle": f"{receipt_handle[:20]}...",
                            "request_queue_url": settings.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL,  # noqa: E501
                            "result_queue_url": settings.TRANSCRIPTION_RESULTS_QUEUE_URL,
                            "component": "LambdaHandler",
                            "method": "process_transcription_request",
                        },
                    )
                )
            except Exception as e:
                logger.warning(
                    f"DeleteMessage skipped (likely local invoke with synthetic receipt handle): {type(e).__name__}"
                )
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,  # classify as system event but continue
                        chat_id=chat_id,
                        audit_id=None,
                        details={
                            "message": "DeleteMessage skipped during local run",
                            "exception_type": type(e).__name__,
                            "receipt_handle_prefix": f"{receipt_handle[:20]}...",
                            "request_queue_url": settings.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL,
                            "component": "LambdaHandler",
                            "method": "process_transcription_request",
                        },
                    )
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
                        "request_queue_url": settings.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL,  # noqa: E501
                        "result_queue_url": settings.TRANSCRIPTION_RESULTS_QUEUE_URL,
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
                        "request_queue_url": settings.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL,  # noqa: E501
                        "result_queue_url": settings.TRANSCRIPTION_RESULTS_QUEUE_URL,
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


print(TranscribeAndSummarizeRequestMessage)
print(DeepgramTranscriptionService)
