import asyncio
import json
from typing import Any, Dict, List, Optional

from app.core.phi_events import PHIEvents
from app.core.phi_logger import PHILogEvent, phi_logger
from app.core.queue.message_models import (
    MessageType,
    TranscribeAndSummarizeResponseMessage,
    TranscriptionResultMessage,
)
from app.core.queue.sqs_queue_service import SQSQueueService
from app.core.storage.s3_service import S3Service
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
        queue_service: Optional[SQSQueueService] = None,
        request_queue_url: str = None,
        result_queue_url: str = None,
        text_generation_service: Optional[OpenAITextGenerationService] = None,
        storage_service: Optional[S3Service] = None,
        bucket_name: str = None,
    ):
        """
        Initialize the transcription handler.

        Parameters:
            queue_service (Optional[SQSQueueService]): The queue service to use for
            sending responses.
            text_generation_service (Optional[OpenAITextGenerationService]): The text
            generation service for diarization and summary.
            storage_service (Optional[S3Service]): The storage service to use for
            uploading results.
            bucket_name (str): The name of the bucket to use for uploading results.
        """
        self.queue_service = queue_service
        self.request_queue_url = request_queue_url
        self.result_queue_url = result_queue_url
        self.text_generation_service = text_generation_service  # Use passed service
        self.storage_service = storage_service
        self.bucket_name = bucket_name

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
                        "result_queue_url": self.result_queue_url,
                        "bucket_name": self.bucket_name,
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
                            "result_queue_url": self.result_queue_url,
                            "bucket_name": self.bucket_name,
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
                            "result_queue_url": self.result_queue_url,
                            "bucket_name": self.bucket_name,
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
                        "result_queue_url": self.result_queue_url,
                        "bucket_name": self.bucket_name,
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
                        "result_queue_url": self.result_queue_url,
                        "bucket_name": self.bucket_name,
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

            # Send the results to bucket and queue
            await self.send_combined_result_to_queue(
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
                        "result_queue_url": self.result_queue_url,
                        "bucket_name": self.bucket_name,
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
                        "result_queue_url": self.result_queue_url,
                        "bucket_name": self.bucket_name,
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
                        "result_queue_url": self.result_queue_url,
                        "bucket_name": self.bucket_name,
                    },
                )
            )
            # Send error response
            await self._send_error_response(chat_id, "Processing failed")
            raise Exception("Summary generation failed")

    async def send_combined_result_to_queue(
        self, chat_id: int, transcription: List[Dict[str, Any]], summary: Dict[str, Any]
    ) -> None:
        """
        Upload transcription and summary results to bucket and send
        presigned URLs to the result queue.

        Parameters:
            chat_id (int): The chat ID.
            transcription (List[Dict[str, Any]]): The transcription data.
            summary (Dict[str, Any]): The summary data.
        """
        try:
            # Create the result payload
            result_payload = {
                "chat_id": chat_id,
                "transcription": transcription,
                "summary": summary,
            }

            # Create the bucket object key
            bucket_object_key = f"transcription-results/result_{chat_id}.json"

            # Upload results to bucket
            await self.storage_service.upload_to_s3(
                bucket_name=self.bucket_name,
                object_key=bucket_object_key,
                payload=result_payload,
            )

            # Generate presigned URLs for download and delete
            download_presigned_url = (
                await self.storage_service.generate_presigned_download_url(
                    bucket_name=self.bucket_name,
                    object_key=bucket_object_key,
                    expiration=3600,  # 1 hour
                )
            )

            delete_presigned_url = (
                await self.storage_service.generate_presigned_delete_url(
                    bucket_name=self.bucket_name,
                    object_key=bucket_object_key,
                    expiration=3600,  # 1 hour
                )
            )

            if not download_presigned_url or not delete_presigned_url:
                logger.error(
                    f"Failed to generate presigned URLs for chat_id: {chat_id}"
                )
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=str(chat_id),
                        audit_id=None,
                        details={
                            "error": f"Failed to generate presigned URLs for chat_id: {chat_id}",
                            "chat_id": chat_id,
                            "component": "TranscriptionHandler",
                            "method": "send_combined_result_to_queue",
                            "bucket_object_key": bucket_object_key,
                            "bucket_name": self.bucket_name,
                            "result_queue_url": self.result_queue_url,
                            "download_url_generated": bool(download_presigned_url),
                            "delete_url_generated": bool(delete_presigned_url),
                        },
                    )
                )
                raise Exception("Failed to generate presigned URLs")

            # Create message with presigned URLs
            message = TranscribeAndSummarizeResponseMessage(
                message_type=MessageType.TRANSCRIBE_AND_SUMMARIZE_RESPONSE,
                timestamp=int(asyncio.get_event_loop().time() * 1000),
                chat_id=chat_id,
                download_presigned_url=download_presigned_url,
                delete_presigned_url=delete_presigned_url,
            )

            await self.queue_service.send_message(
                queue_url=self.result_queue_url,
                message_body=json.dumps(message.model_dump()),
            )

            logger.info(f"Sent presigned URLs to queue for chat_id: {chat_id}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "message": f"Sent presigned URLs to queue for chat_id: {chat_id}",
                        "chat_id": chat_id,
                        "component": "TranscriptionHandler",
                        "method": "send_combined_result_to_queue",
                        "bucket_object_key": bucket_object_key,
                        "bucket_name": self.bucket_name,
                        "result_queue_url": self.result_queue_url,
                        "download_url_length": len(download_presigned_url),
                        "delete_url_length": len(delete_presigned_url),
                        "transcription_count": len(transcription),
                        "summary_keys": list(summary.keys()) if summary else [],
                    },
                )
            )
            logger.info(f"  - Download URL: {download_presigned_url[:50]}...")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "message": f"  - Download URL: {download_presigned_url[:50]}...",
                        "chat_id": chat_id,
                        "component": "TranscriptionHandler",
                        "method": "send_combined_result_to_queue",
                        "url_type": "download",
                        "url_preview": download_presigned_url[:50] + "...",
                        "bucket_name": self.bucket_name,
                        "result_queue_url": self.result_queue_url,
                    },
                )
            )
            logger.info(f"  - Delete URL: {delete_presigned_url[:50]}...")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "message": f"  - Delete URL: {delete_presigned_url[:50]}...",
                        "chat_id": chat_id,
                        "component": "TranscriptionHandler",
                        "method": "send_combined_result_to_queue",
                        "url_type": "delete",
                        "url_preview": delete_presigned_url[:50] + "...",
                        "bucket_name": self.bucket_name,
                        "result_queue_url": self.result_queue_url,
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
                        "method": "send_combined_result_to_queue",
                        "exception_type": type(e).__name__,
                        "bucket_name": self.bucket_name,
                        "result_queue_url": self.result_queue_url,
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
            error_response = TranscribeAndSummarizeResponseMessage(
                message_type=MessageType.TRANSCRIBE_AND_SUMMARIZE_RESPONSE,
                timestamp=int(asyncio.get_event_loop().time() * 1000),
                chat_id=chat_id,
                transcription=None,
                summary=None,
                error=error_message,
            )

            await self.queue_service.send_message(
                queue_url=self.result_queue_url,
                message_body=json.dumps(error_response.model_dump()),
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
                        "result_queue_url": self.result_queue_url,
                        "bucket_name": self.bucket_name,
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
                        "result_queue_url": self.result_queue_url,
                        "bucket_name": self.bucket_name,
                    },
                )
            )
