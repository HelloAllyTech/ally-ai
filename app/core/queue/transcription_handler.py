import json
import asyncio
import os
from typing import Dict, Any, Optional, Tuple, List

from app.core.queue.sqs_queue_client import SQSQueueClient
from app.core.queue.sqs_queue_service import SQSQueueService
from app.core.queue.message_models import (
    TranscribeAndSummarizeRequestMessage, 
    TranscribeAndSummarizeResultMessage,
    MessageType
)
from app.core.transcriptions.base import BaseTranscriptionService
from app.core.text_generations.openai_text_generation_service import OpenAITextGenerationService
from app.core.s3.s3_service import S3Service
from app.core.config import settings
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
        transcription_service: Optional[BaseTranscriptionService] = None, 
        s3_service: Optional[S3Service] = None, 
        s3_bucket_name: str = None
        ):
        """
        Initialize the transcription handler.
        
        Parameters:
            queue_service (Optional[SQSQueueService]): The queue service to use for sending responses.
                If not provided, a new one will be created.
        """
        self.queue_service = queue_service
        self.request_queue_url = request_queue_url
        self.result_queue_url = result_queue_url
        self.transcription_service = transcription_service
        self.s3_service = s3_service
        self.s3_bucket_name = s3_bucket_name
        
    
    async def process_request(self, message_data: Dict[str, Any]) -> None:
        """
        Process a transcription request message.
        
        Parameters:
            message_data (Dict[str, Any]): The message data from the queue.
        """
        try:
            chat_id = message_data.get('chat_id', 'unknown')
            logger.info(f"Processing transcription request for chat_id: {chat_id}")
            
            # Parse the request message
            request = TranscribeAndSummarizeRequestMessage(**message_data)
            
            # Process the transcription and get results
            success = await self._process_transcription(request)
            
            if success:
                logger.info(f"Transcription processing completed successfully for chat_id: {chat_id}")
            else:
                logger.error(f"Transcription processing failed for chat_id: {chat_id}")
            
        except Exception as e:
            chat_id = message_data.get('chat_id', 'unknown')
            logger.exception(f"Error processing transcription request for chat_id {chat_id}: {str(e)}")

    async def _process_transcription(self, request: TranscribeAndSummarizeRequestMessage) -> bool:
        """
        Process the transcription request using the existing transcription service.
        
        Parameters:
            request (TranscribeAndSummarizeRequestMessage): The transcription request.
            
        Returns:
            bool: True if transcription was successful.
        """
        try:
            # Use the existing transcription service to get results
            chat_id, transcription_data, summary_data = await self.transcription_service.transcribe_audio_from_url(
                audio_url=request.audio_url,
                chat_id=request.chat_id,
                sample_rate=request.sample_rate
            )
            
            # Send the results to S3 and queue
            await self.send_combined_result_to_queue(chat_id, transcription_data, summary_data)
            
            logger.info(f"Transcription completed for chat_id {request.chat_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error in transcription for chat_id {request.chat_id}: {str(e)}")
            return False
    
    async def send_combined_result_to_queue(self, chat_id: int, transcription: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        """
        Upload transcription and summary results to S3 and send the S3 path to the result queue.
        
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
                "summary": summary
            }
            
            # Create the S3 object key
            s3_object_key = f"transcription-results/{chat_id}/result_{chat_id}.json"
            
            # Upload results to S3
            s3_path = await self.s3_service.upload_to_s3(
                bucket_name=self.s3_bucket_name,
                object_key=s3_object_key,
                payload=result_payload
            )
            
            # Create message with S3 path
            message = TranscribeAndSummarizeResultMessage(
                message_type=MessageType.TRANSCRIBE_AND_SUMMARIZE_RESULT,
                timestamp=int(asyncio.get_event_loop().time() * 1000),
                chat_id=chat_id,
                s3_result_path=s3_path
            )
            
            await self.queue_service.send_message(
                queue_url=self.result_queue_url,
                message_body=json.dumps(message.model_dump())
            )
            
            logger.info(f"Sent S3 result path to queue for chat_id: {chat_id} - {s3_path}")
            
        except Exception as e:
            logger.error(f"Error sending combined result to queue for chat_id {chat_id}: {str(e)}")

    
