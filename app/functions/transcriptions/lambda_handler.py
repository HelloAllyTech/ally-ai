"""
AWS Lambda handler for transcription processing.

This Lambda function processes SQS messages containing transcription requests,
performs the transcription and summarization, and sends results back to the result queue.
"""

import json
import os
import time
import asyncio
from typing import Dict, Any
import boto3

import openai
from openai import OpenAI
from core.message_models import (
    TranscribeAndSummarizeRequestMessage,
    TranscriptionResultMessage,
    MessageType
)
from utils.logger import get_logger

from core.config import settings
from services import OpenAITranscriptionService, DeepgramTranscriptionService

logger = get_logger(__name__)


def create_transcription_service(provider=None):
    """
    Create a transcription service based on the specified provider.
    
    Args:
        provider (str, optional): Provider to use ('openai' or 'deepgram').
            If None, will use settings.TRANSCRIPTION_PROVIDER or default to 'openai'.
            
    Returns:
        The transcription service instance
        
    Raises:
        ValueError: If provider is not supported or required API keys are missing
    """
    # Determine provider
    if provider is None:
        provider = settings.TRANSCRIPTION_PROVIDER.lower()
    
    logger.info(f"Creating transcription service with provider: {provider}")
    
    if provider == 'openai':
        # Check for OpenAI API key
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required in settings for OpenAI provider")
        
        return OpenAITranscriptionService()
        
    elif provider == 'deepgram':
        # Check for Deepgram API key
        if not settings.DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY is required in settings for Deepgram provider")
        
        # Also need OpenAI for summarization
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required in settings for summarization")
        
        return DeepgramTranscriptionService()
        
    else:
        raise ValueError(f"Unsupported transcription provider: {provider}. Supported providers: 'openai', 'deepgram'")

# Initialize services (these will be initialized once when the Lambda container starts)
transcription_service = None

def initialize_services():
    """Initialize all services needed for transcription processing."""
    global transcription_service
    
    if transcription_service is None:
        logger.info("Initializing transcription services...")
        
        transcription_service = create_transcription_service()
        
        logger.info("Transcription services initialized successfully")


async def process_transcription_request(request_data: Dict[str, Any], receipt_handle: str) -> Dict[str, Any]:
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
        
        # Process transcription
        chat_id, segments_text = await transcription_service.transcribe_audio_from_url(
            audio_url=request.audio_url,
            chat_id=request.chat_id,
            sample_rate=request.sample_rate
        )
        
        # Create result message
        result_message = TranscriptionResultMessage(
            chat_id=chat_id,
            segments_text=segments_text,
            timestamp=int(time.time() * 1000)
        )
        
        sqs_client = boto3.client('sqs')
        # Send the result message to the result queue
        await asyncio.to_thread(
            sqs_client.send_message,
            QueueUrl=settings.TRANSCRIPTION_RESULTS_QUEUE_URL,
            MessageBody=json.dumps(result_message.model_dump())
        )

        # Delete the message from the queue
        delete_response = await asyncio.to_thread(
            sqs_client.delete_message,
            QueueUrl=settings.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL,
            ReceiptHandle=receipt_handle
        )

        # Check if deletion was successful
        if delete_response.get('ResponseMetadata', {}).get('HTTPStatusCode') == 200:
            logger.info(f"Successfully deleted message from requests queue for chat_id: {chat_id}")
        else:
            logger.warning(f"Failed to delete message from requests queue. Response: {delete_response}")
        
        logger.info(f"Successfully processed chat_id: {chat_id}")
        
        return {
            "status": "success",
            "chat_id": chat_id,
            "segments_text": segments_text
        }
        
    except Exception as e:
        logger.exception(f"Error processing transcription request: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "chat_id": request_data.get('chat_id', 'unknown')
        }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.
    
    Args:
        event: SQS event containing a single message
        context: Lambda context
        
    Returns:
        Dict containing the processing result
    """
    logger.info(f"Lambda function started. Event: {json.dumps(event)}")
    # Initialize services
    initialize_services()
    
    try:
        # Get the single message from the event
        records = event.get('Records', [])
        if not records:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No messages found in event"})
            }
        
        # Process the single message
        record = records[0]  # Each Lambda processes one message
        message_body = json.loads(record['body'])
        receipt_handle = record['receiptHandle']  # Get receipt handle for deletion
        logger.info(f"Processing message: {message_body.get('chat_id', 'unknown')}")
        
        # Process the transcription request
        result = asyncio.run(process_transcription_request(message_body, receipt_handle))
        
        logger.info(f"Message processed successfully: {result}")
        
        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }
        
    except Exception as e:
        logger.exception(f"Lambda function error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status": "error",
                "error": str(e)
            })
        }
