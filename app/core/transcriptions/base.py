from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import httpx
import asyncio
import random
from dataclasses import dataclass

from app.schemas.common import ChatMessage
from app.core.text_generations.base import BaseTextGenerationService
from app.utils.logger import get_logger
from app.core.config import settings
from app.exceptions.custom_exceptions import CoreAPIFailedException

logger = get_logger(__name__)


@dataclass
class TranscriptPayload:
    """
    Payload for sending transcript data to core service.
    """
    chat_id: int
    messages: List[ChatMessage]
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {
                "messages": [msg.model_dump() for msg in self.messages]
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the payload to a dictionary for API calls.
        """
        return {
            "chat_id": self.chat_id,
            **self.data
        }


@dataclass
class SummaryPayload:
    """
    Payload for sending summary data to core service.
    """
    chat_id: int
    summary: Any
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {
                "summary": self.summary.model_dump() if hasattr(self.summary, 'model_dump') else self.summary
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the payload to a dictionary for API calls.
        """
        return {
            "chat_id": self.chat_id,
            **self.data
        }


class BaseTranscriptionService[ModelT](ABC):
    """
    Abstract base class for transcription services.
    
    This class defines the interface that all transcription services must implement,
    providing a common contract for audio transcription functionality.
    """
    
    def __init__(self, model: ModelT, text_generation_service: BaseTextGenerationService) -> None:
        """
        Initialize the base transcription service with a model and text generation service.
        
        Parameters:
            model (ModelT): The model to use for transcription.
            text_generation_service (BaseTextGenerationService): The text generation service for summary generation.
        """
        self.model = model
        self.text_generation_service = text_generation_service

    @abstractmethod
    async def transcribe_audio_from_url(
        self, 
        presigned_url: str,
        chat_id: int
    ) -> bool:
        """
        Transcribe audio from URL and generate a summary.
        
        Args:
            presigned_url (str): URL containing the audio file
            chat_id (int): Chat ID for the transcription session
            
        Returns:
            bool: True if transcription and summarization was successful
            
        Raises:
            Exception: If transcription fails
        """
        pass

    async def _send_to_core_with_retry(
        self, 
        endpoint: str, 
        payload: TranscriptPayload | SummaryPayload, 
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 30.0
    ) -> bool:
        """
        Generic function to send data to core service with exponential backoff retry.
        
        Args:
            endpoint (str): The API endpoint path (e.g., '/api/v1/chats/ai/transcript')
            payload (TranscriptPayload | SummaryPayload): The payload object to send
            max_retries (int): Maximum number of retry attempts (default: 3)
            base_delay (float): Base delay in seconds for exponential backoff (default: 1.0)
            max_delay (float): Maximum delay in seconds (default: 60.0)
            timeout (float): Request timeout in seconds (default: 30.0)
            
        Returns:
            bool: True if API call was successful
        """
        last_exception: Optional[Exception] = None
        
        for attempt in range(max_retries + 1):
            try:
                # Make API call to core service
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{settings.CORE_SERVICE_ENDPOINT}{endpoint}",
                        json=payload.to_dict(),
                        headers={
                            "x-api-key": settings.CORE_API_KEY,
                            "Content-Type": "application/json"
                        }
                    )
                    
                    # Check response status and raise custom exception if needed
                    if response.status_code >= 400:
                        response_body = ""
                        try:
                            response_body = response.text
                        except:
                            pass
                        
                        raise CoreAPIFailedException(
                            message=f"Core API returned error status",
                            status_code=response.status_code,
                            endpoint=endpoint,
                            response_status=response.status_code,
                            response_body=response_body
                        )
                    
                logger.info(f"Data sent to core successfully: {endpoint}")
                return True
                
            except CoreAPIFailedException as e:
                # Don't retry on client errors (4xx), only on server errors (5xx)
                if 400 <= e.response_status < 500:
                    logger.error(f"Client error on {endpoint}: {e.response_status} - {str(e)}")
                    return False
                else:
                    last_exception = e
                    logger.warning(f"Server error on attempt {attempt + 1}/{max_retries + 1} for {endpoint}: {e.response_status} - {str(e)}")
                    
            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries + 1} for {endpoint}: {str(e)}")
                
            except httpx.ConnectError as e:
                last_exception = e
                logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries + 1} for {endpoint}: {str(e)}")
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Unexpected error on attempt {attempt + 1}/{max_retries + 1} for {endpoint}: {str(e)}")
            
            # If this was the last attempt, don't wait
            if attempt < max_retries:
                # Calculate delay with exponential backoff and jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, 0.1 * delay)  # 10% jitter
                total_delay = delay + jitter
                
                logger.info(f"Retrying {endpoint} in {total_delay:.2f} seconds...")
                await asyncio.sleep(total_delay)
        
        # All retries exhausted
        logger.error(f"Failed to send data to core {endpoint} after {max_retries + 1} attempts. Last error: {str(last_exception)}")
        return False

    async def _send_to_core(self, endpoint: str, payload: TranscriptPayload | SummaryPayload) -> bool:
        """
        Generic function to send data to core service.
        
        Args:
            endpoint (str): The API endpoint path (e.g., '/api/v1/chats/ai/transcript')
            payload (TranscriptPayload | SummaryPayload): The payload object to send
            
        Returns:
            bool: True if API call was successful
        """
        return await self._send_to_core_with_retry(endpoint, payload)

    async def _send_transcript_to_core(self, chat_id: int, messages: List[ChatMessage]) -> bool:
        """
        Send transcript data to core service.
        
        Args:
            chat_id (int): Chat ID for the session
            messages: List of ChatMessage objects
            
        Returns:
            bool: True if API call was successful
        """
        payload = TranscriptPayload(chat_id=chat_id, messages=messages)
        return await self._send_to_core("/api/v1/chats/ai/transcript", payload)

    async def _send_summary_to_core(self, chat_id: int, summary) -> bool:
        """
        Send summary data to core service.
        
        Args:
            chat_id (int): Chat ID for the session
            summary: Summary object to send
            
        Returns:
            bool: True if API call was successful
        """
        payload = SummaryPayload(chat_id=chat_id, summary=summary)
        return await self._send_to_core("/api/v1/chats/ai/summary", payload)

    async def _generate_summary(self, messages: List[ChatMessage], chat_id: int):
        """
        Generate summary from messages.
        
        Args:
            messages: List of ChatMessage objects
            chat_id (int): Chat ID for the session
            
        Returns:
            Summary object from the text generation service
        """
        try:
            # Generate summary using the text generation service
            summary = await self.text_generation_service.generate_summary_notes(
                chat_history=messages,
                keys=None  # Generate full summary
            )
            logger.info(f"Summary generated successfully for chat_id: {chat_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary for chat_id {chat_id}: {str(e)}")
            raise Exception(f"Summary generation failed: {str(e)}") 