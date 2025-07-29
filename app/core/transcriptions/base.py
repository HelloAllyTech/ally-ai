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