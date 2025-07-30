from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import httpx
import asyncio
import random

from app.schemas.common import ChatMessage
from app.core.text_generations.base import BaseTextGenerationService
from app.utils.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)


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
        audio_url: str,
        chat_id: int,
        sample_rate: int = 8000
    ) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Transcribe audio from URL and generate a summary.
        
        Args:
            audio_url (str): URL containing the audio file
            chat_id (int): Chat ID for the transcription session
            sample_rate (int): Expected sample rate of the audio (default: 8000)
            
        Returns:
            Tuple[int, List[Dict[str, Any]], Dict[str, Any]]: (chat_id, transcription_data, summary_data)
            
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