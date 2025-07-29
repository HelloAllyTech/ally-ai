import json
import uuid
import time
from typing import Any, Callable, Dict, Optional, Awaitable

from app.utils.logger import get_logger

logger = get_logger(__name__)


class MessageRouter:
    """
    Simplified router for processing transcription messages.
    Handles a single transcription request processor and routes messages to it.
    """

    def __init__(self):
        """
        Initialize the message router with no processor.
        """
        self._transcription_processor = None

    def register_transcription_processor(self, processor: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """
        Register a processor function for transcription requests.

        Parameters:
            processor (Callable[[Dict[str, Any]], Awaitable[None]]): The processor function that processes transcription requests.
        """
        self._transcription_processor = processor
        logger.info("Registered transcription processor")

    async def route_message(self, message: Dict[str, Any]) -> None:
        """
        Route a message to the transcription processor.

        Parameters:
            message (Dict[str, Any]): The message to route.
        """
        try:
            # Extract the message body
            body = message.get('body', {})
            
            if not isinstance(body, dict):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse message body as JSON: {body}")
                    return
            
            # Check if we have a processor
            if not self._transcription_processor:
                logger.warning("No transcription processor registered")
                return
            
            # Process the message
            logger.info(f"Routing transcription message to processor")
            await self._transcription_processor(body)
                
        except Exception as e:
            logger.exception(f"Error routing message: {str(e)}")
