import json
import uuid
import time
from typing import Any, Callable, Dict, Optional, Awaitable

from app.utils.logger import get_logger

logger = get_logger(__name__)


class MessageRouter:
    """
    Simplified router for processing transcription messages.
    Handles a single transcription request handler and routes messages to it.
    """

    def __init__(self):
        """
        Initialize the message router with no handler.
        """
        self._transcription_handler = None

    def register_transcription_handler(self, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """
        Register a handler for transcription requests.

        Parameters:
            handler (Callable[[Dict[str, Any]], Awaitable[None]]): The handler function that processes transcription requests.
        """
        self._transcription_handler = handler
        logger.info("Registered transcription handler")

    async def route_message(self, message: Dict[str, Any]) -> None:
        """
        Route a message to the transcription handler.

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
            
            # Check if we have a handler
            if not self._transcription_handler:
                logger.warning("No transcription handler registered")
                return
            
            # Process the message
            logger.info(f"Routing transcription message to handler")
            await self._transcription_handler(body)
                
        except Exception as e:
            logger.exception(f"Error routing message: {str(e)}")

    async def create_response(self, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a response message based on a request message.

        Parameters:
            request_data (Dict[str, Any]): The original request data.
            response_data (Dict[str, Any]): The response data to include.

        Returns:
            Dict[str, Any]: The formatted response message.
        """
        # Generate a message ID and timestamp
        message_id = str(uuid.uuid4())
        timestamp = int(time.time() * 1000)  # Current time in milliseconds
        
        # Create the response with correlation to the request
        response = {
            "message_id": message_id,
            "timestamp": timestamp,
            "correlation_id": request_data.get("message_id"),
            **response_data
        }
        
        return response
