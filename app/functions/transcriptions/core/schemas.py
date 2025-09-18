"""
Schemas for the Lambda transcription function.
"""

from typing import Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """
    Represents a chat message with speaker information and content.
    """

    speaker: str
    content: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
