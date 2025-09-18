from enum import Enum

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """
    Enum for message types.
    """

    TRANSCRIBE_AND_SUMMARIZE_REQUEST = "transcribe_and_summarize_request"
    TRANSCRIPTION_RESULT = "transcription_result"


class BaseQueueMessage(BaseModel):
    """
    Base model for all queue messages.
    """

    message_type: MessageType
    timestamp: int


class TranscribeAndSummarizeRequestMessage(BaseQueueMessage):
    """
    Message for requesting audio transcription and summarization.
    """

    message_type: MessageType = MessageType.TRANSCRIBE_AND_SUMMARIZE_REQUEST
    chat_id: int
    audio_url: str
    sample_rate: int = Field(default=8000)


class TranscriptionResultMessage(BaseQueueMessage):
    """
    Message containing the transcription result.
    """

    message_type: MessageType = MessageType.TRANSCRIPTION_RESULT
    chat_id: int
    segments_text: str
