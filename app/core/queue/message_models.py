from enum import Enum
from typing import Optional

from pydantic import BaseModel


class MessageType(str, Enum):
    """
    Enum for message types.
    """

    TRANSCRIPTION_RESULT = "transcription_result"
    TRANSCRIBE_AND_SUMMARIZE_RESPONSE = "transcribe_and_summarize_response"


class BaseQueueMessage(BaseModel):
    """
    Base model for all queue messages.
    """

    message_type: MessageType
    timestamp: int  # Unix timestamp in milliseconds


class TranscriptionResultMessage(BaseQueueMessage):
    """
    Message containing the transcription result.
    """

    message_type: MessageType = MessageType.TRANSCRIPTION_RESULT
    chat_id: int
    segments_text: str


# TODO - Remove after testing ally-ai and ally-core integration
class TranscribeAndSummarizeResponseMessage(BaseQueueMessage):
    """
    Message containing presigned URLs for downloading and deleting transcription
    results.
    Can also contain error information for failed processing.
    """

    message_type: MessageType = MessageType.TRANSCRIBE_AND_SUMMARIZE_RESPONSE
    chat_id: int
    download_presigned_url: Optional[str] = None
    delete_presigned_url: Optional[str] = None
    error: Optional[str] = None
