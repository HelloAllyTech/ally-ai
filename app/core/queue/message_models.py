from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """
    Enum for message types.
    """
    TRANSCRIBE_AND_SUMMARIZE_REQUEST = "transcribe_and_summarize_request"
    TRANSCRIBE_AND_SUMMARIZE_RESULT = "transcribe_and_summarize_result"


class BaseQueueMessage(BaseModel):
    """
    Base model for all queue messages.
    """
    message_type: MessageType
    timestamp: int  # Unix timestamp in milliseconds


class TranscribeAndSummarizeRequestMessage(BaseQueueMessage):
    """
    Message for requesting audio transcription and summarization.
    """
    message_type: MessageType = MessageType.TRANSCRIBE_AND_SUMMARIZE_REQUEST
    chat_id: int
    presigned_url: str
    sample_rate: int = Field(default=8000)


class TranscribeAndSummarizeResultMessage(BaseQueueMessage):
    """
    Message containing the S3 path to the transcription and summary results.
    """
    message_type: MessageType = MessageType.TRANSCRIBE_AND_SUMMARIZE_RESULT
    chat_id: int
    s3_result_path: str

