from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """
    Enum for message types.
    """

    TRANSCRIPTION_RESULT = "transcription_result"
    TRANSCRIBE_AND_SUMMARIZE_RESPONSE = "transcribe_and_summarize_response"
    TRANSCRIBE_AND_SUMMARIZE_REQUEST = "transcribe_and_summarize_request"


class BaseQueueMessage(BaseModel):
    """
    Base model for all queue messages.
    """

    message_type: MessageType
    timestamp: int  # Unix timestamp in milliseconds
    # End-to-end trace id, minted by ally-core at dispatch. Carried through the
    # AI pipeline logs and echoed back on the result callback so a single chat's
    # journey can be grepped across both services. Optional for backward compat
    # with in-flight messages that predate this field.
    correlation_id: Optional[str] = None

class TranscribeAndSummarizeRequestMessage(BaseQueueMessage):
    """
    Message for requesting audio transcription and summarization.
    """

    message_type: MessageType = MessageType.TRANSCRIBE_AND_SUMMARIZE_REQUEST
    chat_id: int
    audio_url: str
    sample_rate: int = Field(default=8000)
    mode: Optional[str] = None
    # Mobile uploads headerless linear16 (s16le) PCM. ffprobe can't identify
    # raw PCM, so this flag tells the converter to decode it directly as raw
    # rather than relying on container detection.
    is_linear16_encoded: Optional[bool] = None

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
