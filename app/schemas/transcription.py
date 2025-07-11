from typing import List
from pydantic import BaseModel, Field


class TranscribeAndSummarizeRequest(BaseModel):
    """Request model for audio transcription and summarization."""
    presigned_url: str = Field(
        ..., 
        description="URL containing the audio file"
    )
    chat_id: int = Field(
        ..., 
        description="Chat ID for the transcription session"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "presigned_url": "https://example-bucket.s3.amazonaws.com/audio-file.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=...",
                "chat_id": 12345
            }
        }



class TranscribeAndSummarizeResponse(BaseModel):
    """Response model for audio transcription and summarization."""
    success: bool = Field(
        True, 
        description="Indicates if the transcription and summarization was successful"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "success": True
            }
        } 