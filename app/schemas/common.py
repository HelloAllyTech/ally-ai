from pydantic import BaseModel, Field
from typing import Optional


# Shared schema for a chat message
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the sender (e.g., user, assistant)")
    content: str = Field(..., description="The content of the message")
    start_time: Optional[int] = Field(None, description="Start time of the message in seconds (timestamp)")
    end_time: Optional[int] = Field(None, description="End time of the message in seconds (timestamp)")

    class Config:
        json_schema_extra = {
            "example": {
                "role": "counselor",
                "content": "Hello, how are you?",
                "start_time": 1640995200,
                "end_time": 1640995205
            }
        }
