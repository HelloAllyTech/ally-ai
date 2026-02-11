from typing import Optional

from pydantic import BaseModel, Field


# Shared schema for a chat message
class ChatMessage(BaseModel):
    id: Optional[str] = Field(
        None, description="Unique identifier for the message (required for API requests)"
    )
    role: str = Field(..., description="The role of the sender (e.g., user, assistant)")
    content: str = Field(..., description="The content of the message")
    start_time: Optional[float] = Field(
        None, description="Start time of the message in seconds (relative position)"
    )
    end_time: Optional[float] = Field(
        None, description="End time of the message in seconds (relative position)"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "id": "msg-1",
                "role": "counselor",
                "content": "Hello, how are you?",
                "start_time": 0.5,
                "end_time": 5.7,
            }
        }
