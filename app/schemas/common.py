from pydantic import BaseModel, Field


# Shared schema for a chat message
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the sender (e.g., user, assistant)")
    content: str = Field(..., description="The content of the message")

    class Config:
        schema_extra = {
            "example": {
                "role": "counselor",
                "content": "Hello, how are you?"
            }
        }
