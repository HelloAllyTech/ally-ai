from typing import List
from pydantic import BaseModel, Field

from app.schemas.common import ChatMessage


class SummaryRequest(BaseModel):
    chat_history: List[ChatMessage] = Field(..., description="List of chat messages")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "chat_history": [
                    {"role": "counselor", "content": "How are you feeling?"},
                    {"role": "client", "content": "Not feeling well"},
                ]
            }
        }


class SummaryResponse(BaseModel):
    summary: str = Field(..., description="The generated summary content")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "summary": "### Client was not feeling well"
            }
        }
