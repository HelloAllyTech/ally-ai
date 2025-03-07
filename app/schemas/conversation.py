from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.common import ChatMessage


class Nudge(BaseModel):
    """Generated Nudge response"""
    nudge: str = Field(..., description="The generated nudge in markdown format.")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "nudge": "### Be empathetic and ask open-ended questions"
            }
        }


class AnalyzeRequest(BaseModel):
    latest_message: str = Field(..., description="The latest message to analyze")
    chat_history: List[ChatMessage] = Field(..., description="Full history of the chat")
    force_nudge: Optional[bool] = Field(
        False,
        description="Optional flag indicating whether to always generate a nudge"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "latest_message": "Can you share what happened?",
                "chat_history": [
                    {"role": "client", "content": "Hello! I am not feeling good"},
                    {"role": "counselor", "content": "Can you share what happened?"}
                ],
                "generate_nudge": False
            }
        }


class AnalyzeResponse(BaseModel):
    nudge: Optional[str] = Field(
        None,
        description="The generated nudge. Will be non-null if `generate_nudge` is True."
    )
    stage: str = Field(..., description="The current stage of the conversation (always present)")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "nudge": "### Be empathetic and ask open-ended questions",
                "stage": "Rapport Building"
            }
        }
