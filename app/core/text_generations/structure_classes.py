from pydantic import BaseModel, Field


class NudgeOutput(BaseModel):
    """Generated Nudge response"""
    nudge: str = Field(..., description="The generated nudge in markdown format.")
