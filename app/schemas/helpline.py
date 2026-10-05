"""Request and response shapes for the text-helpline copilot endpoints.

Exactly the contract in ally-be's docs/text-helpline.md §9.1. Both endpoints are
stateless: the conversation window, the rolling summary and any prompt overrides arrive
on the request, and nothing is stored here.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HelplineChatMessage(BaseModel):
    """One turn of the conversation. `talker` is the person seeking support."""

    role: Literal["talker", "listener"]
    content: str


class HelplineRiskRequest(BaseModel):
    message: str = Field(
        ..., description="The talker's latest message — the one being screened."
    )
    recent: List[HelplineChatMessage] = Field(
        default_factory=list,
        description="The turns before it, oldest first. At most the last 4 are used.",
    )
    language: str = Field("en", description="The language the talker chose.")
    prompts: Optional[Dict[str, Any]] = Field(
        None, description="ally-be prompt overrides, keyed by prompt code."
    )


class HelplineRiskResponse(BaseModel):
    """The classifier's verdict.

    `failed` is reported rather than swallowed: a caller must be able to tell "the
    classifier looked and said no" from "the classifier could not run". The second is a
    degraded safety net that only the keyword screen is holding up.
    """

    is_crisis: bool = False
    confidence: float = 0.0
    # A verbatim substring of the request's `message`, at most 120 characters, or empty.
    signal: str = ""
    subject: Literal["SELF", "OTHER", "UNCLEAR"] = "UNCLEAR"
    failed: bool = False
    provider: str = ""
    model: str = ""


class HelplineTurnRequest(BaseModel):
    messages: List[HelplineChatMessage] = Field(
        ...,
        description=(
            "The conversation so far, oldest first. At most the last 12 are used."
        ),
    )
    rolling_summary: str = ""
    language: str = Field("en", description="The language suggestions are written in.")
    include_nudge: bool = Field(
        False, description="Whether to also return a coaching nudge for the listener."
    )
    risk_level: Literal["NONE", "ELEVATED", "HIGH"] = Field(
        "NONE",
        description=(
            "The highest risk ally-be's screen has flagged in this chat. When not NONE "
            "the first suggestion must ask directly about safety."
        ),
    )
    risk_subject: Literal["SELF", "OTHER", "UNCLEAR", ""] = Field(
        "", description="Who the flagged risk is about, when the classifier said."
    )
    prompts: Optional[Dict[str, Any]] = Field(
        None, description="ally-be prompt overrides, keyed by prompt code."
    )


class HelplineSuggestionOut(BaseModel):
    text: str
    skill_key: str


class HelplineTurnResponse(BaseModel):
    """Draft replies for the LISTENER. Nothing here is ever shown to a talker."""

    stage: Literal["Engage", "Understand", "Support", "Close", ""] = ""
    nudge: str = ""
    suggestions: List[HelplineSuggestionOut] = Field(default_factory=list)
    failed: bool = False
    provider: str = ""
    model: str = ""
