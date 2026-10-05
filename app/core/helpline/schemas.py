"""Structured output shapes for the text-helpline copilot's two LLM calls.

These are passed as the provider's response schema (Gemini `response_schema`, Anthropic
forced-tool `input_schema`, OpenAI `json_schema`), so keep them flat and avoid
`Optional` — an absent value comes back as the empty default rather than null, which
keeps the parsed object total.

The closed vocabularies (`subject`, `stage`, `skill_key`) are plain strings that carry
their allowed values as a JSON-schema `enum`, rather than Python Enums. The provider is
still steered to the allowed set, but a value that drifts (wrong case, an unlisted key)
does not fail validation of the WHOLE response. That matters most for the risk
classifier: a spelling slip in `subject` must not cost a crisis verdict. The service
normalises each value and decides what to do with one that is still out of vocabulary.
"""

from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, Field

SUBJECTS: Tuple[str, ...] = ("SELF", "OTHER", "UNCLEAR")

STAGES: Tuple[str, ...] = ("Engage", "Understand", "Support", "Close")

# The helping-skills rubric keys a suggestion can be tagged with. `non-verbal` is left
# out on purpose: it cannot be expressed in text. Must match the contract in ally-be's
# docs/text-helpline.md §9.1.
SKILL_KEYS: Tuple[str, ...] = (
    "rapport",
    "confidentiality",
    "feelings",
    "empathy",
    "harm",
    "functioning",
    "explanation",
    "family",
    "goals",
    "hope",
    "coping",
    "psychoeducation",
    "feedback",
    "verbal",
)


class HelplineRiskVerdict(BaseModel):
    """The risk classifier's structured output.

    Deliberately biased towards the false positive, as the knowledge agent's crisis
    verdict is, and for the same reason: a low `confidence` alongside
    ``is_crisis=True`` is a correct and expected combination. ally-be turns the pair
    into a level (HIGH above its configured confidence, ELEVATED below); the
    classifier itself never second-guesses its own verdict with a threshold.
    """

    is_crisis: bool = False
    # How clear-cut the verdict is, 0 to 1. Calibration guidance lives in the prompt.
    confidence: float = 0.0
    # The shortest phrase from the talker's message that drove the verdict, copied
    # verbatim. The service re-checks that it really is a substring of the message.
    signal: str = ""
    # Whose risk it is: the talker's own (SELF), someone they are describing (OTHER —
    # "my brother is thinking of…"), or can't tell (UNCLEAR).
    subject: str = Field("UNCLEAR", json_schema_extra={"enum": list(SUBJECTS)})


class HelplineSuggestion(BaseModel):
    """One draft reply for the listener to edit and send."""

    text: str = ""
    skill_key: str = Field("", json_schema_extra={"enum": list(SKILL_KEYS)})


class HelplineCopilotTurn(BaseModel):
    """The copilot turn's structured output."""

    stage: str = Field("", json_schema_extra={"enum": list(STAGES)})
    # One coaching tip for the LISTENER, or empty. Never a message to send.
    nudge: str = ""
    suggestions: List[HelplineSuggestion] = Field(default_factory=list)
