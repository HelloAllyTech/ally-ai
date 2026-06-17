"""Schemas for the conversation drift judge (see drift-metrics-spec.md).

The judge LLM emits ONLY the per-turn array (``JudgeOutput``). The session
rollup (drifted? / first-drift turn / attribution mix) is computed
deterministically in code from those per-turn rows — never asked of the LLM —
so the headline numbers are reproducible and don't depend on the model's
arithmetic.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Anchored ordinal — calibrate against hand labels on the LEVEL, not a number.
CoherenceLevel = Literal[
    "fully_coherent",
    "minor_disfluency",
    "degrading",
    "mostly_incoherent",
    "gibberish",
]
TopicLabel = Literal["on_topic", "tangent", "off_topic", "gibberish"]
GarbleLevel = Literal["none", "partial", "severe"]
SttErrorType = Literal[
    "entity_swap",
    "phonetic_garble",
    "wrong_language",
    "number_format",
    "code_mix_fail",
    "truncation",
    "none",
]
LlmFailureMode = Literal[
    "hallucination",
    "context_lockin",
    "wrong_language_reply",
    "repetition",
    "role_slip",
    "wrong_intent",
    "none",
]
RootAttribution = Literal[
    "stt_direct",
    "stt_cascade",
    "llm_direct",
    "context_lockin",
    "none",
]

# Coherence levels at or below this rank count as a drift turn (when not
# in-character). fully_coherent=4 … gibberish=0; "degrading" = 2.
COHERENCE_RANK = {
    "fully_coherent": 4,
    "minor_disfluency": 3,
    "degrading": 2,
    "mostly_incoherent": 1,
    "gibberish": 0,
}
COHERENCE_DRIFT_CUTOFF = 2  # <= degrading


class PerTurnJudgment(BaseModel):
    """One AI-client turn, judged. This is the LLM's structured output unit."""

    turn_index: int = Field(description="Index of the AI-client turn being judged.")
    coherence: CoherenceLevel
    topic_label: TopicLabel
    in_character: bool = Field(
        description="True if odd output is realistic in-character distress, not drift."
    )
    counselor_utterance_garbled: GarbleLevel = Field(
        description="STT quality of the counselor (human) utterance this turn replies to."
    )
    stt_error_type: SttErrorType = Field(
        description="Sub-type of STT garble, or 'none' if not garbled."
    )
    ai_reply_failure_mode: LlmFailureMode = Field(
        description="How the AI reply failed, or 'none' if clean."
    )
    root_attribution: RootAttribution = Field(
        description="Root cause, considering the prior ~3 turns; 'none' if not a drift turn."
    )
    reasoning: str = Field(description="One sentence justifying the labels.")


class JudgeOutput(BaseModel):
    """Exactly what the judge LLM returns (per-turn array, wrapped)."""

    per_turn: List[PerTurnJudgment]


class AttributionMix(BaseModel):
    stt_direct: int = 0
    stt_cascade: int = 0
    llm_direct: int = 0
    context_lockin: int = 0


class SessionRollup(BaseModel):
    """Computed in code from per-turn rows — NOT emitted by the LLM."""

    drifted: bool
    first_drift_turn: Optional[int] = None
    attribution_mix: AttributionMix


class DriftJudgmentResult(BaseModel):
    per_turn: List[PerTurnJudgment]
    session: SessionRollup
