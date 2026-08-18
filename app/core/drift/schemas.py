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
        description="STT quality of the counselor (human) utterance this turn replies "
                    "to."
    )
    stt_error_type: SttErrorType = Field(
        description="Sub-type of STT garble, or 'none' if not garbled."
    )
    ai_reply_failure_mode: LlmFailureMode = Field(
        description="How the AI reply failed, or 'none' if clean."
    )
    root_attribution: RootAttribution = Field(
        description="Root cause, considering the prior ~3 turns; 'none' if not a drift "
                    "turn."
    )

    # ---- v2 labels ------------------------------------------------------
    #
    # Added for the Weak Performing Metrics dashboard. Every one is a boolean
    # or a count — never a score, rating or rate. Anything derived (per-100
    # rates, session rollups, over-compliance thresholds) is computed in SQL
    # from these, so re-weighting never requires re-judging 17k turns.
    #
    # Optional with defaults so a v1 row read back through this model still
    # validates, and so a judge that omits one degrades to "not observed"
    # rather than failing the whole session.
    role_inversion: Optional[bool] = Field(
        default=None,
        description=(
            "True if the AI asked the counselor about the COUNSELOR "
            "(their views, feelings, experience) or gave them advice — i.e. "
            "acted as the counselor. A client asking for help ('what should I "
            "do?') is NOT role inversion."
        ),
    )
    offered_solution: Optional[bool] = Field(
        default=None,
        description=(
            "True if the AI proposed a solution or coping plan for its OWN "
            "problem, unprompted, rather than letting the counselor work "
            "toward it."
        ),
    )
    solutions_offered: Optional[int] = Field(
        default=None,
        description=(
            "How many DISTINCT solutions the AI proposed for its own problem "
            "this turn. 0 when none. A real client offers at most one or two; "
            "the acceptable ceiling is a product decision applied downstream, "
            "not a judgement made here."
        ),
    )
    introduced_new_information: Optional[bool] = Field(
        default=None,
        description=(
            "True if this turn added anything the client had not already said "
            "— a new detail, feeling, event or objection. False when it only "
            "restates earlier content, however differently worded."
        ),
    )
    stuck_is_appropriate: Optional[bool] = Field(
        default=None,
        description=(
            "Only meaningful when introduced_new_information is False. True if "
            "holding the same position was CORRECT portrayal given the brief "
            "and what the counselor just did (a resistant client should not "
            "yield to a weak intervention); False if the client should have "
            "moved and did not. Set null when the turn did advance."
        ),
    )
    resistance_briefed: Optional[bool] = Field(
        default=None,
        description=(
            "True if the persona/scenario brief calls for resistance, denial "
            "or reluctance. Judged from the brief, not from this turn — it is "
            "the same answer for every turn in a session, and it is what makes "
            "over-compliance readable as a failure rather than a style."
        ),
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
