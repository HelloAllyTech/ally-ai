"""Schemas for the language-quality judge (see language-eval-judge-schema.md).

The judge LLM emits ONLY per-turn error annotations (``JudgeOutput``). The
layer of each error is derived in code from its dimension (fixed mapping),
category↔dimension consistency is validated in code (invalid annotations are
dropped, never guessed), and STT conditioning (``conditioned_out``) is applied
in code — none of that is asked of the LLM. Weighted error rates are computed
downstream (ally-be) from the persisted rows; this service never aggregates.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Set

from pydantic import BaseModel, Field

# --- Frozen typology (v1). Changing any of these = new PROMPT_VERSION. ------

Dimension = Literal[
    "understanding",
    "adequacy",
    "fluency",
    "coherence",
    "register",
    "dialect_lexicon",
    "colloquialness",
    "persona_social",
    "codeswitch",
]

ErrorCategory = Literal[
    # understanding
    "misinterpreted_intent",
    "ignored_context",
    # adequacy
    "off_topic",
    "hallucination",
    "omission",
    # fluency
    "grammar",
    "script_error",
    "disfluency",
    "truncation",
    # coherence
    "contradiction",
    "non_sequitur",
    # register
    "too_formal_diglossia",
    "too_casual",
    # dialect_lexicon
    "wrong_regional_variety",
    # v2: the two lexical failures partners actually report. The dimension was
    # scoped to regional VARIETY only, which is a different question from "is
    # this a real word" and "does it mean what the sentence needs" — so the
    # complaints had nowhere to land and the dimension fired on almost nothing.
    "nonexistent_word",
    "wrong_sense",
    # colloquialness
    "literal_translation_stilt",
    # persona_social
    "too_blunt",
    "persona_break",
    # codeswitch
    "foreign_token_leak",
    "unnatural_switch",
]

Severity = Literal["minor", "major", "critical"]
GarbleLevel = Literal["none", "partial", "severe"]  # same semantics as drift
IsolationBasis = Literal[
    "input_clean",
    "input_garbled",
    "persona_specified",
    "persona_unspecified",
    "pattern_systemic",
]
Layer = Literal["comprehension", "content", "appropriateness"]

# Layer is derived from dimension in code — never asked of the LLM.
DIMENSION_LAYER: Dict[str, str] = {
    "understanding": "comprehension",
    "adequacy": "content",
    "fluency": "content",
    "coherence": "content",
    "register": "appropriateness",
    "dialect_lexicon": "appropriateness",
    "colloquialness": "appropriateness",
    "persona_social": "appropriateness",
    "codeswitch": "appropriateness",
}

# Valid categories per dimension; annotations violating this are dropped.
DIMENSION_CATEGORIES: Dict[str, Set[str]] = {
    "understanding": {"misinterpreted_intent", "ignored_context"},
    "adequacy": {"off_topic", "hallucination", "omission"},
    "fluency": {"grammar", "script_error", "disfluency", "truncation"},
    "coherence": {"contradiction", "non_sequitur"},
    "register": {"too_formal_diglossia", "too_casual"},
    "dialect_lexicon": {
        "wrong_regional_variety",
        "nonexistent_word",
        "wrong_sense",
    },
    "colloquialness": {"literal_translation_stilt"},
    "persona_social": {"too_blunt", "persona_break"},
    "codeswitch": {"foreign_token_leak", "unnatural_switch"},
}

# Weights are aggregation constants (weighted error rate / 100 turns computed
# in ally-be); kept here as the single normative definition.
SEVERITY_WEIGHT: Dict[str, int] = {"minor": 1, "major": 5, "critical": 10}

# Dimensions whose errors are conditioned out on garbled counselor input
# (PRD conditioning rule: mishearing must not be billed to the LLM).
CONDITIONED_DIMENSIONS: Set[str] = {"understanding", "adequacy"}

# Guardrail for evidence quotes (LLM is instructed to stay short; we clamp).
MAX_EVIDENCE_CHARS = 200


class ErrorAnnotation(BaseModel):
    """One language error on one AI-client turn. The LLM's output unit."""

    dimension: Dimension
    category: ErrorCategory
    severity: Severity
    evidence_quote: str = Field(
        description="Shortest span exhibiting the error, verbatim, original script."
    )
    isolation_basis: IsolationBasis
    reasoning: str = Field(description="One sentence justifying the annotation.")


class TurnJudgment(BaseModel):
    """One AI-client turn: STT-conditioning flag + its error annotations."""

    turn_index: int = Field(description="Index of the AI-client turn being judged.")
    input_garbled: GarbleLevel = Field(
        description="STT quality of the counselor utterance this turn replies to."
    )
    errors: List[ErrorAnnotation] = Field(default_factory=list)


class JudgeOutput(BaseModel):
    """Exactly what the judge LLM returns (per-turn array, wrapped)."""

    per_turn: List[TurnJudgment]


class ProcessedError(ErrorAnnotation):
    """An annotation after code-side post-processing (layer + conditioning)."""

    layer: Layer
    conditioned_out: bool = Field(
        description=(
            "True when the error is on a conditioned dimension and the turn's "
            "input was garbled — excluded from that dimension's error rate."
        )
    )


class ProcessedTurn(BaseModel):
    turn_index: int
    input_garbled: GarbleLevel
    errors: List[ProcessedError]


class LanguageJudgmentResult(BaseModel):
    per_turn: List[ProcessedTurn]
    turns_judged: int
    # Annotations the LLM emitted with an invalid category-for-dimension pair;
    # dropped in code (never guessed). Non-zero values are a rubric-tuning signal.
    dropped_annotations: int = 0
