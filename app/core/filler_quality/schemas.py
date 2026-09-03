"""Schemas for the thinking-filler judge.

A thinking filler is the short back-channel the AI client utters the instant the
learner stops speaking, while its real reply is still being generated (see
ally-ai-learn's `app/core/livekit/`). It is latency masking, so it is easy to
measure how FAST it was and nearly impossible to tell from timing alone whether
it was any GOOD — a filler that arrives instantly but sounds nothing like the
character, or answers the previous turn instead of this one, is a regression
that every latency dashboard will report as an improvement.

This is the rubric that closes that gap, and it follows the same shape as the
drift and language-quality judges rather than inventing its own:

**The LLM annotates; code computes.** The judge emits zero or more findings per
filler and nothing else. No scores, no ratings, no rates — the drift schema's
rule ("never a score, rating or rate") is a platform convention, not a
language-eval quirk, because scalar quality scores from an LLM are poorly
calibrated and average into dashboard numbers nobody can act on. Character fit
and context fit therefore surface as FINDING RATES computed downstream by
ally-be from these rows, exactly as the language judge's dimensions do.

**Most fillers should have no findings.** "Hmm" is a correct, complete filler
and will be the right answer many times a session. A rubric that rewards finding
fault marks a healthy session down.

Derived in code, never asked of the model:
  - ``repeated_within_window`` — a fact about the sequence of played phrases,
    which the judge sees each filler too narrowly to know and would only guess.
  - ``plays_since_last_use`` and ``distinct_phrase_ratio``.

Nothing here is aggregated into a session verdict. Like the language-quality
judge, this service is a stateless transform; ally-be persists and aggregates.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field

# --- Frozen typology (v1). Changing any of these = new PROMPT_VERSION. ------

#: What a finding is about. These are the two quality dimensions the filler
#: rework set out to move (character fit, context fit) plus the safety property
#: that makes a filler dangerous rather than merely poor.
Dimension = Literal["character_fit", "context_fit", "safety"]

FindingCategory = Literal[
    # character_fit — does it sound like THIS character?
    "generic_for_character",  # could belong to any character, on a configured one
    "wrong_register",  # formality/warmth wrong for the character's state
    "persona_break",  # breaks character, or wrong language/script
    # context_fit — does it fit what the learner just said?
    "answers_earlier_turn",  # responds to a previous moment, not this one
    "incongruous_reaction",  # a reaction this turn does not support
    # safety — could the real reply contradict it?
    "committed",  # asserts, answers, agrees, takes a side, names a specific
    "echoes_specific",  # repeats a name/number/place from the partial transcript
]

Severity = Literal["minor", "major", "critical"]

#: Which categories belong to which dimension. Validated in code; an annotation
#: whose category does not belong to its dimension is dropped, never repaired.
DIMENSION_CATEGORIES: Dict[str, Set[str]] = {
    "character_fit": {"generic_for_character", "wrong_register", "persona_break"},
    "context_fit": {"answers_earlier_turn", "incongruous_reaction"},
    "safety": {"committed", "echoes_specific"},
}

#: Findings that only make sense when the character HAS a configured style.
#: Marking a filler generic on a scenario that never specified how the character
#: speaks blames the model for a configuration gap, so these are conditioned out
#: in code the same way the language judge conditions on garbled STT.
STYLE_CONDITIONED_CATEGORIES: Set[str] = {"generic_for_character"}

#: Longest evidence quote kept on a finding. Matches the language judge's cap so
#: both sets of rows stay comparable in storage and in the UI.
MAX_EVIDENCE_CHARS = 240

#: How many recent fillers count as "recent" when deciding whether a phrase
#: repeated. Mirrors the player's own anti-repeat window (counted in PLAYS, not
#: conversational turns, because one turn can play two fillers), so the judge's
#: repeat finding means the same thing as the player's own guard.
DEFAULT_REPEAT_WINDOW_PLAYS = 12


class FillerObservation(BaseModel):
    """One filler the learner actually heard, with the context to judge it.

    Built by the caller (ally-be) from the session transcript and the per-turn
    filler metadata that ally-ai-learn records (``fillerDecision``,
    ``fillerClipSource`` and friends). The judge sees only what a listener
    would: what was said before, what the character said, and what followed.
    """

    turn_index: int
    #: What the learner said on the turn this filler responded to.
    learner_utterance: str = ""
    #: The filler phrase as spoken.
    filler_text: str
    #: The character's real reply, which followed the filler. This is what makes
    #: `committed` decidable: a filler is only unsafe if it presupposes or
    #: contradicts what actually came next.
    reply_text: str = ""
    #: Where the phrase came from — static | seed | exchange | in_turn. Recorded
    #: so a systematic difference between the context-aware in-turn path and the
    #: rest is visible, which is the whole reason that path exists.
    source: Optional[str] = None
    #: hesitation | acknowledgement | reflection | encouragement.
    filler_type: Optional[str] = None


class FillerFinding(BaseModel):
    """One thing wrong with one filler."""

    dimension: Dimension
    category: FindingCategory
    severity: Severity = "major"
    evidence_quote: str = Field(
        default="", description="Shortest verbatim span exhibiting the problem."
    )
    reasoning: str = Field(default="", description="One sentence.")


class FillerJudgment(BaseModel):
    """The LLM's annotation of one filler: usually an empty findings array."""

    turn_index: int
    findings: List[FillerFinding] = Field(default_factory=list)


class JudgeOutput(BaseModel):
    """Structured LLM output: one annotation per observation."""

    per_filler: List[FillerJudgment]


class ProcessedFinding(FillerFinding):
    """A finding after validation, with what code derived about it."""

    #: True when the finding depends on a configured style the scenario never
    #: had. Kept rather than dropped: it is real signal about the scenario, just
    #: not about the model, and collapsing the two loses that distinction.
    conditioned_out: bool = False


class ProcessedFiller(BaseModel):
    """One judged filler after deterministic post-processing."""

    turn_index: int
    filler_text: str
    source: Optional[str] = None
    filler_type: Optional[str] = None
    findings: List[ProcessedFinding] = Field(default_factory=list)
    #: Computed in code: this phrase also played within the recent window.
    repeated_within_window: bool = False
    #: Plays since this phrase last played; None the first time it is heard.
    plays_since_last_use: Optional[int] = None


class FillerJudgmentResult(BaseModel):
    """What the endpoint returns. Rows and counts, never rates — ally-be
    aggregates, exactly as it does for the language judge."""

    per_filler: List[ProcessedFiller] = Field(default_factory=list)
    fillers_judged: int = 0
    #: Findings dropped because their category does not belong to their
    #: dimension, or because they name a turn we never sent. Counted so a rubric
    #: that has started drifting is visible instead of silent.
    dropped_annotations: int = 0
    #: Distinct phrases / total played. A fact, not a rate over findings, and
    #: the one thing a timing dashboard structurally cannot show.
    distinct_phrase_ratio: Optional[float] = None
    repeat_window_plays: int = DEFAULT_REPEAT_WINDOW_PLAYS
