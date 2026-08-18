"""Schemas for the feedback-groundedness judge.

Asks one question of each feedback claim: **is this true about the transcript?**
Post-session feedback is the only output the learner is graded by, and one
counsellor described being left doubting whether she was a good therapist after
receiving feedback that was wrong. Nothing measured whether it was.

The judge emits ONLY booleans and an enum choice per claim. Support rates,
false-negative rates and per-100 figures are computed downstream in SQL, so a
threshold can be re-cut without re-judging the corpus — the same division of
labour as the drift and language judges.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Which list the claim came from. Kept explicit rather than inferred, because
# the two fail in opposite directions and must be reported separately: an
# unearned compliment is a different product problem from valid work marked
# wrong, and only the second is what learners described as harmful.
ClaimKind = Literal["positive", "improvement"]

# How the claim relates to the transcript.
#
#   supported     — the transcript shows what the claim says
#   unsupported   — the transcript does not show it (nothing corroborates)
#   contradicted  — the transcript shows the OPPOSITE. For an `improvement`
#                   this is the false negative that stings: the learner is told
#                   they failed to do something they demonstrably did
#   misattributed — the behaviour occurred, but the claim pins it to the wrong
#                   turn or the wrong speaker
Verdict = Literal["supported", "unsupported", "contradicted", "misattributed"]


class ClaimJudgment(BaseModel):
    """One feedback claim, judged against the session transcript."""

    claim_index: int = Field(
        description="Position of the claim within its list, 0-based."
    )
    kind: ClaimKind = Field(description="Which list the claim came from.")
    verdict: Verdict = Field(
        description="How the claim stands up against the transcript."
    )
    quotes_transcript: bool = Field(
        description=(
            "True if the claim cites specific words as having been said "
            "(quoted or closely paraphrased), rather than only characterising "
            "the counsellor's behaviour."
        )
    )
    quote_is_accurate: Optional[bool] = Field(
        default=None,
        description=(
            "Only when quotes_transcript is true: does the cited wording "
            "actually appear in the transcript? Null otherwise. A fabricated "
            "citation is the most concrete groundedness failure there is."
        ),
    )
    reasoning: str = Field(description="One sentence justifying the verdict.")


class GroundednessOutput(BaseModel):
    """Exactly what the judge LLM returns."""

    claims: List[ClaimJudgment]
