"""Schemas for the recall-quality judge.

Labels only, like every other judge here: rates are computed downstream in SQL so a definition
can be re-cut without re-judging a month of turns.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

#: What the ranking did with the pool it had.
#:
#: `no_demand` is the label that makes the others readable. Most turns of a counselling
#: conversation do not call for a particular piece of backstory — "mm-hmm", "go on", a
#: greeting — and without somewhere to put those, every one of them would land in a bucket
#: that reads as a ranking failure, and the rate would be dominated by turns where nothing
#: was needed.
#:
#: `nothing_apt` and `missed_better` separate the two fixes. Nothing apt in the whole pool is
#: a CORPUS problem: the character profile lacks the material, and no weighting recovers it.
#: A better fact sitting in the passed-over list is a RANKING problem: the material was there
#: and the scoring put it below the cap.
RecallVerdict = Literal["well_chosen", "missed_better", "nothing_apt", "no_demand"]


class RecallJudgment(BaseModel):
    """One turn's verdict."""

    verdict: RecallVerdict = Field(
        description="What the ranking did with the pool available on this turn"
    )
    better_fact: Optional[str] = Field(
        default=None,
        description=(
            "On `missed_better` only: the passed-over fact that should have been recalled "
            "instead, quoted from the candidates given. Null for every other verdict."
        ),
    )
    unused_selected: List[str] = Field(
        default_factory=list,
        description=(
            "Selected facts the turn gave no occasion for. Not a failure on its own — five "
            "are always selected and a turn rarely calls for five — but a persistently high "
            "count means the cap is larger than the conversation can use."
        ),
    )
    reasoning: str = Field(
        default="",
        description="One or two sentences, citing what in the turn called for what",
    )


class RecallQualityOutput(BaseModel):
    """What the model returns for one turn."""

    judgment: RecallJudgment
