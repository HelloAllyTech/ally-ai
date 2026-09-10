"""Schemas for the RAG-quality judge.

Asks two questions of one retrieval: **was each passage actually relevant to
the query**, and **was the set as a whole enough to work from**.

It exists because the numbers governing retrieval were set by argument and lost
to measurement. The character corpus's similarity floor was 0.5, then 0.45,
then 0.35 — and what settled it was one production query returning nothing
against a document whose section heading answered it, while an equivalent
phrasing of the same question cleared the same floor. A cosine similarity is
not calibrated across queries, so a threshold chosen without labels is a guess
with a decimal point.

The judge emits ONLY per-passage labels and one per-retrieval label. Precision,
the score distribution per label, and the precision/recall curve over candidate
floors are computed downstream in SQL — the same division of labour as the
drift, language and feedback-groundedness judges, and for the same reason: a
threshold can be re-cut without re-judging the corpus.

WHAT IT CANNOT SEE, stated because the gap is easy to forget when reading the
rates. Only passages that ally-ai actually returned are logged, and those are
by definition already above the floor. So these labels measure PRECISION of
what got through — never the recall of what the floor rejected. Measuring that
needs the same queries re-run at a lower floor, which the retrieval preview's
floor control does by hand and a calibration sweep would do in bulk.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

# How a passage relates to the query that retrieved it.
#
#   relevant    — bears directly on what was asked; a person writing from this
#                 query would use it
#   tangential  — same subject area, does not answer the question. This is the
#                 label that matters most: it is what a floor tuned on vibes
#                 lets through, and what makes a draft feel specific while
#                 being about something else
#   irrelevant  — not about the question at all
PassageRelevance = Literal["relevant", "tangential", "irrelevant"]

# Whether the returned set was enough to act on. The LLM-derived stand-in for
# retrieval_accuracy@k, which otherwise needs a hand-labelled expected set —
# and hand-labelling is not how judges are validated on this platform.
#
#   sufficient     — the set answers the query
#   partial        — something useful, but a gap a writer would notice
#   nothing_useful — nothing in the set helps, whatever the scores said
RetrievalSufficiency = Literal["sufficient", "partial", "nothing_useful"]


class PassageJudgment(BaseModel):
    """One retrieved passage, judged against the query that retrieved it."""

    chunk_id: str = Field(
        description="The passage's id, exactly as supplied. Used to join the label back."
    )
    relevance: PassageRelevance = Field(
        description="How directly the passage bears on the query."
    )
    superficial_match: bool = Field(
        description=(
            "True when the passage looks retrievable only because it SHARES "
            "VOCABULARY with the query rather than meaning — the same words, a "
            "different subject. Recorded separately from relevance because it "
            "names the mechanism: this is what embedding similarity is worst "
            "at, and a high-scoring superficial match is the single most "
            "misleading thing retrieval can return."
        )
    )
    reasoning: str = Field(description="One sentence justifying the label.")


class RetrievalJudgment(BaseModel):
    """The retrieval as a whole."""

    sufficiency: RetrievalSufficiency = Field(
        description="Whether the returned set was enough to work from."
    )
    missing: str = Field(
        default="",
        description=(
            "When sufficiency is not `sufficient`: what a writer would still "
            "need, in one phrase. Empty otherwise. This is the only field that "
            "points at a CORPUS gap rather than a retrieval fault, so it is "
            "what tells a curator which document to go and find."
        ),
    )


class RagQualityOutput(BaseModel):
    """Exactly what the judge LLM returns."""

    passages: List[PassageJudgment]
    retrieval: RetrievalJudgment
