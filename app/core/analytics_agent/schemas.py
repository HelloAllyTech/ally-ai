"""Structured output shapes for the analytics agent's two LLM calls.

These are the ``response_schema`` passed to Gemini, so keep them flat and avoid
``Optional`` — the model fills an absent value with the empty default rather
than null, which keeps the parsed object total.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Step 1 — planning
# --------------------------------------------------------------------------


class PlanIntent(str, Enum):
    """What the planner decided to do with the question.

    ``CLARIFY`` and ``REFUSE`` exist so an ambiguous or out-of-scope question
    comes back as a sentence rather than as a guessed query: a plausible SQL
    over the wrong column produces a confident wrong number, which is the one
    failure mode an analytics agent must not have.
    """

    SQL = "sql"
    CLARIFY = "clarify"
    REFUSE = "refuse"


class QueryPlan(BaseModel):
    intent: PlanIntent = PlanIntent.CLARIFY
    # A single read-only SELECT (or WITH ... SELECT). Empty unless intent=SQL.
    sql: str = ""
    # One sentence, shown to the reader next to the SQL: what the query counts
    # and which filters it applied. This is how a reader audits the number
    # without reading the SQL.
    rationale: str = ""
    # The clarifying question (intent=CLARIFY) or the reason the question
    # cannot be answered from these tables (intent=REFUSE).
    message: str = ""


# --------------------------------------------------------------------------
# Step 2 — answering
# --------------------------------------------------------------------------


class ChartType(str, Enum):
    """Deliberately small: every value maps onto an existing chart in the admin
    dashboard's shared chart kit. ``NONE`` is a first-class answer — a single
    scalar or a handful of labelled rows reads better as text and a table than
    as a plot, and the wiki's data-visualisation rules would rather have no
    chart than a decorative one.
    """

    NONE = "none"
    LINE = "line"
    BAR = "bar"
    STACKED_BAR = "stacked_bar"
    SCATTER = "scatter"


class ChartSpec(BaseModel):
    """How to plot the result set, in terms of its own column names."""

    type: ChartType = ChartType.NONE
    # Column driving the x axis (a date/bucket column for line, a category for bar).
    x: str = ""
    # Column holding the numeric measure.
    y: str = ""
    # Optional column that splits y into series. Empty = one series.
    group: str = ""
    x_label: str = ""
    y_label: str = ""
    title: str = ""


class AnswerOutput(BaseModel):
    # Short markdown: the answer to the question in the first sentence, then at
    # most a couple of sentences of what stands out in the data.
    answer: str = ""
    chart: ChartSpec = Field(default_factory=ChartSpec)
    # Honest limits of THIS result — small samples, excluded rows, a metric with
    # more than one defensible definition. Rendered under the answer, not hidden
    # in a tooltip.
    caveats: List[str] = Field(default_factory=list)
    # Suggested next questions; the UI offers them as one-click follow-ups.
    follow_ups: List[str] = Field(default_factory=list)
