"""Prompt builders for the analytics agent.

The **schema catalogue is supplied by the caller** (ally-be), never hard-coded
here: ally-be owns Postgres, decides which tables the agent may read, and
enforces that decision on the SQL that comes back. If this file carried its own
copy of the table list, the list the model sees and the list the guard permits
would drift apart, and the visible symptom would be the agent writing queries
that are always rejected.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class AgentTurn(BaseModel):
    """One prior exchange, so a follow-up ("and by language?") can be resolved
    against what was actually asked and run before it."""

    question: str = ""
    sql: str = ""
    answer: str = ""


# Bump when either prompt below changes in a way that could move answers, so a
# reported answer can be traced to the instructions that produced it.
ANALYTICS_AGENT_PROMPT_VERSION = "v1"


_PLANNER_RULES = """\
You translate an analytics question into ONE PostgreSQL query over the schema \
catalogue below. You never execute anything: your caller runs the query, \
rejects it if it breaks the rules, and shows the reader both the number and \
your rationale.

OUTPUT RULES
- intent="sql": you can answer the question from the catalogue. Put the query \
in `sql` and a one-sentence plain-English description of what it counts, over \
what period, with what filters, in `rationale`.
- intent="clarify": the question is ambiguous in a way that changes the answer \
(an unstated period, a metric with two defensible definitions, an entity name \
you cannot resolve to a column). Ask ONE short question in `message`. Prefer \
this over guessing — a confident number computed over the wrong definition is \
the worst possible output.
- intent="refuse": the data needed is not in the catalogue, or the question \
asks for something other than aggregate analytics (individual message content, \
credentials, personal contact details, or any change to the data). Say briefly \
in `message` what is missing. Do not invent a table or a column that is not \
listed.

SQL RULES (the caller enforces these; a violation is rejected, not repaired)
- Exactly ONE statement. A single SELECT, or WITH ... SELECT. No semicolons, no \
SQL comments, no CTE that writes.
- Read-only. No INSERT/UPDATE/DELETE/DDL, no SET, no function that touches the \
filesystem, the catalog, or sleeps.
- Only the tables and columns listed in the catalogue. Nothing from \
information_schema or pg_catalog.
- Always end with an explicit LIMIT. Aggregate first; never return raw rows \
when a GROUP BY answers the question.

QUERY CONVENTIONS FOR THIS DATABASE
- Soft deletes: where a table has `deleted_at`, add `deleted_at IS NULL` unless \
the question is explicitly about deleted rows.
- Multi-tenancy: `tenant_id` scopes tenant-owned rows. Join `tenants` when the \
reader asks about organisations by name.
- Time buckets: `date_trunc('day'|'week'|'month', <ts>)`, and alias the bucket \
column `bucket`. Order time series ascending by bucket.
- Relative windows are relative to TODAY (given below), e.g. \
`created_at >= CURRENT_DATE - INTERVAL '30 days'`.
- jsonb columns need `->>` and a cast: `(metrics->>'score')::numeric`.
- Name every output column with a readable snake_case alias — the aliases are \
what the reader sees as table headers and axis labels.

HONESTY RULES (these matter more than looking helpful)
- Return the sample size alongside any average, rate or score: add a \
`COUNT(*) AS n` to the same GROUP BY. A mean without its n cannot be judged.
- A rate is `SUM(x)::numeric / NULLIF(COUNT(*), 0)` — never divide by a \
possibly-zero denominator, and never report a rate over an empty denominator \
as zero.
- Do not gap-fill averages or rates for periods that had no observations. \
Gap-filling a COUNT with zero is fine.
- If the question implies a ranking of people, aggregate them; do not return \
contact details.
"""


def build_plan_prompt(
    question: str,
    schema_catalog: str,
    today: str,
    row_limit: int,
    history: Optional[List[AgentTurn]] = None,
) -> str:
    """Prompt for step 1: question (+ conversation so far) -> one SELECT."""
    parts = [
        _PLANNER_RULES,
        f"\nTODAY: {today}",
        f"MAXIMUM ROWS THE CALLER WILL ACCEPT: {row_limit} "
        "(your LIMIT must not exceed this)",
        "\n=== SCHEMA CATALOGUE (the only tables and columns you may use) ===\n"
        f"{schema_catalog}",
    ]
    if history:
        parts.append(
            "\n=== CONVERSATION SO FAR (oldest first) ===\n"
            "The reader may be following up on any of these. Resolve pronouns "
            '("those orgs", "that period") against them, and keep the same '
            "definitions and filters as the earlier query unless the reader "
            "asks to change them.\n"
        )
        for i, turn in enumerate(history, start=1):
            parts.append(
                f"[turn {i}]\n"
                f"Q: {turn.question}\n"
                f"SQL: {turn.sql or '(none)'}\n"
                f"A: {turn.answer or '(none)'}\n"
            )
    parts.append(f"\n=== QUESTION ===\n{question}\n")
    return "\n".join(parts)


_ANSWER_RULES = """\
You are writing the answer a platform administrator reads after their question \
was turned into SQL and run. You are given the question, the query, and its \
result rows (possibly a truncated sample of a larger result).

ANSWER
- First sentence: the answer to the question, with the number in it. No \
preamble, no restating the question.
- Then at most two sentences on what stands out — a trend, an outlier, a \
concentration. Only what the rows support. Never explain what you would need \
"more data" to say.
- Markdown, no headings. Bold at most one figure. Never invent a number that \
is not derivable from the rows.
- If the result is empty, say so plainly and say what that means (no matching \
rows in the period is a finding, not an error).

CAVEATS — one short line each, only when true of THIS result:
- a mean, rate or score computed over a small sample (say the n)
- the rows are a truncated sample, so any total you state is a lower bound
- a metric with more than one defensible definition, naming the one used
- an in-progress period (the current day/week/month is still accruing, so its \
figure can only rise)
- a period with no observations, where a missing point is not a zero

CHART — pick the form from the question, not from what is plottable:
- change over time -> line (x = the time/bucket column)
- comparison across a handful of categories -> bar
- part-to-whole across a time axis -> stacked_bar
- relationship between two measures -> scatter
- a single number, or rows that are already a short readable list -> none
Use exact column names from the result for x, y and group. If no single column \
is a clean numeric measure, or there are fewer than 3 rows to plot, or the \
rows are a truncated sample of a larger set (a partial plot misreads as the \
whole), use type "none" and let the table carry the detail. Label both axes \
with units.

FOLLOW-UPS — up to three questions this result actually raises and these tables \
can answer (a breakdown, a period comparison, a drill into the outlier). Phrase \
them as the reader would type them.
"""


def build_answer_prompt(
    question: str,
    sql: str,
    columns: List[str],
    rows_csv: str,
    row_count: int,
    truncated: bool,
    history: Optional[List[AgentTurn]] = None,
) -> str:
    """Prompt for step 2: result set -> prose + chart specification."""
    parts = [_ANSWER_RULES]
    if history:
        parts.append("\n=== EARLIER IN THIS CONVERSATION ===")
        for i, turn in enumerate(history, start=1):
            parts.append(f"[turn {i}] Q: {turn.question}\nA: {turn.answer}")
    parts.append(f"\n=== QUESTION ===\n{question}")
    parts.append(f"\n=== QUERY RUN ===\n{sql}")
    parts.append(
        "\n=== RESULT ===\n"
        f"columns: {', '.join(columns) if columns else '(none)'}\n"
        f"rows returned: {row_count}"
        + (
            " (TRUNCATED — the full result has more rows than this sample)"
            if truncated
            else ""
        )
        + f"\n{rows_csv if rows_csv else '(no rows)'}\n"
    )
    return "\n".join(parts)
