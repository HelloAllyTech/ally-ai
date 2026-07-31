"""The analytics agent's two LLM calls.

Both are stateless transforms and neither touches a database: ally-be supplies
the schema catalogue, runs the planned SQL behind its own guard, and sends the
rows back for narration. Splitting planning from narration is deliberate — the
planner must not see result rows (it would start writing the answer instead of
the query), and the narrator must not be able to change the query that produced
the number it is describing.

The Gemini SDK and client are imported/constructed lazily so this module can be
imported without the ``google-genai`` dependency installed or a key configured
(mirrors ``app.core.language_quality.judge``).
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional

from app.core.analytics_agent.prompt import (
    AgentTurn,
    build_answer_prompt,
    build_plan_prompt,
)
from app.core.analytics_agent.schemas import (
    AnswerOutput,
    ChartType,
    PlanIntent,
    QueryPlan,
)
from app.core.config import settings
from app.core.llm_usage.tasks import LLMTask
from app.utils.logger import get_logger

logger = get_logger(__name__)

_client = None

# Result rows are stringified into the narration prompt; cap the per-cell length
# so one long text column cannot crowd out the rest of the sample.
MAX_CELL_CHARS = 200


def _get_client():
    """Lazily build the Gemini client; clear error if the key is missing."""
    global _client
    if _client is None:
        if not settings.GEMINI.API_KEY:
            raise RuntimeError(
                "GEMINI__API_KEY is not configured — cannot run the analytics agent."
            )
        from google import genai  # imported lazily; optional dependency

        _client = genai.Client(api_key=settings.GEMINI.API_KEY)
    return _client


def _emit_usage(response: Any, model: str, task: str) -> None:
    """Best-effort token-usage emission for the cost-by-model/task dashboard."""
    try:
        from app.core.llm_usage.emitter import emit_llm_usage_blocking

        um = getattr(response, "usage_metadata", None)
        if um is None:
            return
        prompt_tokens = int(getattr(um, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(um, "candidates_token_count", 0) or 0)
        total_tokens = int(getattr(um, "total_token_count", 0) or 0) or (
            prompt_tokens + completion_tokens
        )
        emit_llm_usage_blocking(
            provider="gemini",
            model=model,
            task=task,
            usage=(prompt_tokens, completion_tokens, total_tokens),
        )
    except Exception:  # noqa: BLE001 — usage accounting never fails a request
        pass


def rows_to_csv(columns: List[str], rows: List[Dict[str, Any]]) -> str:
    """Render result rows as CSV for the narration prompt.

    CSV rather than JSON: one line per row keeps a 200-row sample legible to the
    model at roughly half the tokens, and the column header is stated once.
    """
    if not columns:
        return ""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(
            [_format_cell(row.get(col)) for col in columns],
        )
    return buf.getvalue().strip()


def _format_cell(value: Any) -> str:
    if value is None:
        # Distinguishable from an empty string: a NULL average is "not measured",
        # which the narrator must not read as a zero.
        return "NULL"
    text = str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[:MAX_CELL_CHARS] + "…"


def plan_query(
    question: str,
    schema_catalog: str,
    today: str,
    row_limit: int,
    history: Optional[List[AgentTurn]] = None,
) -> QueryPlan:
    """Step 1: turn the question into one read-only SELECT (or ask/refuse)."""
    from google.genai import types  # imported lazily; optional dependency

    prompt = build_plan_prompt(
        question,
        schema_catalog=schema_catalog,
        today=today,
        row_limit=row_limit,
        history=history,
    )
    client = _get_client()
    model = settings.ANALYTICS_AGENT.PLANNER_MODEL
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            # Zero temperature: the same question over the same schema should
            # produce the same query, or a reader comparing two answers a minute
            # apart cannot tell a data change from a sampling wobble.
            temperature=0,
            response_mime_type="application/json",
            response_schema=QueryPlan,
        ),
    )
    _emit_usage(response, model, LLMTask.ANALYTICS_AGENT_PLAN.value)
    plan: Optional[QueryPlan] = response.parsed
    if plan is None:
        raise RuntimeError("analytics agent planner returned no parsable output")
    if plan.intent == PlanIntent.SQL and not plan.sql.strip():
        # An empty query with a "yes I can answer this" intent would surface to
        # the reader as a silent nothing; make it a clarification instead.
        logger.warning("analytics agent planner returned intent=sql with empty sql")
        return QueryPlan(
            intent=PlanIntent.CLARIFY,
            message=(
                "I could not turn that into a query. Could you rephrase it, "
                "naming the metric and the period you want?"
            ),
        )
    return plan


def compose_answer(
    question: str,
    sql: str,
    columns: List[str],
    rows: List[Dict[str, Any]],
    row_count: int,
    truncated: bool,
    history: Optional[List[AgentTurn]] = None,
) -> AnswerOutput:
    """Step 2: turn the result set into prose, caveats and a chart spec."""
    from google.genai import types  # imported lazily; optional dependency

    prompt = build_answer_prompt(
        question,
        sql=sql,
        columns=columns,
        rows_csv=rows_to_csv(columns, rows),
        row_count=row_count,
        truncated=truncated,
        history=history,
    )
    client = _get_client()
    model = settings.ANALYTICS_AGENT.ANSWER_MODEL
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=AnswerOutput,
        ),
    )
    _emit_usage(response, model, LLMTask.ANALYTICS_AGENT_ANSWER.value)
    output: Optional[AnswerOutput] = response.parsed
    if output is None or not output.answer.strip():
        raise RuntimeError("analytics agent narrator returned no parsable output")
    return validate_chart(output, columns, row_count, truncated)


def validate_chart(
    output: AnswerOutput,
    columns: List[str],
    row_count: int,
    truncated: bool,
) -> AnswerOutput:
    """Drop a chart specification that the result cannot honestly support.

    Deterministic code, not a prompt instruction: a chart naming a column that
    is not in the result renders as an empty plot, and a plot of a truncated
    sample reads as the whole population. Both are silent failures, so they are
    checked here rather than trusted to the model.
    """
    chart = output.chart
    if chart.type.value == "none":
        return output

    reason = None
    if chart.x not in columns or chart.y not in columns:
        reason = "chart references a column not in the result"
    elif chart.group and chart.group not in columns:
        reason = "chart group references a column not in the result"
    elif row_count < 3:
        reason = "too few rows to plot"
    elif truncated:
        reason = "result is a truncated sample"

    if reason:
        logger.info(f"analytics agent: dropping chart — {reason}")
        chart.type = ChartType.NONE
    return output
