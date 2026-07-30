"""Analytics Agent — two stateless transforms behind the admin Analytics tab.

ally-be owns the data and the trust boundary: it decides which tables the agent
may read, sends that catalogue with every request, validates the SQL that comes
back against its own guard, executes it read-only, and sends the rows here to be
narrated. This service performs no database access and holds no conversation
state — the history it needs arrives on the request, so "reset chat" is a
client-side act with nothing to clean up here.

Two endpoints rather than one because the two halves must not see each other's
inputs: the planner writing SQL must not see result rows, and the narrator
describing a number must not be able to change the query that produced it.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.analytics_agent.agent import compose_answer, plan_query
from app.core.analytics_agent.prompt import AgentTurn
from app.core.analytics_agent.schemas import AnswerOutput, QueryPlan
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class PlanRequest(BaseModel):
    question: str
    # The tables and columns the agent is allowed to use, rendered by ally-be
    # from its own allowlist. Required: without it the planner would invent a
    # schema, and every query it wrote would be rejected by the guard.
    schema_catalog: str
    # ally-be's date, so "last 30 days" is resolved against the server that runs
    # the query rather than against the model's training cutoff.
    today: str
    row_limit: int = 500
    history: List[AgentTurn] = Field(default_factory=list)


class PlanResponse(BaseModel):
    planner_model: str
    prompt_version: str
    plan: QueryPlan


class AnswerRequest(BaseModel):
    question: str
    sql: str
    columns: List[str] = Field(default_factory=list)
    # The result rows, or a sample of them when the full result was larger than
    # ally-be's narration cap; `truncated` says which.
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    history: List[AgentTurn] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    answer_model: str
    prompt_version: str
    result: AnswerOutput


@router.post("/plan", response_model=PlanResponse)
async def plan(req: PlanRequest) -> PlanResponse:
    """Question (+ conversation so far) -> one read-only SELECT, or a
    clarifying question when the answer would otherwise be a guess."""
    if not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="empty question"
        )
    if not req.schema_catalog.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="empty schema catalog"
        )
    try:
        result = plan_query(
            req.question,
            schema_catalog=req.schema_catalog,
            today=req.today,
            row_limit=req.row_limit,
            history=req.history,
        )
    except Exception as e:  # noqa: BLE001 - surface as 500, keep caller decoupled
        logger.error(f"analytics agent planning failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="analytics agent planning failed",
        )
    return PlanResponse(
        planner_model=settings.ANALYTICS_AGENT.PLANNER_MODEL,
        prompt_version=settings.ANALYTICS_AGENT.PROMPT_VERSION,
        plan=result,
    )


@router.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest) -> AnswerResponse:
    """Result rows -> the answer, its caveats, a chart specification and
    suggested follow-ups. An empty result is narrated, not rejected: "no
    sessions in that period" is a finding."""
    if not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="empty question"
        )
    try:
        result = compose_answer(
            req.question,
            sql=req.sql,
            columns=req.columns,
            rows=req.rows,
            row_count=req.row_count,
            truncated=req.truncated,
            history=req.history,
        )
    except Exception as e:  # noqa: BLE001 - surface as 500, keep caller decoupled
        logger.error(f"analytics agent narration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="analytics agent narration failed",
        )
    return AnswerResponse(
        answer_model=settings.ANALYTICS_AGENT.ANSWER_MODEL,
        prompt_version=settings.ANALYTICS_AGENT.PROMPT_VERSION,
        result=result,
    )
