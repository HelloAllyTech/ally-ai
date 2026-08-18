"""Conversation drift judge — a stateless transform.

ally-be owns the data: it selects which sessions to judge, builds the transcript
from its own database, and persists the per-turn judgments. This service is just
the judge: given a transcript it runs the Gemini judge and returns the per-turn
labels plus the derived session rollup. It performs no database access.

The response echoes the judge model + rubric version actually used so the caller
can stamp them onto the stored rows without hard-coding our config on its side.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.drift.judge import judge_session, judge_session_labels_only
from app.core.drift.schemas import DriftJudgmentResult, LeanTurnLabels
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class JudgeRequest(BaseModel):
    # Whole-session transcript: counselor turns {role:"counselor", text} and
    # AI-client turns {role:"client", turn_index, text}. Built by the caller.
    transcript: List[dict] = Field(default_factory=list)
    persona: str = ""
    language: str = "en"
    scenario_goal: Optional[str] = None
    # Static rubric from prompt management; falls back to the inline default.
    rubric: Optional[str] = None


class JudgeResponse(BaseModel):
    judge_model: str
    judge_prompt_version: str
    result: DriftJudgmentResult


@router.post("/judge", response_model=JudgeResponse)
async def judge(req: JudgeRequest) -> JudgeResponse:
    """Judge one session transcript for drift. Stateless: transcript in →
    per-turn judgments + rollup out. The caller persists the result."""
    if not req.transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="empty transcript"
        )
    try:
        result = judge_session(
            req.transcript,
            persona=req.persona or "",
            language=req.language or "en",
            scenario_goal=req.scenario_goal,
            rubric=req.rubric,
        )
    except Exception as e:  # noqa: BLE001 - surface as 500, keep caller decoupled
        logger.error(f"drift judge failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="drift judge failed",
        )
    return JudgeResponse(
        judge_model=settings.DRIFT_JUDGE.MODEL,
        judge_prompt_version=settings.DRIFT_JUDGE.PROMPT_VERSION,
        result=result,
    )


class LeanJudgeResponse(BaseModel):
    judge_model: str
    judge_prompt_version: str
    per_turn: List[LeanTurnLabels]


@router.post("/judge-labels", response_model=LeanJudgeResponse)
async def judge_labels(req: JudgeRequest) -> LeanJudgeResponse:
    """Judge ONLY the labels the v2 rubric added — a backfill path.

    For turns already judged under the earlier rubric, where everything except
    the clienthood and progression labels is already stored. Same request shape,
    same rubric, same model as /judge; the response is constrained to the added
    fields, which is where roughly three quarters of the cost sits.

    NOT for live judging. New sessions must go through /judge so their rows are
    complete: a row topped up by this path relies on a prior full judgment
    existing, and there is none for a session judged for the first time.
    """
    if not req.transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="empty transcript"
        )
    try:
        per_turn = judge_session_labels_only(
            req.transcript,
            persona=req.persona or "",
            language=req.language or "en",
            scenario_goal=req.scenario_goal,
            rubric=req.rubric,
        )
    except Exception as e:  # noqa: BLE001 - surface as 500, keep caller decoupled
        logger.error(f"lean drift judge failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="lean drift judge failed",
        )
    return LeanJudgeResponse(
        judge_model=settings.DRIFT_JUDGE.MODEL,
        judge_prompt_version=settings.DRIFT_JUDGE.PROMPT_VERSION,
        per_turn=per_turn,
    )
