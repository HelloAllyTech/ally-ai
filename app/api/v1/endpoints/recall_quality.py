"""Recall-quality judge — a stateless transform.

Same contract as the other judges: ally-be owns the data, picks which turns to judge, pairs
each recall decision with the turn it belongs to out of its own transcript, and persists the
labels. This service only judges.

The response echoes the model and rubric version actually used, so the caller stamps the
stored rows with what ran rather than hard-coding our config.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.recall_quality.judge import judge_recall
from app.core.recall_quality.schemas import RecallJudgment
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class RecallQualityRequest(BaseModel):
    """One turn's recall decision to audit."""

    # The turn itself. Supplied by ally-be from scenario_session_messages — this service does
    # not look it up, and MUST not: pairing a recall decision with the wrong turn would
    # produce a confident verdict about a conversation that never happened.
    counsellor_turn: str
    client_reply: str
    # Each: {text, score, cue_hits, similarity, ...}. Recalled facts and the near misses.
    selected: List[dict] = Field(default_factory=list)
    passed_over: List[dict] = Field(default_factory=list)
    # The stance recall was scored against. A guarded client withholding a fact is not a
    # recall failure, and a judge that cannot see the stance would read it as one.
    stance: Optional[str] = None
    # Which source supplied the cues. `scenario` means nothing in the conversation drove this
    # selection at all, which is context for a weak choice.
    cue_tier: Optional[str] = None
    rubric: Optional[str] = None


class RecallQualityResponse(BaseModel):
    judge_model: str
    judge_prompt_version: str
    # None only when the judge failed, or when there was no pool to rank. Never a way of
    # saying "this turn needed nothing" — that is `no_demand`, a real verdict, and it is
    # stored as one.
    judgment: Optional[RecallJudgment]


@router.post("/judge", response_model=RecallQualityResponse)
async def judge(request: RecallQualityRequest) -> RecallQualityResponse:
    """Audit one turn's recall: was the fact the turn called for among the five?"""
    if not request.counsellor_turn.strip() and not request.client_reply.strip():
        # Neither half of the turn means nothing to judge against. Refused rather than
        # answered, because a verdict on an empty turn is a verdict about nothing.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="counsellor_turn or client_reply is required",
        )

    try:
        judgment, model = await judge_recall(
            counsellor_turn=request.counsellor_turn,
            client_reply=request.client_reply,
            selected=request.selected,
            passed_over=request.passed_over,
            stance=request.stance,
            cue_tier=request.cue_tier,
            rubric=request.rubric,
        )
    except Exception:
        logger.exception("[recall_quality] judge failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recall-quality judging failed",
        )

    return RecallQualityResponse(
        judge_model=model,
        judge_prompt_version=settings.RECALL_QUALITY_JUDGE.PROMPT_VERSION,
        judgment=judgment,
    )
