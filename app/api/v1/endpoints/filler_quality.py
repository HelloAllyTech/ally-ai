"""Thinking-filler judge — a stateless transform.

The thinking filler is the short back-channel the AI client utters the instant
the learner stops speaking, masking the wait for its real reply. Its speed is
already measured per turn in ally-ai-learn; its QUALITY was not measured at all,
which is the gap this endpoint closes. That gap matters more than it sounds: a
filler counts as the character's first words, so a filler that is fast but
sounds nothing like the character — or answers the previous turn — improves
every latency chart while making the roleplay worse.

Ownership mirrors the language-quality judge exactly. ally-be selects which
sessions to judge, builds the observations from its own transcript and per-turn
filler metadata, and persists the returned rows. This service performs no
database access and no aggregation.

The response echoes the judge model + rubric version actually used so the caller
can stamp them onto the stored rows without hard-coding our config.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.filler_quality.judge import judge_session
from app.core.filler_quality.prompt import FillerStyleParams
from app.core.filler_quality.schemas import (
    DEFAULT_REPEAT_WINDOW_PLAYS,
    FillerJudgmentResult,
    FillerObservation,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class JudgeRequest(BaseModel):
    #: Every filler the learner actually heard this session, in play order.
    #: Order is load-bearing: repeat distance is computed from it.
    observations: List[FillerObservation] = Field(default_factory=list)
    persona: str = ""
    language: str = "en"
    #: How the character was configured to speak. Without it the judge cannot
    #: tell a model failure from an unconfigured scenario.
    style_config: Optional[FillerStyleParams] = None
    #: Static rubric from prompt management; falls back to the inline default.
    rubric: Optional[str] = None
    #: Recent-play window for repeat detection. Defaults to the player's own.
    repeat_window_plays: int = DEFAULT_REPEAT_WINDOW_PLAYS


class JudgeResponse(BaseModel):
    judge_model: str
    judge_prompt_version: str
    result: FillerJudgmentResult


@router.post("/judge", response_model=JudgeResponse)
async def judge(req: JudgeRequest) -> JudgeResponse:
    """Judge one session's thinking fillers. Stateless: observations in →
    per-filler judgements out. The caller persists the result.

    An empty observation list is a valid, cheap answer rather than a 400: a
    session that played no fillers is the normal state of a fast session, and
    the caller needs to record that it was judged and found nothing to judge —
    not retry it forever as a failure.
    """
    try:
        result = judge_session(
            req.observations,
            req.persona,
            req.language,
            style_params=req.style_config,
            rubric=req.rubric,
            window_plays=req.repeat_window_plays,
        )
    except RuntimeError as exc:
        # Missing key, or the model returned nothing parsable. Both are the
        # caller's cue to skip this session and log, not to store an empty
        # judgement as "every filler was fine".
        logger.warning(f"filler judge failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    return JudgeResponse(
        judge_model=settings.FILLER_JUDGE.MODEL,
        judge_prompt_version=settings.FILLER_JUDGE.PROMPT_VERSION,
        result=result,
    )
