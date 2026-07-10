"""Language-quality judge — a stateless transform.

ally-be owns the data: it selects which sessions to judge, builds the
transcript from its own database, and persists the per-turn error annotations
(session-level rows surface in Roleplay Session Logs; the analytics dashboard
aggregates the same rows). This service is just the judge: transcript in →
per-turn annotations out. It performs no database access.

The response echoes the judge model + rubric version actually used so the
caller can stamp them onto the stored rows without hard-coding our config.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.language_quality.judge import judge_session
from app.core.language_quality.prompt import LanguageEvalParams, ScenarioStyleParams
from app.core.language_quality.schemas import LanguageJudgmentResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class JudgeRequest(BaseModel):
    # Whole-session transcript: counselor turns {role:"counselor", text} and
    # AI-client turns {role:"client", turn_index, text}. Built by the caller.
    transcript: List[dict] = Field(default_factory=list)
    persona: str = ""
    language: str = "en"
    # Per-language eval config (languages.evalConfig) — optional; absent
    # values render as "unknown" in the judge's parameter block.
    language_eval_config: Optional[LanguageEvalParams] = None
    # Scenario per-language style presence flags + fillers — lets the judge
    # label persona_specified vs persona_unspecified (prompt-vs-model rule).
    scenario_style_config: Optional[ScenarioStyleParams] = None
    # Static rubric from prompt management; falls back to the inline default.
    rubric: Optional[str] = None


class JudgeResponse(BaseModel):
    judge_model: str
    judge_prompt_version: str
    result: LanguageJudgmentResult


@router.post("/judge", response_model=JudgeResponse)
async def judge(req: JudgeRequest) -> JudgeResponse:
    """Judge one session transcript for language-quality errors. Stateless:
    transcript in → per-turn annotations out. The caller persists the result."""
    if not req.transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="empty transcript"
        )
    try:
        result = judge_session(
            req.transcript,
            persona=req.persona or "",
            language=req.language or "en",
            language_params=req.language_eval_config,
            style_params=req.scenario_style_config,
            rubric=req.rubric,
        )
    except Exception as e:  # noqa: BLE001 - surface as 500, keep caller decoupled
        logger.error(f"language judge failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="language judge failed",
        )
    return JudgeResponse(
        judge_model=settings.LANGUAGE_JUDGE.MODEL,
        judge_prompt_version=settings.LANGUAGE_JUDGE.PROMPT_VERSION,
        result=result,
    )
