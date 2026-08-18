"""Feedback-groundedness judge — a stateless transform.

Same contract as the drift and language judges: ally-be owns the data, selects
the sessions, builds the transcript and claim list from its own database, and
persists the result. This service only judges.

The response echoes the model and rubric version actually used, so the caller
stamps the stored rows with what ran rather than hard-coding our config.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.feedback_groundedness.judge import judge_feedback
from app.core.feedback_groundedness.schemas import ClaimJudgment
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class GroundednessRequest(BaseModel):
    # Whole-session transcript, same shape the drift judge takes.
    transcript: List[dict] = Field(default_factory=list)
    # Each: {claim_index, kind: "positive"|"improvement", text}
    claims: List[dict] = Field(default_factory=list)
    language: str = "en"
    rubric: Optional[str] = None


class GroundednessResponse(BaseModel):
    judge_model: str
    judge_prompt_version: str
    claims: List[ClaimJudgment]


@router.post("/judge", response_model=GroundednessResponse)
async def judge(req: GroundednessRequest) -> GroundednessResponse:
    """Judge one session's feedback claims against its transcript."""
    if not req.transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="empty transcript"
        )
    if not req.claims:
        # Not an error: a session can legitimately have feedback with no
        # checkable claims. Returning an empty list lets the caller record
        # "judged, nothing to check" rather than retrying forever.
        return GroundednessResponse(
            judge_model=settings.FEEDBACK_GROUNDEDNESS_JUDGE.MODEL,
            judge_prompt_version=settings.FEEDBACK_GROUNDEDNESS_JUDGE.PROMPT_VERSION,
            claims=[],
        )

    try:
        claims = judge_feedback(
            req.transcript,
            req.claims,  # type: ignore[arg-type]
            req.language,
            rubric=req.rubric,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[groundedness] judge failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"groundedness judge failed: {exc}",
        ) from exc

    return GroundednessResponse(
        judge_model=settings.FEEDBACK_GROUNDEDNESS_JUDGE.MODEL,
        judge_prompt_version=settings.FEEDBACK_GROUNDEDNESS_JUDGE.PROMPT_VERSION,
        claims=claims,
    )
