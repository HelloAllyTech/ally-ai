"""Round-trip WER — a stateless transform (PRD FR2, the intelligibility
gate). ally-be samples a session's AI turns and persists the result; this
endpoint just synthesizes → transcribes → scores. See round_trip/service.py.
"""

from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.round_trip.service import run_round_trip
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class UtteranceIn(BaseModel):
    turn_index: int
    text: str


class RoundTripRequest(BaseModel):
    utterances: List[UtteranceIn] = Field(default_factory=list)
    language: str = "en"
    # The session's live TTS provider; unsupported ones fall back to a
    # language default and the response reports provider_used.
    tts_provider: Optional[str] = None
    unit: Literal["wer", "cer"] = "wer"


class UtteranceResultOut(BaseModel):
    turn_index: int
    error_pct: float
    hypothesis: str


class RoundTripResponse(BaseModel):
    provider_used: str
    unit: str
    n_requested: int
    n_measured: int
    avg_error_pct: Optional[float]
    per_utterance: List[UtteranceResultOut]


@router.post("/", response_model=RoundTripResponse)
async def round_trip(req: RoundTripRequest) -> RoundTripResponse:
    if not req.utterances:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="no utterances"
        )
    try:
        result = await run_round_trip(
            [u.model_dump() for u in req.utterances],
            language=req.language,
            requested_provider=req.tts_provider,
            unit=req.unit,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"round-trip failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="round-trip failed",
        )
    return RoundTripResponse(**result)
