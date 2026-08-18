"""Feedback-groundedness judge — one Gemini call per session's feedback.

Mirrors the drift and language judges: the LLM returns per-claim labels only,
every rate is computed by the caller in SQL, and token usage is emitted so the
backfill's cost is visible rather than arriving as an invoice.
"""

from __future__ import annotations

from typing import List, Optional

from app.core.config import settings
from app.core.feedback_groundedness.prompt import (
    FeedbackClaim,
    TranscriptTurn,
    build_judge_prompt,
)
from app.core.feedback_groundedness.schemas import (
    ClaimJudgment,
    GroundednessOutput,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai  # lazily imported; optional dependency

        _client = genai.Client(api_key=settings.GEMINI.API_KEY)
    return _client


def judge_feedback(
    transcript: List[TranscriptTurn],
    claims: List[FeedbackClaim],
    language: str,
    rubric: Optional[str] = None,
) -> List[ClaimJudgment]:
    """Judge one session's feedback claims against its transcript.

    Returns [] when there is nothing to judge — no claims, or no transcript to
    judge them against. An empty result is meaningfully different from a
    session where every claim was supported, and the caller stores neither as a
    zero.
    """
    if not claims or not transcript:
        return []

    from google.genai import types  # lazily imported; optional dependency

    prompt = build_judge_prompt(transcript, claims, language, rubric=rubric)
    client = _get_client()
    response = client.models.generate_content(
        model=settings.FEEDBACK_GROUNDEDNESS_JUDGE.MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=GroundednessOutput,
        ),
    )

    # Best-effort token-usage emission, so a backfill over a year of feedback
    # shows up on the cost dashboard while it runs rather than afterwards.
    try:
        from app.core.llm_usage.emitter import emit_llm_usage_blocking
        from app.core.llm_usage.tasks import LLMTask

        um = getattr(response, "usage_metadata", None)
        if um is not None:
            prompt_tokens = int(getattr(um, "prompt_token_count", 0) or 0)
            completion_tokens = int(getattr(um, "candidates_token_count", 0) or 0)
            total_tokens = int(getattr(um, "total_token_count", 0) or 0) or (
                prompt_tokens + completion_tokens
            )
            emit_llm_usage_blocking(
                provider="gemini",
                model=settings.FEEDBACK_GROUNDEDNESS_JUDGE.MODEL,
                task=LLMTask.FEEDBACK_GROUNDEDNESS_JUDGE.value,
                usage=(prompt_tokens, completion_tokens, total_tokens),
            )
    except Exception:  # noqa: BLE001 — usage must never fail a judge run
        pass

    output: Optional[GroundednessOutput] = response.parsed
    if output is None or not output.claims:
        logger.warning(
            "[groundedness] judge returned no claims for %d submitted", len(claims)
        )
        return []

    # Drop anything pointing at a claim we did not send. A hallucinated index
    # would otherwise be stored against a real claim's row and silently
    # misreport it.
    valid_indices = {c["claim_index"] for c in claims}
    kept: List[ClaimJudgment] = []
    for judged in output.claims:
        if judged.claim_index in valid_indices:
            kept.append(judged)
        else:
            logger.warning(
                "[groundedness] dropping judgment for unknown claim_index=%s",
                judged.claim_index,
            )
    return kept
