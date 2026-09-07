"""Feedback-groundedness judge — one Gemini call per session's feedback.

Mirrors the drift and language judges: the LLM returns per-claim labels only,
every rate is computed by the caller in SQL, and token usage is emitted so the
backfill's cost is visible rather than arriving as an invoice.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from app.core.config import settings
from app.core.feedback_groundedness.prompt import (
    FeedbackClaim,
    TranscriptTurn,
    build_judge_prompt,
)
from app.core.feedback_groundedness.schemas import ClaimJudgment, GroundednessOutput
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def judge_feedback(
    transcript: List[TranscriptTurn],
    claims: List[FeedbackClaim],
    language: str,
    rubric: Optional[str] = None,
) -> Tuple[List[ClaimJudgment], str]:
    """Judge one session's feedback claims against its transcript.

    Returns [] when there is nothing to judge — no claims, or no transcript to
    judge them against. An empty result is meaningfully different from a
    session where every claim was supported, and the caller stores neither as a
    zero.

    The second return value is THE MODEL THAT ACTUALLY RAN, which the caller
    must store rather than its own setting: a fallback recorded under the
    configured model would pollute a pinned series. On the paths where no call
    is made it is the configured model — the honest answer to "what would have
    judged this".
    """
    if not claims or not transcript:
        return [], settings.FEEDBACK_GROUNDEDNESS_JUDGE.MODEL

    prompt = build_judge_prompt(transcript, claims, language, rubric=rubric)

    from app.core.llm.dispatch import PROVIDER_GEMINI, generate_structured
    from app.core.llm_usage.tasks import LLMTask

    output, meta = await generate_structured(
        schema=GroundednessOutput,
        prompt=prompt,
        task=LLMTask.FEEDBACK_GROUNDEDNESS_JUDGE.value,
        provider=PROVIDER_GEMINI,
        model=settings.FEEDBACK_GROUNDEDNESS_JUDGE.MODEL,
        temperature=0,
        # Uncapped, as this call always was: the output length is a
        # property of the input, not a choice.
        max_tokens=None,
    )
    if meta.get("fell_back_from"):
        # These rows will carry a different judgeModel, so a trend that
        # looks like a behaviour change may just be this.
        logger.warning(
            "feedback groundedness judge fell back from %s to %s/%s",
            meta["fell_back_from"],
            meta["provider"],
            meta["model"],
        )
    if output is None or not output.claims:
        logger.warning(
            "[groundedness] judge returned no claims for %d submitted", len(claims)
        )
        return [], meta["model"]

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
    return kept, meta["model"]
