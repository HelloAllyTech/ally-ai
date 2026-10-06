"""Text-helpline copilot endpoints — risk screening and the listener's suggestions.

Stateless. The conversation window, the rolling summary and every prompt override arrive
on the request; ally-be owns the chats, the flags, the timeouts (3 s for /risk, 6 s for
/turn) and the decision about what reaches a listener. Both endpoints are staff-side:
nothing they return is ever delivered to a talker.

Both are ALWAYS 200. A failure is reported in `failed`, never as a 5xx: ally-be treats
any failure as "copilot unavailable" and carries on, and a 5xx here would only route it
through a generic error path that knows less about what happened.
"""

from fastapi import APIRouter, Depends, status

from app.core.dependencies import get_helpline_copilot_service
from app.core.helpline.service import HelplineCopilotService
from app.schemas.helpline import (
    HelplineRiskRequest,
    HelplineRiskResponse,
    HelplineTurnRequest,
    HelplineTurnResponse,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/risk",
    response_model=HelplineRiskResponse,
    status_code=status.HTTP_200_OK,
    tags=["helpline"],
)
async def classify_risk(
    payload: HelplineRiskRequest,
    service: HelplineCopilotService = Depends(get_helpline_copilot_service),
):
    """
    Decide whether a talker's latest message signals risk of suicide, self-harm or
    serious harm — to them, from someone else, or to someone they describe.

    The second layer of the helpline's safety net. ally-be's keyword rules are the first
    and are instant and auditable; this catches what a keyword list structurally cannot:
    indirect disclosure, Hinglish, and short replies that only mean something in
    context.
    Deliberately biased towards the false positive.

    ally-be applies its per-org confidence threshold to the verdict (HIGH above it,
    ELEVATED below); this endpoint returns the model's own confidence untouched.

    Always 200, including when the classifier itself failed; `failed` carries that. On
    failure the keyword rules are what still hold.
    """
    try:
        verdict = await service.classify_risk(
            payload.message,
            recent=[turn.model_dump() for turn in payload.recent],
            language=payload.language,
            prompts=payload.prompts,
        )
        return HelplineRiskResponse(**verdict)
    except Exception as e:  # noqa: BLE001
        # The exception CLASS only: the text of a validation error quotes its input,
        # which here is a talker's message.
        logger.error("Helpline risk check failed unexpectedly: %s", type(e).__name__)
        return HelplineRiskResponse(failed=True)


@router.post(
    "/turn",
    response_model=HelplineTurnResponse,
    status_code=status.HTTP_200_OK,
    tags=["helpline"],
)
async def copilot_turn(
    payload: HelplineTurnRequest,
    service: HelplineCopilotService = Depends(get_helpline_copilot_service),
):
    """
    Draft two or three reply options for the listener, name the conversation stage and,
    when asked, give the listener one coaching nudge.

    Suggestions are DRAFTS written as the listener speaking to the talker: tentative and
    validating, each tagged with the helping skill it uses, never a promise of
    unconditional confidentiality, a diagnosis or a mention of medication (a
    post-generation filter drops any that are). When the latest talker message shows
    risk, they include asking directly about safety. The list can be empty if nothing
    safe was generated; the caller must not backfill it.

    Always 200, including on failure; `failed` carries that.
    """
    try:
        result = await service.copilot_turn(
            [turn.model_dump() for turn in payload.messages],
            rolling_summary=payload.rolling_summary,
            language=payload.language,
            include_nudge=payload.include_nudge,
            risk_level=payload.risk_level,
            risk_subject=payload.risk_subject,
            prompts=payload.prompts,
        )
        return HelplineTurnResponse(**result)
    except Exception as e:  # noqa: BLE001
        logger.error("Helpline copilot turn failed unexpectedly: %s", type(e).__name__)
        return HelplineTurnResponse(failed=True)
