from fastapi import APIRouter, HTTPException, status
from fastapi.params import Depends

from app.core.conversations.conversation_service import ConversationService
from app.core.dependencies import get_conversation_service
from app.exceptions.custom_exceptions import ConversationAnalysisFailedException
from app.schemas.conversation import (
    AnalyzeRequest,
    AnalyzeResponse,
    IdentifyRequest,
    IdentifyResponse,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/analyze", tags=["nudge", "stage"], response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> AnalyzeResponse:
    """
    Analyzes the conversation and provides a stage & optional nudge.
    """
    try:
        stage, nudge = await conversation_service.analyze(
            request.latest_message, request.chat_history, request.force_nudge
        )

        return AnalyzeResponse(nudge=nudge, stage=stage)

    except ConversationAnalysisFailedException:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Conversation analysis failed",
        )

    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong. Please try again later.",
        )


@router.post("/identify", tags=["nudge", "stage"], response_model=IdentifyResponse)
async def identify(
    request: IdentifyRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> IdentifyResponse:
    """
    Identifies the users who did the conversation from the conversation history.
    """
    return await conversation_service.identify(request.chat_history)
