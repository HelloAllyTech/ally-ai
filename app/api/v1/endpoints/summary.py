from fastapi import APIRouter
from fastapi.params import Depends

from app.core.dependencies import get_summary_service
from app.core.summaries.summary_service import SummaryService
from app.schemas.summary import SummaryNoteRequest, SummaryNoteResponse, ContentEnhanceResponse, ContentEnhanceRequest
from app.utils.common import convert_chat_messages_to_string

router = APIRouter()


@router.post("/note", tags=["note", "summary"], response_model=SummaryNoteResponse)
async def summarize(
        request: SummaryNoteRequest,
        summary_service: SummaryService = Depends(get_summary_service)
):
    """
    Summarizes the conversation based on chat history.
    """
    summary_note = await summary_service.generate_summary_and_tags(
        convert_chat_messages_to_string(request.chat_history)
    )

    return SummaryNoteResponse(summary_note=summary_note)


@router.post("/enhance", tags=["enhance", "summary"], response_model=ContentEnhanceResponse)
async def enhance(
        request: ContentEnhanceRequest,
        summary_service: SummaryService = Depends(get_summary_service)
):
    """
    Enhances the content
    """
    enhanced_content = await summary_service.enhance_content(request.content)

    return ContentEnhanceResponse(enhanced_content=enhanced_content)
