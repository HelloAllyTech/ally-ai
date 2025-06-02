from fastapi import APIRouter
from fastapi.params import Depends
from typing import Union

from app.core.dependencies import get_summary_service
from app.core.summaries.summary_service import SummaryService
from app.schemas.summary import SummaryNoteAndTagsRequest, SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse, ContentEnhanceResponse, ContentEnhanceRequest, TagPositivityRatingRequest, TagPositivityRatingResponse
from app.utils.common import convert_chat_messages_to_string

router = APIRouter()


@router.post("/note", tags=["note", "summary"], response_model=Union[SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse])
async def create_note_and_tags(
        request: SummaryNoteAndTagsRequest,
        summary_service: SummaryService = Depends(get_summary_service)
):
    """
    Summarizes the conversation based on chat history.
    """
    summary_note_and_tags_response = await summary_service.generate_summary_and_tags(
        convert_chat_messages_to_string(request.chat_history),
        request.keys
    )

    return summary_note_and_tags_response


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


@router.post("/tag-positivity-ratings", tags=["tags", "summary"], response_model=TagPositivityRatingResponse)
async def get_tag_positivity_ratings(
        request: TagPositivityRatingRequest,
        summary_service: SummaryService = Depends(get_summary_service)
):
    """
    Get positivity ratings for a list of tags.
    """
    tags = await summary_service.get_tag_positivity_ratings(request.tags)
    
    return TagPositivityRatingResponse(tags=tags)
