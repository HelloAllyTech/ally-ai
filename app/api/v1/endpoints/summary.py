from typing import Union

from fastapi import APIRouter, HTTPException, status
from fastapi.params import Depends

from app.core.dependencies import get_summary_service
from app.core.summaries.summary_service import SummaryService
from app.exceptions.custom_exceptions import (
    CounselorTrainingAnalysisFailedException,
    SummarizationFailedException,
)
from app.schemas.summary import (
    ContentEnhanceRequest,
    ContentEnhanceResponse,
    DynamicSummaryNoteResponse,
    SimulationAnalysisRequest,
    SimulationAnalysisResponse,
    SummaryNoteAndTagsRequest,
    SummaryNoteAndTagsResponse,
    TagPositivityRatingRequest,
    TagPositivityRatingResponse,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/note",
    tags=["note", "summary"],
    response_model=Union[SummaryNoteAndTagsResponse, DynamicSummaryNoteResponse],
)
async def create_note_and_tags(
    request: SummaryNoteAndTagsRequest,
    summary_service: SummaryService = Depends(get_summary_service),
):
    """
    Summarizes the conversation based on chat history.
    """
    try:
        summary_note_and_tags_response = (
            await summary_service.generate_summary_and_tags(
                request.chat_history, request.keys
            )
        )
        return summary_note_and_tags_response
    except SummarizationFailedException:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Summary generation failed",
        )
    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong. Please try again later.",
        )


@router.post(
    "/enhance", tags=["enhance", "summary"], response_model=ContentEnhanceResponse
)
async def enhance(
    request: ContentEnhanceRequest,
    summary_service: SummaryService = Depends(get_summary_service),
):
    """
    Enhances the content
    """
    try:
        enhanced_content = await summary_service.enhance_content(request.content)
        return ContentEnhanceResponse(enhanced_content=enhanced_content)
    except SummarizationFailedException:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content enhancement failed",
        )
    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong. Please try again later.",
        )


@router.post(
    "/tag-positivity-ratings",
    tags=["tags", "summary"],
    response_model=TagPositivityRatingResponse,
)
async def get_tag_positivity_ratings(
    request: TagPositivityRatingRequest,
    summary_service: SummaryService = Depends(get_summary_service),
):
    """
    Get positivity ratings for a list of tags.
    """
    try:
        tags = await summary_service.get_tag_positivity_ratings(request.tags)
        return TagPositivityRatingResponse(tags=tags)
    except SummarizationFailedException:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tag positivity rating generation failed",
        )
    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong. Please try again later.",
        )


@router.post(
    "/scenario/feedback",
    tags=["simulation", "analysis"],
    response_model=SimulationAnalysisResponse,
)
async def generate_simulation_analysis(
    request: SimulationAnalysisRequest,
    summary_service: SummaryService = Depends(get_summary_service),
):
    """
    Generates simulation analysis based on chat history and goal.

    Analyzes conversation performance to identify improvement areas and positives.
    """
    try:
        analysis_response = await summary_service.generate_simulation_summary(
            request.chat_history, request.goal
        )

        return SimulationAnalysisResponse(**analysis_response)
    except CounselorTrainingAnalysisFailedException:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Counselor training analysis generation failed",
        )
    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong. Please try again later.",
        )
