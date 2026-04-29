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
    ScenarioEvaluationRequest,
    ScenarioEvaluationResponse,
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
                request.chat_history,
                keys=request.keys,
                key_descriptions=request.key_descriptions,
                prompts=request.prompts,
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
        enhanced_content = await summary_service.enhance_content(
            request.content, prompts=request.prompts
        )
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
        tags = await summary_service.get_tag_positivity_ratings(
            request.tags, prompts=request.prompts
        )
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
    "/scenario/evaluate",
    tags=["simulation", "analysis"],
    response_model=ScenarioEvaluationResponse,
)
async def generate_scenario_evaluation(
    request: ScenarioEvaluationRequest,
    summary_service: SummaryService = Depends(get_summary_service),
):
    """
    Generates comprehensive scenario evaluation based on chat history.

    Analyzes conversation performance to identify improvement areas and positives
    based on clinical counseling competencies.

    Requires:
    - chat_history: List of messages with unique IDs

    Always returns:
    - improvements: Areas needing development
    - positives: Demonstrated strengths

    When need_memory=True, additionally returns:
    - session_glimpse: Brief overview of the current session
    - cumulative_memory: Comprehensive cumulative narrative across sessions
    """
    try:
        evaluation_response = await summary_service.generate_scenario_evaluation(
            chat_history=request.chat_history,
            need_memory=request.need_memory,
            previous_memory=request.previous_memory,
            memory_prompt=request.memory_prompt,
            prompts=request.prompts,
        )

        return ScenarioEvaluationResponse(**evaluation_response)
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
