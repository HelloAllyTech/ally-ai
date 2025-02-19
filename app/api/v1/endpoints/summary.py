from fastapi import APIRouter

from app.schemas.summary import SummaryRequest, SummaryResponse
router = APIRouter()

@router.post("", tags=["summary"], response_model=SummaryResponse)
async def summarize(request: SummaryRequest):
    """
    Summarizes the conversation based on chat history.
    """
    # Mock summary generation (Replace this with actual summarization logic)
    summary_text = "### Mock content based on chat history."

    return SummaryResponse(summary=summary_text)
