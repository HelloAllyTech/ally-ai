from fastapi import APIRouter

from app.schemas.conversation import AnalyzeRequest, AnalyzeResponse

router = APIRouter()

@router.post("/analyze", tags=["nudge", "stage"], response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyzes the conversation and provides a stage & optional nudge.
    """
    # Mock response (Replace with actual logic)
    stage = "Mock stage"
    nudge = "### Mock content" if request.generate_nudge else None

    return AnalyzeResponse(nudge=nudge, stage=stage)
