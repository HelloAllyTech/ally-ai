from fastapi import APIRouter, HTTPException, status
from fastapi.params import Depends

from app.core.transcriptions.openai.transcription_service import OpenAITranscriptionService
from app.core.dependencies import get_transcription_service
from app.schemas.transcription import TranscribeAndSummarizeRequest, TranscribeAndSummarizeResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/transcribe-and-summarize", tags=["transcription"], response_model=TranscribeAndSummarizeResponse)
async def transcribe_and_summarize_audio(
    request: TranscribeAndSummarizeRequest,
    transcription_service: OpenAITranscriptionService = Depends(get_transcription_service)
) -> TranscribeAndSummarizeResponse:
    """
    Transcribe audio from URL and generate a summary using OpenAI.
    
    This endpoint accepts an URL containing an audio file, transcribes it,
    performs speaker diarization, and generates a summary of the conversation.
    Returns a success indicator.
    """
    try:
        success = await transcription_service.transcribe_audio_from_url(
            presigned_url=request.presigned_url,
            chat_id=request.chat_id,
            sample_rate=request.sample_rate
        )
        
        return TranscribeAndSummarizeResponse(success=success)
        
    except Exception as e:
        logger.exception(f"Error in transcribe-and-summarize endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription and summarization failed: {str(e)}"
        ) 