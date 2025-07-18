from fastapi import APIRouter
from app.api.v1.endpoints import health, conversation, summary, reference_document, transcription

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(conversation.router, prefix="/conversation", tags=["health"])
api_router.include_router(summary.router, prefix="/summary", tags=["health"])
api_router.include_router(reference_document.router, prefix="/reference-documents", tags=["reference_documents"])
api_router.include_router(transcription.router, prefix="/transcription", tags=["transcription"])
