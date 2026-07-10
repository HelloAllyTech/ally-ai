from fastapi import APIRouter

from app.api.v1.endpoints import (
    conversation,
    drift,
    language_quality,
    reference_document,
    round_trip,
    summary,
)

api_router = APIRouter()

api_router.include_router(conversation.router, prefix="/conversation", tags=["health"])
api_router.include_router(summary.router, prefix="/summary", tags=["health"])
api_router.include_router(
    reference_document.router,
    prefix="/reference-documents",
    tags=["reference_documents"],
)
api_router.include_router(drift.router, prefix="/drift", tags=["drift"])
api_router.include_router(
    language_quality.router, prefix="/language-quality", tags=["language_quality"]
)
api_router.include_router(
    round_trip.router, prefix="/round-trip-wer", tags=["language_quality"]
)
