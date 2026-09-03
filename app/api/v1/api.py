from fastapi import APIRouter

from app.api.v1.endpoints import (
    analytics_agent,
    conversation,
    drift,
    feedback_groundedness,
    filler_quality,
    knowledge_agent,
    knowledge_chunk,
    language_quality,
    reference_document,
    roadmap_opportunity,
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
api_router.include_router(
    roadmap_opportunity.router,
    prefix="/roadmap-opportunities",
    tags=["roadmap_opportunities"],
)
api_router.include_router(
    knowledge_chunk.router,
    prefix="/knowledge-chunks",
    tags=["knowledge_chunks"],
)
api_router.include_router(
    knowledge_agent.router,
    prefix="/knowledge-agent",
    tags=["knowledge_agent"],
)
api_router.include_router(drift.router, prefix="/drift", tags=["drift"])
api_router.include_router(
    feedback_groundedness.router,
    prefix="/feedback-groundedness",
    tags=["feedback-groundedness"],
)
api_router.include_router(
    analytics_agent.router, prefix="/analytics-agent", tags=["analytics_agent"]
)
api_router.include_router(
    language_quality.router, prefix="/language-quality", tags=["language_quality"]
)
api_router.include_router(
    filler_quality.router, prefix="/filler-quality", tags=["filler_quality"]
)
api_router.include_router(
    round_trip.router, prefix="/round-trip-wer", tags=["language_quality"]
)
