from fastapi import APIRouter
from app.api.v1.endpoints import health, conversation, summary

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(conversation.router, prefix="/conversation", tags=["health"])
api_router.include_router(summary.router, prefix="/summary", tags=["health"])
