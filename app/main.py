from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.constants import APISettings
from app.core.vector_db.weaviate_client import WeaviateClient
from app.middleware import get_middlewares
from app.utils.logger import logger, logging_config
from app.utils.startup import initialize_openai_clients


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Application startup and shutdown lifecycle"""
    logger.info("Starting application...")
    # Initialize Weaviate client
    WeaviateClient.create_client()
    await WeaviateClient.connect(WeaviateClient.get_client())

    # Initialize OpenAI clients
    initialize_openai_clients()
    logger.info("OpenAI clients initialized")

    yield
    await WeaviateClient.close(WeaviateClient.get_client())
    logger.info("Shutting down application...")


# Initialize FastAPI app
app = FastAPI(
    title="Lifeline AI",
    description="AI Service for Lifeline Project",
    version="1.0.0",
    lifespan=lifespan,
    middleware=get_middlewares(),
)

# Include routers
app.include_router(api_router, prefix=APISettings.API_V1_STR)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.SERVER.HOST,
        port=settings.SERVER.PORT,
        log_config=logging_config,
        access_log=False,
    )
