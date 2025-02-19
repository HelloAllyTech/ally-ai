import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.v1.api import api_router
from app.core.config import settings
from app.middleware import get_middlewares
from app.utils.logger import logger, logging_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle"""
    logger.info("Starting application...")
    yield
    logger.info("Shutting down application...")


# Initialize FastAPI app
app = FastAPI(
    title="My FastAPI Project",
    description="A well-structured FastAPI boilerplate",
    version="1.0.0",
    lifespan=lifespan,
    middleware=get_middlewares(),
)

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    uvicorn.run(app, host=settings.SERVER_HOST, port=settings.SERVER_PORT, log_config=logging_config, access_log=False)
