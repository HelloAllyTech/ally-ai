from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import root
from app.api.v1.api import api_router
from app.core.ally_core.client import AllyCoreClient
from app.core.config import settings
from app.core.constants import APISettings
from app.core.vector_db.weaviate_client import WeaviateClient
from app.middleware import get_middlewares
from app.utils.logger import get_trace_id, logger, logging_config
from app.utils.startup import initialize_openai_clients


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Application startup and shutdown lifecycle"""
    logger.info("Starting application...")
    # Initialize Weaviate client
    WeaviateClient.create_client()
    await WeaviateClient.connect(WeaviateClient.get_client())
    await AllyCoreClient.create_client()

    # Initialize OpenAI clients
    initialize_openai_clients()
    logger.info("OpenAI clients initialized")

    yield
    await WeaviateClient.close(WeaviateClient.get_client())
    await AllyCoreClient.close(AllyCoreClient.get_client())
    logger.info("Shutting down application...")


# Initialize FastAPI app
app = FastAPI(
    title="Lifeline AI",
    description="AI Service for Lifeline Project",
    version="1.0.0",
    lifespan=lifespan,
    middleware=get_middlewares(),
)

# Catch-all exception handler so dep-resolution / middleware errors don't
# escape as bare uvicorn 500s with no trace_id and no traceback.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    trace_id = get_trace_id()
    logger.exception(
        "Unhandled exception on %s %s (exception_type=%s): %s",
        request.method,
        request.url.path,
        type(exc).__name__,
        str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal Server Error: {type(exc).__name__}",
            "trace_id": trace_id,
        },
        headers={"X-Trace-ID": trace_id},
    )


# Include router

app.include_router(root.router, prefix=APISettings.API_STR)
app.include_router(api_router, prefix=APISettings.API_V1_STR)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.SERVER.HOST,
        port=settings.SERVER.PORT,
        log_config=logging_config,
        access_log=False,
    )
