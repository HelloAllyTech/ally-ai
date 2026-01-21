"""
Production SQS Worker with proper cleanup and shutdown.
"""

import asyncio

from app.core.ally_core import AllyCoreClient, AllyCoreService
from app.core.config import settings
from app.core.constants import SQSWorkerConstants
from app.core.embeddings.openai_embedding_client import OpenAIEmbeddingClient
from app.core.embeddings.openai_embedding_service import OpenAIEmbeddingService
from app.core.queue.message_processor import MessageProcessor
from app.core.queue.sqs_queue_client import SQSQueueClient
from app.core.queue.sqs_queue_service import SQSQueueService
from app.core.queue.transcription_handler import TranscriptionHandler
from app.core.storage.s3_service import S3Service
from app.core.text_generations.openai_text_generation_client import (
    OpenAITextGenerationClient,
)
from app.core.text_generations.openai_text_generation_service import (
    OpenAITextGenerationService,
)
from app.utils.logger import get_logger
from app.utils.startup import initialize_openai_clients

logger = get_logger(__name__)


async def main():
    """Main function with proper cleanup."""
    logger.info("Starting SQS Worker")

    try:
        # Initialize components
        SQSQueueClient.create_client()
        queue_service = SQSQueueService(client=SQSQueueClient.get_client())
        ally_core_service = AllyCoreService(AllyCoreClient.get_client())

        # Initialize OpenAI clients
        initialize_openai_clients()

        text_generation_service = OpenAITextGenerationService(
            client=OpenAITextGenerationClient.get_client(),
            embedding_service=OpenAIEmbeddingService(
                OpenAIEmbeddingClient.get_client()
            ),
        )

        # Pass text_generation_service to the handler
        transcription_handler = TranscriptionHandler(
            ally_core_service=ally_core_service,
            request_queue_url=settings.QUEUE.TRANSCRIPTION_RESULTS_QUEUE_URL,
            text_generation_service=text_generation_service,
            storage_service=S3Service(),
            bucket_name=settings.QUEUE.TRANSCRIBE_AND_SUMMARIZE_RESULTS_BUCKET,
        )

        # Use direct handler instead of router
        transcription_processor = MessageProcessor(
            queue_service=queue_service,
            handler=transcription_handler.process_request,
            queue_url=settings.QUEUE.TRANSCRIPTION_RESULTS_QUEUE_URL,
            max_messages=SQSWorkerConstants.MAX_MESSAGES,
            wait_time_seconds=SQSWorkerConstants.WAIT_TIME_SECONDS,
            visibility_timeout=SQSWorkerConstants.VISIBILITY_TIMEOUT,
            polling_interval=SQSWorkerConstants.POLLING_INTERVAL,
            delete_after_processing=True,
        )

        logger.info(
            f"Starting processor for: {settings.QUEUE.TRANSCRIPTION_RESULTS_QUEUE_URL}"
        )

        # Start processor and wait for it to complete
        await transcription_processor.start()

        # Wait for the processor task to finish (it handles KeyboardInterrupt)
        if transcription_processor._task:
            await transcription_processor._task

    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Worker error: {type(e).__name__}")
        raise
    finally:
        # Critical: Close the SQS client to shut down its ThreadPoolExecutor
        logger.info("Cleaning up SQS client...")
        await SQSQueueClient.close_client()
        logger.info("SQS client cleanup completed")


async def run_worker():
    """Async wrapper for the main worker function."""
    try:
        await main()
    except KeyboardInterrupt:
        logger.info("Worker stopped")
    except Exception as e:
        logger.exception(f"Fatal error: {type(e).__name__}")
        raise
    finally:
        logger.info("Worker exited")


if __name__ == "__main__":
    asyncio.run(run_worker())
