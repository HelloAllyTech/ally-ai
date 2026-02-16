"""
Production SQS Worker with proper cleanup and shutdown.
"""

import asyncio

from app.core.config import settings
from app.core.constants import SQSWorkerConstants
from app.core.queue.message_processor import MessageProcessor
from app.core.queue.sqs_queue_client import SQSQueueClient
from app.core.queue.sqs_queue_service import SQSQueueService
from app.core.queue.transcription_request_handler import TranscriptionRequestHandler

from app.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Main function with proper cleanup."""
    logger.info("Starting SQS Worker")

    try:
        # Initialize components
        SQSQueueClient.create_client()
        queue_service = SQSQueueService(client=SQSQueueClient.get_client())

        # Pass text_generation_service to the handler
        transcription_request_handler = TranscriptionRequestHandler()

        # Use direct handler instead of router
        transcription_request_processor = MessageProcessor(
            queue_service=queue_service,
            handler=transcription_request_handler.process_transcription_request,
            queue_url=settings.QUEUE.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL,
            max_messages=SQSWorkerConstants.MAX_MESSAGES,
            wait_time_seconds=SQSWorkerConstants.WAIT_TIME_SECONDS,
            visibility_timeout=SQSWorkerConstants.VISIBILITY_TIMEOUT,
            polling_interval=SQSWorkerConstants.POLLING_INTERVAL,
            delete_after_processing=True,
        )

        logger.info(
            f"Starting processor for: {settings.QUEUE.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL}"
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
