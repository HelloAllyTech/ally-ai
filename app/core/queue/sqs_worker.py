"""
Production SQS Worker with proper cleanup and single Ctrl+C shutdown.
"""

import asyncio
import sys
from app.core.queue.sqs_queue_client import SQSQueueClient
from app.core.queue.sqs_queue_service import SQSQueueService
from app.core.queue.message_router import MessageRouter
from app.core.queue.message_processor import MessageProcessor
from app.core.queue.transcription_handler import TranscriptionHandler
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Main function with proper cleanup."""
    logger.info("Starting SQS Worker")

    try:
        # Initialize components
        SQSQueueClient.create_client()
        sqs_client = SQSQueueClient.get_client()
        queue_service = SQSQueueService(client=sqs_client)

        transcription_handler = TranscriptionHandler(queue_service=queue_service)
        router = MessageRouter()
        router.register_transcription_handler(transcription_handler.process_request)

        processor = MessageProcessor(
            queue_service=queue_service,
            router=router,
            queue_url=settings.TRANSCRIPTION_REQUEST_QUEUE_URL,
            max_messages=10,
            wait_time_seconds=20,
            visibility_timeout=30,
            polling_interval=0,
            auto_delete=True,
        )

        logger.info(f"Starting processor for: {settings.TRANSCRIPTION_REQUEST_QUEUE_URL}")

        # Start processor and wait for it to complete
        await processor.start()

        # Wait for the processor task to finish (it handles KeyboardInterrupt)
        if processor._task:
            await processor._task

    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Worker error: {e}")
        raise
    finally:
        # Critical: Close the SQS client to shut down its ThreadPoolExecutor
        logger.info("Cleaning up SQS client...")
        SQSQueueClient.close_client()
        logger.info("SQS client cleanup completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("Worker exited")