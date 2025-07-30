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
from app.core.text_generations.openai_text_generation_service import OpenAITextGenerationService
from app.core.transcriptions.deepgram import DeepgramTranscriptionService
from app.core.s3.s3_service import S3Service
from app.core.constants import SQSWorkerConstants
from app.utils.startup import initialize_openai_clients
from app.core.text_generations.openai_text_generation_client import OpenAITextGenerationClient
from app.core.embeddings.openai_embedding_client import OpenAIEmbeddingClient
from app.core.embeddings.openai_embedding_service import OpenAIEmbeddingService


logger = get_logger(__name__)


async def main():
    """Main function with proper cleanup."""
    logger.info("Starting SQS Worker")

    try:
        # Initialize components
        SQSQueueClient.create_client()
        sqs_client = SQSQueueClient.get_client()
        queue_service = SQSQueueService(client=sqs_client)
        # Initialize OpenAI clients
        initialize_openai_clients()        
        
        text_generation_client = OpenAITextGenerationClient.get_client()
        embedding_client = OpenAIEmbeddingClient.get_client()
        embedding_service = OpenAIEmbeddingService(embedding_client)
        text_generation_service = OpenAITextGenerationService(
            client=text_generation_client,
            embedding_service=embedding_service
        )
        transcription_service = DeepgramTranscriptionService(text_generation_service)
        s3_service = S3Service()
        transcription_handler = TranscriptionHandler(
            queue_service=queue_service,
            request_queue_url=settings.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL,
            result_queue_url=settings.TRANSCRIBE_AND_SUMMARIZE_RESULTS_QUEUE_URL,
            transcription_service=transcription_service,
            s3_service=s3_service,
            s3_bucket_name=settings.S3_TRANSCRIBE_AND_SUMMARIZE_RESULTS_BUCKET,
        )
        router = MessageRouter()
        router.register_transcription_processor(processor=transcription_handler.process_request)

        transcription_processor = MessageProcessor(
            queue_service=queue_service,
            router=router,
            queue_url=settings.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL,
            max_messages=SQSWorkerConstants.MAX_MESSAGES,
            wait_time_seconds=SQSWorkerConstants.WAIT_TIME_SECONDS,
            visibility_timeout=SQSWorkerConstants.VISIBILITY_TIMEOUT,
            polling_interval=SQSWorkerConstants.POLLING_INTERVAL,
            auto_delete=True,
        )

        logger.info(f"Starting processor for: {settings.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL}")

        # Start processor and wait for it to complete
        await transcription_processor.start()

        # Wait for the processor task to finish (it handles KeyboardInterrupt)
        if transcription_processor._task:
            await transcription_processor._task

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