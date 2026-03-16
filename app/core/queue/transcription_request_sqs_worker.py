"""
Production SQS Worker with proper cleanup and shutdown.
"""

import asyncio

from app.core.ally_core import AllyCoreClient, AllyCoreService
from app.core.config import settings
from app.core.constants import SQSWorkerConstants
from app.core.queue.message_processor import MessageProcessor
from app.core.queue.sqs_queue_client import SQSQueueClient
from app.core.queue.sqs_queue_service import SQSQueueService
from app.core.queue.transcription_request_handler import TranscriptionRequestHandler, TranscriptionServiceProvider
from app.core.embeddings.openai_embedding_client import OpenAIEmbeddingClient
from app.core.embeddings.openai_embedding_service import OpenAIEmbeddingService
from app.core.text_generations.openai_text_generation_client import (
    OpenAITextGenerationClient,
)
from app.core.text_generations.openai_text_generation_service import (
    OpenAITextGenerationService,
)

from app.core.transcriptions.services import (
    DeepgramTranscriptionService,
    OpenAITranscriptionService,
    SarvamTranscriptionService,
)

from app.utils.logger import get_logger
from app.utils.startup import initialize_openai_clients

logger = get_logger(__name__)

def create_transcription_service(provider: str | None = None):
    """
    Create a transcription service based on the specified provider.

    Args:
        provider (str, optional): Provider to use ('openai', 'deepgram', 'sarvam').
            If None, will use settings.TRANSCRIPTION_PROVIDER.

    Returns:
        The transcription service instance

    Raises:
        ValueError: If provider is not supported or required API keys are missing
    """
    provider_str = provider
    if provider_str is None:
        provider_str = settings.TRANSCRIPTION.PROVIDER.lower()

    try:
        provider_enum = TranscriptionServiceProvider[provider_str.upper()]
    except KeyError:
        raise ValueError(
            f"Unsupported transcription provider: {provider_str}. Supported providers: "
            "'openai', 'deepgram', 'sarvam'"
        )

    logger.info(f"Using {provider_str} as transcription service provider.")

    # 3. Compare against Enum members
    if provider_enum == TranscriptionServiceProvider.OPENAI:
        if not settings.OPENAI.API_KEY:
            raise ValueError(
                "OPENAI__API_KEY is required in settings for OpenAI provider"
            )
        return OpenAITranscriptionService()

    elif provider_enum == TranscriptionServiceProvider.DEEPGRAM:
        if not settings.DEEPGRAM.API_KEY:
            raise ValueError(
                "DEEPGRAM__API_KEY is required in settings for Deepgram provider"
            )
        return DeepgramTranscriptionService()

    elif provider_enum == TranscriptionServiceProvider.SARVAM:
        if not settings.SARVAM.API_KEY:
            raise ValueError(
                "SARVAM__API_KEY is required in settings for Sarvam provider"
            )
        return SarvamTranscriptionService()

async def main():
    """Main function with proper cleanup."""
    logger.info("Starting SQS Worker")

    try:
        # Initialize components
        SQSQueueClient.create_client()
        queue_service = SQSQueueService(client=SQSQueueClient.get_client())

        await AllyCoreClient.create_client()
        ally_core_service = AllyCoreService(AllyCoreClient.get_client())

        # Initialize OpenAI clients
        initialize_openai_clients()

        # Select transcription service
        selected_transcription_service = create_transcription_service()

        text_generation_service = OpenAITextGenerationService(
            client=OpenAITextGenerationClient.get_client(),
            embedding_service=OpenAIEmbeddingService(
                OpenAIEmbeddingClient.get_client()
            ),
        )

        # Pass text_generation_service to the handler
        transcription_request_handler = TranscriptionRequestHandler(
            ally_core_service=ally_core_service,
            text_generation_service=text_generation_service,
            transcription_service=selected_transcription_service

        )

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
        await transcription_request_processor.start()

        # Wait for the processor task to finish (it handles KeyboardInterrupt)
        if transcription_request_processor._task:
            await transcription_request_processor._task

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
