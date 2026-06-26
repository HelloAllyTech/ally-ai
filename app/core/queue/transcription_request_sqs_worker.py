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
    FallbackTranscriptionService,
    OpenAITranscriptionService,
    SarvamTranscriptionService,
)

from app.utils.logger import get_logger
from app.utils.startup import initialize_openai_clients

logger = get_logger(__name__)

# Common stand-in values seen in committed/example .env files. A provider
# whose key matches one of these is treated as unconfigured and skipped at
# startup (rather than failing on every request with a 403), so the chain
# degrades cleanly until a real key is set. Matched case-insensitively as a
# prefix; real keys (e.g. OpenAI 'sk-…', Deepgram hex, the test suite's
# 'test-…' keys) are unaffected.
_PLACEHOLDER_KEY_PREFIXES = (
    "fill",
    "your",
    "changeme",
    "change-me",
    "change_me",
    "placeholder",
    "todo",
    "dummy",
    "replace",
    "xxx",
    "<",
)


def _is_missing_or_placeholder_key(value: str | None) -> bool:
    """True if a key is empty or an obvious stand-in (not a real credential)."""
    if not value or not value.strip():
        return True
    return value.strip().lower().startswith(_PLACEHOLDER_KEY_PREFIXES)


def _build_single_service(provider_str: str):
    """
    Build one concrete transcription service for a provider name.

    Raises:
        ValueError: If the provider is unsupported or its API key is missing
            or an obvious placeholder.
    """
    try:
        provider_enum = TranscriptionServiceProvider[provider_str.upper()]
    except KeyError:
        raise ValueError(
            f"Unsupported transcription provider: {provider_str}. Supported providers: "
            "'openai', 'deepgram', 'sarvam'"
        )

    if provider_enum == TranscriptionServiceProvider.OPENAI:
        if _is_missing_or_placeholder_key(settings.OPENAI.API_KEY):
            raise ValueError(
                "OPENAI__API_KEY is missing or a placeholder for OpenAI provider"
            )
        return OpenAITranscriptionService()

    elif provider_enum == TranscriptionServiceProvider.DEEPGRAM:
        if _is_missing_or_placeholder_key(settings.DEEPGRAM.API_KEY):
            raise ValueError(
                "DEEPGRAM__API_KEY is missing or a placeholder for Deepgram provider"
            )
        return DeepgramTranscriptionService()

    elif provider_enum == TranscriptionServiceProvider.SARVAM:
        if _is_missing_or_placeholder_key(settings.SARVAM.API_KEY):
            raise ValueError(
                "SARVAM__API_KEY is missing or a placeholder for Sarvam provider"
            )
        return SarvamTranscriptionService()


def _resolve_provider_order() -> list[str]:
    """
    Resolve the ordered provider chain from settings.

    Uses TRANSCRIPTION.PROVIDERS (comma-separated) when set, otherwise falls
    back to the single TRANSCRIPTION.PROVIDER. De-duplicates while preserving
    order so a chain like "deepgram,deepgram,sarvam" collapses sensibly.
    """
    raw = getattr(settings.TRANSCRIPTION, "PROVIDERS", None)
    if raw:
        names = [p.strip().lower() for p in raw.split(",") if p.strip()]
    else:
        names = [settings.TRANSCRIPTION.PROVIDER.lower()]

    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def create_transcription_service(provider: str | None = None):
    """
    Create the transcription service for the worker.

    When a single provider is configured (default), returns that concrete
    service. When TRANSCRIPTION.PROVIDERS lists more than one, returns a
    FallbackTranscriptionService that tries them in order, failing over on
    error or an empty transcript.

    Args:
        provider (str, optional): Force a single provider ('openai',
            'deepgram', 'sarvam'), bypassing the configured chain. Mainly for
            tests. If None, resolves from settings.

    Returns:
        A transcription service exposing transcribe_audio_from_url(...).

    Raises:
        ValueError: If no usable provider can be built (all unsupported or
            missing API keys).
    """
    if provider is not None:
        logger.info(f"Using {provider} as transcription service provider.")
        return _build_single_service(provider.lower())

    provider_names = _resolve_provider_order()

    # Build each provider, skipping (with a loud log) any that can't be
    # constructed — e.g. a fallback whose API key isn't configured — so a
    # half-configured chain still runs on whatever providers are available.
    built: list[tuple[str, object]] = []
    for name in provider_names:
        try:
            built.append((name, _build_single_service(name)))
        except ValueError as e:
            logger.error(
                f"Skipping transcription provider '{name}': {e}"
            )

    if not built:
        raise ValueError(
            "No usable transcription provider could be initialised from "
            f"{provider_names}. Check TRANSCRIPTION settings and API keys."
        )

    if len(built) == 1:
        name, service = built[0]
        logger.info(f"Using {name} as transcription service provider.")
        return service

    chain = ", ".join(name for name, _ in built)
    logger.info(f"Using transcription provider fallback chain: {chain}")
    return FallbackTranscriptionService(
        built,
        per_provider_timeout_seconds=getattr(
            settings.TRANSCRIPTION, "PER_PROVIDER_TIMEOUT_SECONDS", None
        ),
    )

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
        if 'queue_service' in locals():
            await queue_service.close()
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
