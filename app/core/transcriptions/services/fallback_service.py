"""
Fallback transcription service.

Wraps an ordered list of transcription providers and tries them in turn. A
single STT provider is a single point of failure: a Deepgram outage,
rate-limit, regional blip, or a result with no usable text fails every session
for that window. This wrapper falls over to the next provider on either an
exception OR an empty result, so one provider's bad patch no longer silently
costs a summary.

All concrete services (Deepgram / OpenAI / Sarvam) share the same
`transcribe_audio_from_url(audio_url, chat_id, sample_rate,
is_linear16_encoded) -> (chat_id, str)` interface, so this wrapper is a drop-in
replacement wherever a single service is used today.
"""

import asyncio
from typing import List, Optional, Tuple

from app.core.transcriptions.utils.exceptions import TranscriptionFailedException
from app.core.transcriptions.utils.logger import get_logger
from app.core.transcriptions.utils.phi_events import PHIEvents
from app.core.transcriptions.utils.phi_logger import PHILogEvent, phi_logger

logger = get_logger(__name__)


class FallbackTranscriptionService:
    """Try each provider in order; fail over on error or empty transcript."""

    def __init__(
        self,
        services: List[Tuple[str, object]],
        per_provider_timeout_seconds: Optional[int] = None,
    ) -> None:
        """
        Args:
            services: Ordered list of (provider_name, service) pairs. The first
                is primary; the rest are fallbacks tried in order.
            per_provider_timeout_seconds: Optional hard cap on a single
                provider attempt. The SQS visibility timeout (900s) bounds the
                whole message, so the SUM of provider attempts must stay under
                it or a second worker could pick up the same chat mid-flight.
                Set this when chaining slow providers. None = rely on each
                provider's own internal timeout.
        """
        if not services:
            raise ValueError("FallbackTranscriptionService requires >=1 provider")
        self.services = services
        self.per_provider_timeout_seconds = per_provider_timeout_seconds

    async def transcribe_audio_from_url(
        self,
        audio_url: str,
        chat_id: int,
        sample_rate: int = 8000,
        is_linear16_encoded: bool = False,
    ) -> Tuple[int, str]:
        last_exception: Optional[Exception] = None
        total = len(self.services)

        for index, (name, service) in enumerate(self.services):
            try:
                coro = service.transcribe_audio_from_url(
                    audio_url=audio_url,
                    chat_id=chat_id,
                    sample_rate=sample_rate,
                    is_linear16_encoded=is_linear16_encoded,
                )
                if self.per_provider_timeout_seconds:
                    cid, text = await asyncio.wait_for(
                        coro, timeout=self.per_provider_timeout_seconds
                    )
                else:
                    cid, text = await coro

                # Treat an empty result as a failure worth failing over: a
                # different engine often transcribes audio the first one gave
                # up on, and either way an empty transcript must never be
                # accepted as success.
                if not text or not text.strip():
                    raise TranscriptionFailedException(
                        f"{name} returned an empty transcript"
                    )

                if index > 0:
                    logger.warning(
                        f"Transcription recovered via fallback provider "
                        f"'{name}' (#{index + 1}/{total}) for chat_id={chat_id}"
                    )
                    await phi_logger.log(
                        PHILogEvent(
                            event_type=PHIEvents.DATA_MODIFIED,
                            chat_id=str(chat_id),
                            audit_id=None,
                            details={
                                "message": (
                                    f"Transcription recovered via fallback "
                                    f"provider '{name}'"
                                ),
                                "chat_id": chat_id,
                                "provider": name,
                                "provider_index": index,
                                "component": "FallbackTranscriptionService",
                                "method": "transcribe_audio_from_url",
                            },
                        )
                    )
                return cid, text

            except Exception as e:
                last_exception = e
                is_last = index == total - 1
                logger.error(
                    f"Transcription provider '{name}' "
                    f"(#{index + 1}/{total}) failed for chat_id={chat_id}: "
                    f"{type(e).__name__}: {e}"
                    + ("" if is_last else "; falling over to next provider")
                )
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=str(chat_id),
                        audit_id=None,
                        details={
                            "error": (
                                f"Transcription provider '{name}' failed: "
                                f"{type(e).__name__}"
                            ),
                            "chat_id": chat_id,
                            "provider": name,
                            "provider_index": index,
                            "is_last_provider": is_last,
                            "exception_type": type(e).__name__,
                            "component": "FallbackTranscriptionService",
                            "method": "transcribe_audio_from_url",
                        },
                    )
                )

        # Every provider failed. Surface the last error so the caller (the
        # handler) reports a TRANSCRIBE-stage failure as usual.
        raise TranscriptionFailedException(
            f"All {total} transcription provider(s) failed for chat_id "
            f"{chat_id}: {last_exception}"
        )
