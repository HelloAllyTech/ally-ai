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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.core.transcriptions.utils.exceptions import TranscriptionFailedException
from app.core.transcriptions.utils.logger import get_logger
from app.core.transcriptions.utils.phi_events import PHIEvents
from app.core.transcriptions.utils.phi_logger import PHILogEvent, phi_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionAttempt:
    """One provider attempt in a fallback run.

    Serialised into the `sttAttempts` trail forwarded to ally-be so it can
    record, per pipeline attempt, which providers were tried in order, whether
    each produced a usable transcript, and (on failure) why not.
    """

    provider: str
    ok: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """camelCase-free per-provider record; ally-be's DTO field names match."""
        record: Dict[str, object] = {"provider": self.provider, "ok": self.ok}
        if self.error is not None:
            record["error"] = self.error
        return record


class FallbackTranscriptionService:
    """Try each provider in order; fail over on error or empty transcript.

    After each `transcribe_audio_from_url` call, the run's per-provider trail is
    exposed on `last_attempts` (list of :class:`TranscriptionAttempt`) and the
    provider that actually produced the transcript on `last_succeeded_provider`
    (None if every provider failed). The handler reads these to enrich the
    ally-be callback. The `(chat_id, text)` return contract is unchanged, so
    this stays a drop-in replacement for a single concrete service.
    """

    def __init__(
        self,
        services: List[Tuple[str, object]],
        per_provider_timeout_seconds: Optional[int] = None,
        ration_by_chat: bool = False,
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
        # When True, the PRIMARY provider is chosen per session (rotate the chain
        # by chat_id) instead of always leading with services[0]. This rations
        # traffic evenly across providers so per-provider success/failure rates
        # are comparable rather than dominated by whichever is hard-coded first.
        # The fallback chain (the remaining providers, in order) is preserved.
        self.ration_by_chat = ration_by_chat
        # Per-run signals, refreshed at the start of every call so the handler
        # never reads a stale trail from a previous chat.
        self.last_attempts: List[TranscriptionAttempt] = []
        self.last_succeeded_provider: Optional[str] = None

    async def transcribe_audio_from_url(
        self,
        audio_url: str,
        chat_id: int,
        sample_rate: int = 8000,
        is_linear16_encoded: bool = False,
    ) -> Tuple[int, str]:
        last_exception: Optional[Exception] = None
        total = len(self.services)

        # Reset per-run signals so a stale trail from a prior chat can't leak
        # into this callback.
        self.last_attempts = []
        self.last_succeeded_provider = None

        # Deterministic per-session provider order. chat_id % total is stable
        # across processes/replicas (unlike hash(), which is salted), so the
        # same chat always starts on the same provider and traffic spreads
        # evenly. Disabled → the configured order is used unchanged.
        ordered = self.services
        if self.ration_by_chat and total > 1:
            try:
                start = int(chat_id) % total
            except (TypeError, ValueError):
                start = 0
            ordered = self.services[start:] + self.services[:start]

        for position, (name, service) in enumerate(ordered):
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

                # Success: record it and remember which provider won.
                self.last_attempts.append(TranscriptionAttempt(provider=name, ok=True))
                self.last_succeeded_provider = name

                if position > 0:
                    logger.warning(
                        f"Transcription recovered via fallback provider "
                        f"'{name}' (#{position + 1}/{total}) for chat_id={chat_id}"
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
                                "provider_index": position,
                                "component": "FallbackTranscriptionService",
                                "method": "transcribe_audio_from_url",
                            },
                        )
                    )
                return cid, text

            except Exception as e:
                last_exception = e
                is_last = position == total - 1
                self.last_attempts.append(
                    TranscriptionAttempt(
                        provider=name, ok=False, error=f"{type(e).__name__}: {e}"
                    )
                )
                logger.error(
                    f"Transcription provider '{name}' "
                    f"(#{position + 1}/{total}) failed for chat_id={chat_id}: "
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
                            "provider_index": position,
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
