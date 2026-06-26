"""Tests for FallbackTranscriptionService."""

from unittest.mock import AsyncMock

import pytest

from app.core.transcriptions.services.fallback_service import (
    FallbackTranscriptionService,
)
from app.core.transcriptions.utils.exceptions import TranscriptionFailedException


def _service(return_value=None, side_effect=None):
    """Build a fake provider with a mocked transcribe_audio_from_url."""
    svc = AsyncMock()
    svc.transcribe_audio_from_url = AsyncMock(
        return_value=return_value, side_effect=side_effect
    )
    return svc


class TestFallbackTranscriptionService:
    def test_requires_at_least_one_provider(self):
        with pytest.raises(ValueError):
            FallbackTranscriptionService([])

    @pytest.mark.asyncio
    async def test_primary_success_skips_fallbacks(self):
        primary = _service(return_value=(1, "primary text"))
        secondary = _service(return_value=(1, "secondary text"))
        fb = FallbackTranscriptionService(
            [("deepgram", primary), ("sarvam", secondary)]
        )

        cid, text = await fb.transcribe_audio_from_url(
            audio_url="http://x", chat_id=1, sample_rate=8000
        )

        assert (cid, text) == (1, "primary text")
        primary.transcribe_audio_from_url.assert_awaited_once()
        secondary.transcribe_audio_from_url.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_over_on_exception(self):
        primary = _service(side_effect=Exception("deepgram down"))
        secondary = _service(return_value=(2, "recovered"))
        fb = FallbackTranscriptionService(
            [("deepgram", primary), ("sarvam", secondary)]
        )

        cid, text = await fb.transcribe_audio_from_url(
            audio_url="http://x", chat_id=2, sample_rate=8000
        )

        assert (cid, text) == (2, "recovered")
        primary.transcribe_audio_from_url.assert_awaited_once()
        secondary.transcribe_audio_from_url.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_falls_over_on_empty_result(self):
        # An empty/whitespace transcript from the primary must trigger failover,
        # not be accepted as success.
        primary = _service(return_value=(3, "   "))
        secondary = _service(return_value=(3, "real transcript"))
        fb = FallbackTranscriptionService(
            [("deepgram", primary), ("sarvam", secondary)]
        )

        cid, text = await fb.transcribe_audio_from_url(
            audio_url="http://x", chat_id=3, sample_rate=8000
        )

        assert text == "real transcript"
        secondary.transcribe_audio_from_url.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_all_providers_fail(self):
        primary = _service(side_effect=Exception("down"))
        secondary = _service(return_value=(4, ""))  # empty == failure
        fb = FallbackTranscriptionService(
            [("deepgram", primary), ("sarvam", secondary)]
        )

        with pytest.raises(TranscriptionFailedException):
            await fb.transcribe_audio_from_url(
                audio_url="http://x", chat_id=4, sample_rate=8000
            )

        primary.transcribe_audio_from_url.assert_awaited_once()
        secondary.transcribe_audio_from_url.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_forwards_arguments_to_provider(self):
        primary = _service(return_value=(5, "ok"))
        fb = FallbackTranscriptionService([("deepgram", primary)])

        await fb.transcribe_audio_from_url(
            audio_url="http://audio", chat_id=5, sample_rate=16000,
            is_linear16_encoded=True,
        )

        kwargs = primary.transcribe_audio_from_url.await_args.kwargs
        assert kwargs["audio_url"] == "http://audio"
        assert kwargs["chat_id"] == 5
        assert kwargs["sample_rate"] == 16000
        assert kwargs["is_linear16_encoded"] is True

    @pytest.mark.asyncio
    async def test_per_provider_timeout_triggers_failover(self):
        async def slow(*args, **kwargs):
            import asyncio

            await asyncio.sleep(5)
            return (6, "too late")

        primary = AsyncMock()
        primary.transcribe_audio_from_url = slow
        secondary = _service(return_value=(6, "fast"))
        fb = FallbackTranscriptionService(
            [("deepgram", primary), ("sarvam", secondary)],
            per_provider_timeout_seconds=1,
        )

        cid, text = await fb.transcribe_audio_from_url(
            audio_url="http://x", chat_id=6, sample_rate=8000
        )

        assert text == "fast"
        secondary.transcribe_audio_from_url.assert_awaited_once()
