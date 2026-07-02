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
    async def test_trail_records_recovery_via_fallback(self):
        # deepgram fails, sarvam recovers → trail shows both, succeeded=sarvam.
        primary = _service(side_effect=Exception("deepgram down"))
        secondary = _service(return_value=(2, "recovered"))
        fb = FallbackTranscriptionService(
            [("deepgram", primary), ("sarvam", secondary)]
        )

        await fb.transcribe_audio_from_url(
            audio_url="http://x", chat_id=2, sample_rate=8000
        )

        assert fb.last_succeeded_provider == "sarvam"
        trail = [a.to_dict() for a in fb.last_attempts]
        assert trail[0]["provider"] == "deepgram"
        assert trail[0]["ok"] is False
        assert "error" in trail[0]
        assert trail[1] == {"provider": "sarvam", "ok": True}

    @pytest.mark.asyncio
    async def test_trail_first_try_success(self):
        primary = _service(return_value=(1, "primary text"))
        secondary = _service(return_value=(1, "secondary text"))
        fb = FallbackTranscriptionService(
            [("deepgram", primary), ("sarvam", secondary)]
        )

        await fb.transcribe_audio_from_url(
            audio_url="http://x", chat_id=1, sample_rate=8000
        )

        assert fb.last_succeeded_provider == "deepgram"
        assert [a.to_dict() for a in fb.last_attempts] == [
            {"provider": "deepgram", "ok": True}
        ]

    @pytest.mark.asyncio
    async def test_trail_all_fail_has_no_succeeded_provider(self):
        primary = _service(side_effect=Exception("down"))
        secondary = _service(return_value=(4, ""))  # empty == failure
        fb = FallbackTranscriptionService(
            [("deepgram", primary), ("sarvam", secondary)]
        )

        with pytest.raises(TranscriptionFailedException):
            await fb.transcribe_audio_from_url(
                audio_url="http://x", chat_id=4, sample_rate=8000
            )

        assert fb.last_succeeded_provider is None
        trail = [a.to_dict() for a in fb.last_attempts]
        assert [t["provider"] for t in trail] == ["deepgram", "sarvam"]
        assert all(t["ok"] is False for t in trail)

    @pytest.mark.asyncio
    async def test_trail_is_reset_between_runs(self):
        # A prior run's trail must not leak into the next call's callback.
        primary = _service(return_value=(1, "text"))
        fb = FallbackTranscriptionService([("deepgram", primary)])

        await fb.transcribe_audio_from_url(
            audio_url="http://x", chat_id=1, sample_rate=8000
        )
        assert len(fb.last_attempts) == 1

        await fb.transcribe_audio_from_url(
            audio_url="http://y", chat_id=2, sample_rate=8000
        )
        assert len(fb.last_attempts) == 1  # reset, not accumulated

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
    async def test_rationing_rotates_primary_by_chat_id(self):
        # 3 providers, rationing on: chat_id % 3 picks the primary, rest follow
        # in order. Each provider succeeds, so the chosen primary is the winner.
        def chain():
            return [
                ("deepgram", _service(return_value=(0, "d"))),
                ("sarvam", _service(return_value=(0, "s"))),
                ("openai", _service(return_value=(0, "o"))),
            ]

        # chat_id 0 -> deepgram, 1 -> sarvam, 2 -> openai, 3 -> deepgram ...
        for chat_id, expected in [
            (0, "deepgram"),
            (1, "sarvam"),
            (2, "openai"),
            (3, "deepgram"),
        ]:
            fb = FallbackTranscriptionService(chain(), ration_by_chat=True)
            await fb.transcribe_audio_from_url(
                audio_url="http://x", chat_id=chat_id, sample_rate=8000
            )
            assert fb.last_succeeded_provider == expected
            # exactly one attempt (the assigned primary succeeded)
            assert [a.provider for a in fb.last_attempts] == [expected]

    @pytest.mark.asyncio
    async def test_rationing_preserves_fallback_chain(self):
        # chat_id 1 => primary sarvam; sarvam fails, so it should fall over to
        # the remaining chain in rotated order: openai, then deepgram.
        deepgram = _service(return_value=(0, "d"))
        sarvam = _service(side_effect=Exception("sarvam down"))
        openai = _service(return_value=(0, "o"))
        fb = FallbackTranscriptionService(
            [("deepgram", deepgram), ("sarvam", sarvam), ("openai", openai)],
            ration_by_chat=True,
        )

        await fb.transcribe_audio_from_url(
            audio_url="http://x", chat_id=1, sample_rate=8000
        )

        # tried sarvam (fail) -> openai (ok); deepgram never reached
        assert [a.provider for a in fb.last_attempts] == ["sarvam", "openai"]
        assert fb.last_succeeded_provider == "openai"
        deepgram.transcribe_audio_from_url.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rationing_off_keeps_configured_order(self):
        primary = _service(return_value=(0, "d"))
        secondary = _service(return_value=(0, "s"))
        fb = FallbackTranscriptionService(
            [("deepgram", primary), ("sarvam", secondary)], ration_by_chat=False
        )

        await fb.transcribe_audio_from_url(
            audio_url="http://x", chat_id=1, sample_rate=8000
        )
        # chat_id 1 would rotate to sarvam if rationing were on; it's off, so
        # deepgram still leads.
        assert fb.last_succeeded_provider == "deepgram"

    @pytest.mark.asyncio
    async def test_forwards_arguments_to_provider(self):
        primary = _service(return_value=(5, "ok"))
        fb = FallbackTranscriptionService([("deepgram", primary)])

        await fb.transcribe_audio_from_url(
            audio_url="http://audio",
            chat_id=5,
            sample_rate=16000,
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
