"""Tests for transcription provider chain resolution + placeholder skipping."""

from types import SimpleNamespace

import pytest

import app.core.queue.transcription_request_sqs_worker as worker
from app.core.transcriptions.services import FallbackTranscriptionService


class _FakeService:
    def __init__(self):
        pass


@pytest.fixture(autouse=True)
def patch_services(monkeypatch):
    # Avoid constructing real SDK clients; we only care about selection logic.
    monkeypatch.setattr(worker, "DeepgramTranscriptionService", _FakeService)
    monkeypatch.setattr(worker, "OpenAITranscriptionService", _FakeService)
    monkeypatch.setattr(worker, "SarvamTranscriptionService", _FakeService)


def _set(monkeypatch, *, providers, deepgram="dg-real-key", sarvam="sv-real-key",
         openai="sk-real-key", provider="deepgram"):
    monkeypatch.setattr(
        worker.settings, "TRANSCRIPTION",
        SimpleNamespace(PROVIDERS=providers, PROVIDER=provider,
                        PER_PROVIDER_TIMEOUT_SECONDS=None),
    )
    monkeypatch.setattr(worker.settings, "DEEPGRAM", SimpleNamespace(API_KEY=deepgram))
    monkeypatch.setattr(worker.settings, "SARVAM", SimpleNamespace(API_KEY=sarvam))
    monkeypatch.setattr(worker.settings, "OPENAI", SimpleNamespace(API_KEY=openai))


class TestPlaceholderKey:
    @pytest.mark.parametrize(
        "value",
        ["", "   ", None, "fill-in-your-key", "FILL-IN", "your-key-here",
         "changeme", "placeholder", "<your-key>", "TODO"],
    )
    def test_detects_missing_or_placeholder(self, value):
        assert worker._is_missing_or_placeholder_key(value) is True

    @pytest.mark.parametrize(
        "value",
        ["sk-abc123", "test-deepgram-key", "test-sarvam-key",
         "a1b2c3d4e5f6", "saaras-real-key"],
    )
    def test_accepts_real_keys(self, value):
        assert worker._is_missing_or_placeholder_key(value) is False


class TestCreateTranscriptionService:
    def test_full_chain_when_all_keys_present(self, monkeypatch):
        _set(monkeypatch, providers="deepgram,sarvam,openai")
        svc = worker.create_transcription_service()
        assert isinstance(svc, FallbackTranscriptionService)
        assert [name for name, _ in svc.services] == ["deepgram", "sarvam", "openai"]

    def test_placeholder_sarvam_is_skipped(self, monkeypatch):
        # The exact scenario in dev: Sarvam key is a placeholder -> chain
        # degrades to deepgram -> openai with no Sarvam attempt.
        _set(monkeypatch, providers="deepgram,sarvam,openai",
             sarvam="fill-in-your-sarvam-key")
        svc = worker.create_transcription_service()
        assert isinstance(svc, FallbackTranscriptionService)
        assert [name for name, _ in svc.services] == ["deepgram", "openai"]

    def test_single_remaining_provider_returns_bare_service(self, monkeypatch):
        _set(monkeypatch, providers="deepgram,sarvam,openai",
             sarvam="fill-x", openai="your-openai-key")
        svc = worker.create_transcription_service()
        # Only deepgram survives -> returned directly, not wrapped.
        assert isinstance(svc, _FakeService)

    def test_raises_when_no_usable_provider(self, monkeypatch):
        _set(monkeypatch, providers="deepgram,sarvam,openai",
             deepgram="", sarvam="fill-x", openai="changeme")
        with pytest.raises(ValueError):
            worker.create_transcription_service()

    def test_dedups_provider_order(self, monkeypatch):
        _set(monkeypatch, providers="deepgram,deepgram,openai")
        svc = worker.create_transcription_service()
        assert [name for name, _ in svc.services] == ["deepgram", "openai"]
