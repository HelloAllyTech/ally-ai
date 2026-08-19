"""Round-trip WER fans out over its utterances instead of queueing them.

Sequentially this endpoint could not finish inside the budget its own caller
allowed: every utterance is a TTS call plus an ASR call, each permitted 60s, so
five of them could take 600s while ally-be waited 180s. 37% of judged sessions
recorded "not measured" for that reason alone, and ally-ai went on spending
vendor calls after nobody was listening.

These tests pin the three properties that fix depends on — it runs concurrently,
it keeps turn order, and one bad utterance still does not sink the batch.
"""

import asyncio
import time

import pytest

import app.core.round_trip.service as svc


@pytest.fixture
def fake_round_trip(monkeypatch):
    """Replace the vendor round trip with a fixed delay we can time."""
    calls: list[str] = []

    async def _one(client, provider, text, language):
        calls.append(text)
        await asyncio.sleep(0.3)
        if text == "boom":
            raise RuntimeError("vendor said no")
        return f"{text} (heard)"

    monkeypatch.setattr(svc, "_round_trip_one", _one)
    monkeypatch.setattr(svc, "resolve_provider", lambda requested, language: "sarvam")
    return calls


def _utterances(texts: list[str]) -> list[dict]:
    return [{"turn_index": i, "text": t} for i, t in enumerate(texts)]


@pytest.mark.asyncio
async def test_utterances_run_concurrently(fake_round_trip):
    # Four measurable utterances at 0.3s each: ~1.2s queued, ~0.3s in parallel.
    # The threshold sits between the two so it fails if the loop ever goes back
    # to awaiting one at a time.
    started = time.monotonic()
    await svc.run_round_trip(
        _utterances(["one two", "three four", "five six", "seven eight"]),
        "ta-IN",
        None,
        "wer",
    )
    assert time.monotonic() - started < 0.6


@pytest.mark.asyncio
async def test_one_bad_utterance_does_not_sink_the_batch(fake_round_trip):
    # Per-utterance tolerance is the whole reason the metric can degrade to a
    # smaller sample rather than to nothing.
    result = await svc.run_round_trip(
        _utterances(["one two", "boom", "five six"]), "ta-IN", None, "wer"
    )

    assert result["n_requested"] == 3
    assert result["n_measured"] == 2
    assert result["avg_error_pct"] is not None


@pytest.mark.asyncio
async def test_blank_utterances_are_never_sent(fake_round_trip):
    await svc.run_round_trip(
        _utterances(["one two", "", "   ", "five six"]), "ta-IN", None, "wer"
    )
    assert fake_round_trip == ["one two", "five six"]


@pytest.mark.asyncio
async def test_results_stay_in_turn_order(fake_round_trip):
    # Concurrency must not reorder the response: the per-utterance rows are read
    # against the transcript by turn index.
    result = await svc.run_round_trip(
        _utterances(["one two", "boom", "five six", "seven eight"]),
        "ta-IN",
        None,
        "wer",
    )
    assert [r["turn_index"] for r in result["per_utterance"]] == [0, 2, 3]


@pytest.mark.asyncio
async def test_nothing_measurable_reports_none_not_zero(fake_round_trip):
    # A zero error rate would read as flawless pronunciation, which is the
    # opposite of what "we could not measure it" means.
    result = await svc.run_round_trip(_utterances(["", "  "]), "ta-IN", None, "wer")

    assert result["n_measured"] == 0
    assert result["avg_error_pct"] is None
