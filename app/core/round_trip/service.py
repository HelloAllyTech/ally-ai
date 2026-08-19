"""Round-trip WER service (PRD §6.1, FR2): synthesize the agent's own text
with TTS, transcribe it back with ASR, compare. Isolates TTS pronunciation —
the reference is the LLM's text, so no human transcript is needed.

Stateless like the judges: utterances in → per-utterance error rates out;
ally-be samples the turns, picks the provider, and persists the result.

Providers (v1): 'sarvam' (Indic — bulbul TTS + saarika STT via REST) and
'openai' (tts-1 + whisper-1). Sessions whose live TTS provider isn't
supported here fall back to a language-appropriate default; the response
reports the provider actually used so the caller can record the caveat.
Per-utterance failures are skipped, never fatal — the metric degrades to a
smaller sample (n_measured < n_requested).
"""

from __future__ import annotations

import asyncio
import base64
from typing import List, Literal, Optional, TypedDict

import httpx

from app.core.config import settings
from app.core.round_trip.wer import Unit, error_rate_pct
from app.utils.logger import get_logger

logger = get_logger(__name__)

Provider = Literal["sarvam", "openai"]

SARVAM_BASE = "https://api.sarvam.ai"
SARVAM_TTS_MODEL = "bulbul:v2"
SARVAM_ASR_MODEL = "saarika:v2.5"
OPENAI_TTS_MODEL = "tts-1"
OPENAI_TTS_VOICE = "alloy"
OPENAI_ASR_MODEL = "whisper-1"

# Ceiling on concurrent TTS+ASR round trips across this process. The caller
# judges several sessions at once and each session fans out over its sampled
# utterances, so without a shared bound the two multiply into a vendor rate
# limit. Sized for throughput, not for one request: a single request's five
# utterances still finish in two waves.
_ROUND_TRIP_SLOTS = asyncio.Semaphore(6)

# Sarvam supports these BCP-47 codes for TTS/STT.
SARVAM_LANGUAGES = {
    "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN",
    "pa-IN", "ta-IN", "te-IN", "gu-IN", "en-IN",
}


class Utterance(TypedDict):
    turn_index: int
    text: str


class UtteranceResult(TypedDict):
    turn_index: int
    error_pct: float
    hypothesis: str


def _sarvam_language(language: str) -> str:
    """Normalize to a Sarvam code (bare 'ta' -> 'ta-IN'; 'or' -> 'od-IN')."""
    lang = (language or "en").strip()
    base = lang.split("-")[0].lower()
    if base == "or":
        base = "od"
    if base == "en":
        return "en-IN"
    return f"{base}-IN"


def _has_key(value: Optional[str]) -> bool:
    return bool(value) and "your" not in (value or "").lower()


def resolve_provider(requested: Optional[str], language: str) -> Provider:
    """Honor the session's TTS provider when we can round-trip it; otherwise
    fall back to the best available provider for the language."""
    if requested == "sarvam" and _has_key(settings.SARVAM.API_KEY):
        return "sarvam"
    if requested == "openai" and _has_key(settings.OPENAI.API_KEY):
        return "openai"
    # Unsupported (elevenlabs/google/hume/deepgram) or missing key → default:
    # Sarvam for its covered Indic languages, OpenAI otherwise.
    if (
        _sarvam_language(language) in SARVAM_LANGUAGES
        and _sarvam_language(language) != "en-IN"
        and _has_key(settings.SARVAM.API_KEY)
    ):
        return "sarvam"
    if _has_key(settings.OPENAI.API_KEY):
        return "openai"
    if _has_key(settings.SARVAM.API_KEY):
        return "sarvam"
    raise RuntimeError("no round-trip provider available (keys missing)")


async def _sarvam_tts(client: httpx.AsyncClient, text: str, language: str) -> bytes:
    resp = await client.post(
        f"{SARVAM_BASE}/text-to-speech",
        headers={"api-subscription-key": settings.SARVAM.API_KEY or ""},
        json={
            "text": text,
            "target_language_code": _sarvam_language(language),
            "model": SARVAM_TTS_MODEL,
        },
        timeout=60,
    )
    resp.raise_for_status()
    audios = resp.json().get("audios") or []
    if not audios:
        raise RuntimeError("sarvam TTS returned no audio")
    return base64.b64decode(audios[0])


async def _sarvam_asr(client: httpx.AsyncClient, audio: bytes, language: str) -> str:
    resp = await client.post(
        f"{SARVAM_BASE}/speech-to-text",
        headers={"api-subscription-key": settings.SARVAM.API_KEY or ""},
        files={"file": ("utterance.wav", audio, "audio/wav")},
        data={
            "model": SARVAM_ASR_MODEL,
            "language_code": _sarvam_language(language),
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("transcript") or ""


def _openai_round_trip_sync(text: str, language: str) -> str:
    """One utterance through OpenAI TTS then Whisper (sync SDK, run in a
    thread). Returns the ASR hypothesis."""
    from openai import OpenAI  # local import; client is cheap to build

    client = OpenAI(api_key=settings.OPENAI.API_KEY)
    speech = client.audio.speech.create(
        model=OPENAI_TTS_MODEL, voice=OPENAI_TTS_VOICE, input=text
    )
    audio = speech.content
    transcription = client.audio.transcriptions.create(
        model=OPENAI_ASR_MODEL,
        file=("utterance.mp3", audio),
        language=(language or "en").split("-")[0].lower() or "en",
    )
    return getattr(transcription, "text", "") or ""


async def _round_trip_one(
    client: httpx.AsyncClient, provider: Provider, text: str, language: str
) -> str:
    if provider == "sarvam":
        audio = await _sarvam_tts(client, text, language)
        return await _sarvam_asr(client, audio, language)
    return await asyncio.to_thread(_openai_round_trip_sync, text, language)


async def run_round_trip(
    utterances: List[Utterance],
    language: str,
    requested_provider: Optional[str],
    unit: Unit,
) -> dict:
    """Round-trip every utterance CONCURRENTLY, not one after another.

    Sequentially this was arithmetically unable to finish: each utterance is a
    TTS call plus an ASR call, both allowed 60s, so five of them could take 600s
    against a caller that waits 180s. 37% of sessions timed out — the caller
    recorded "not measured" while this kept spending vendor calls nobody was
    waiting for any more. Concurrently the worst case is one utterance's 120s,
    inside the caller's budget.

    Bounded by a process-wide semaphore rather than let loose: the caller judges
    several sessions at once, so unbounded fan-out here multiplies into the
    vendor's rate limit, which is the failure this change exists to stop making.
    """
    provider = resolve_provider(requested_provider, language)

    async def one(u: Utterance, client: httpx.AsyncClient) -> Optional[UtteranceResult]:
        text = (u.get("text") or "").strip()
        if not text:
            return None
        try:
            async with _ROUND_TRIP_SLOTS:
                hypothesis = await _round_trip_one(client, provider, text, language)
            return {
                "turn_index": int(u["turn_index"]),
                "error_pct": error_rate_pct(text, hypothesis, unit),
                "hypothesis": hypothesis,
            }
        except Exception as e:  # noqa: BLE001 — per-utterance tolerance
            logger.warning(f"round-trip failed for turn {u.get('turn_index')}: {e}")
            return None

    async with httpx.AsyncClient() as client:
        settled = await asyncio.gather(*(one(u, client) for u in utterances))
    # gather preserves order, so the response still reads in turn order.
    results: List[UtteranceResult] = [r for r in settled if r is not None]
    avg = (
        round(sum(r["error_pct"] for r in results) / len(results), 2)
        if results
        else None
    )
    return {
        "provider_used": provider,
        "unit": unit,
        "n_requested": len(utterances),
        "n_measured": len(results),
        "avg_error_pct": avg,
        "per_utterance": results,
    }
