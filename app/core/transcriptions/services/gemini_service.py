"""
Gemini transcription service for the Scribe pipeline.

Replaces Deepgram as the primary batch STT engine. The audio is downloaded and
split exactly as the other services do (``convert_and_segment_audio_async``),
each segment is sent to Gemini inline, and the model is asked for timestamped
utterances which are rendered in the same ``start-end: text`` line format the
diarization step already consumes — so this is a drop-in member of the
``FallbackTranscriptionService`` chain.

Why inline bytes rather than the Files API: an uploaded file lives on Google's
side until it is deleted, which is one more place PHI can be left behind if the
worker dies mid-run. Inline requests are capped at 20MB, so segments are cut
smaller than the 24MB the other providers use.
"""

import asyncio
import os
import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.transcriptions.utils.audio_converter import (
    convert_and_segment_audio_async,
    get_audio_duration,
)
from app.core.transcriptions.utils.exceptions import TranscriptionFailedException
from app.core.transcriptions.utils.logger import get_logger
from app.core.transcriptions.utils.phi_events import PHIEvents
from app.core.transcriptions.utils.phi_logger import PHILogEvent, phi_logger

logger = get_logger(__name__)

# Gemini rejects inline requests above 20MB; keep a margin for the prompt and
# the base64 overhead of the audio part.
MAX_SEGMENT_SIZE_MB = 14

# Hard cap on one model call. The SQS visibility timeout (900s) bounds the
# whole message, and the fallback chain may still try two more providers after
# this one, so a single segment must never be allowed to eat the budget.
REQUEST_TIMEOUT_SECONDS = 240

# Segments of one recording are transcribed concurrently, bounded so a long
# call does not fan out into dozens of simultaneous Gemini requests.
MAX_CONCURRENT_SEGMENTS = 4

_MIME_BY_EXTENSION = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".mpeg": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
}

TRANSCRIPTION_PROMPT = (
    "You are a verbatim transcription engine for a recorded counselling call.\n"
    "Transcribe everything that is spoken in this audio exactly as it is said, "
    "in the language or languages actually spoken. Do not translate. Keep "
    "code-mixed speech as spoken, and write Indian languages in their native "
    "script.\n"
    "Split the speech into short utterances at natural pauses or speaker "
    "changes. For every utterance give its start and end time in seconds from "
    'the beginning of this audio as decimal numbers (for example "12.40").\n'
    "Do not add speaker labels, commentary, summaries or anything that was not "
    "spoken. If there is no intelligible speech, return an empty list of "
    "segments."
)


class TranscriptSegment(BaseModel):
    """One utterance as the model reports it. Times are strings so the parser
    can accept both plain seconds and clock-style ``MM:SS`` values."""

    start: str = Field(description="Start time in seconds, e.g. '12.40'")
    end: str = Field(description="End time in seconds, e.g. '15.02'")
    text: str = Field(description="The words spoken, verbatim")


class Transcript(BaseModel):
    segments: List[TranscriptSegment] = Field(default_factory=list)


_CLOCK_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")


def parse_timestamp(value) -> Optional[float]:
    """Seconds from ``12.4``, ``"12.4"``, ``"0:12.4"`` or ``"1:02:03.5"``.

    Returns None for anything unparseable so one odd line is dropped instead of
    failing the whole recording.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    text = str(value).strip().rstrip("s").strip()
    if not text:
        return None
    try:
        seconds = float(text)
        return seconds if seconds >= 0 else None
    except ValueError:
        pass
    match = _CLOCK_RE.match(text)
    if not match:
        return None
    hours, minutes, secs = match.groups()
    return (int(hours or 0) * 3600) + (int(minutes) * 60) + float(secs)


def format_segments(segments: List[Tuple[float, float, str]]) -> str:
    """Render ``(start, end, text)`` tuples in the line format the diarization
    step expects — identical to what the OpenAI and Deepgram services emit."""
    return "\n".join(
        f"{start:.2f}-{end:.2f}: {text.strip()}"
        for start, end, text in sorted(segments, key=lambda s: (s[0], s[1]))
        if text and text.strip()
    )


def _mime_type_for(path: str) -> str:
    return _MIME_BY_EXTENSION.get(os.path.splitext(path)[1].lower(), "audio/wav")


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _emit_stt_usage(model: str, audio_seconds: float, usage_metadata=None) -> None:
    """Best-effort batch-STT AI-cost emit. Never raises / blocks transcription."""
    try:
        if not audio_seconds or audio_seconds <= 0:
            return
        from app.core.llm_usage.emitter import emit_ai_usage

        prompt_tokens = int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
        completion_tokens = int(
            getattr(usage_metadata, "candidates_token_count", 0) or 0
        )
        emit_ai_usage(
            "stt",
            "gemini",
            model,
            "transcription",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            audio_ms=int(audio_seconds * 1000),
        )
    except Exception:
        logger.debug("batch STT usage emit skipped (best-effort)")


class GeminiTranscriptionService:
    """
    Gemini transcription service for processing audio files.

    Shares the ``transcribe_audio_from_url(audio_url, chat_id, sample_rate,
    is_linear16_encoded) -> (chat_id, segments_text)`` contract with the other
    services so the worker and the fallback wrapper treat it as just another
    provider.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = api_key or settings.GEMINI.API_KEY
        if not key:
            raise ValueError("GEMINI__API_KEY is not configured")
        # Imported lazily so the module imports (and the other providers keep
        # working) in an environment without the optional SDK.
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=key)
        self._model = model or settings.GEMINI.TRANSCRIPTION_MODEL
        self._config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Transcript,
            # Transcription is a copy task; hidden reasoning only adds latency
            # and billed tokens.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0,
        )

    @property
    def model(self) -> str:
        return self._model

    async def transcribe_audio_from_url(
        self,
        audio_url: str,
        chat_id: int,
        sample_rate: int = 8000,
        is_linear16_encoded: bool = False,
    ) -> Tuple[int, str]:
        """
        Transcribe audio from URL into timestamped segments text.

        Args:
            audio_url (str): URL containing the audio file
            chat_id (int): Chat ID for the transcription session
            sample_rate (int): Expected sample rate of the audio (default: 8000)
            is_linear16_encoded (bool): Whether the audio is headerless linear16
                (s16le) PCM (mobile uploads), which must be decoded as raw PCM.

        Returns:
            Tuple[int, str]: (chat_id, segments_text). An empty string means the
            model heard no intelligible speech; the fallback wrapper treats that
            as a failure worth trying the next provider on.

        Raises:
            TranscriptionFailedException: If transcription fails
        """
        segment_paths: List[str] = []
        try:
            segment_paths = await convert_and_segment_audio_async(
                audio_url=audio_url,
                sample_rate=sample_rate,
                max_segment_size_mb=MAX_SEGMENT_SIZE_MB,
                chat_id=chat_id,
                is_linear16_encoded=is_linear16_encoded,
            )
            logger.info(f"Audio converted into {len(segment_paths)} segments")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Audio converted into {len(segment_paths)} segments",  # noqa: E501
                        "chat_id": chat_id,
                        "audio_url": audio_url,
                        "sample_rate": sample_rate,
                        "segment_count": len(segment_paths),
                        "max_segment_size_mb": MAX_SEGMENT_SIZE_MB,
                        "component": "GeminiTranscriptionService",
                        "method": "transcribe_audio_from_url",
                    },
                )
            )

            segments = await self._transcribe_segments(segment_paths, chat_id)
            segments_text = format_segments(segments)

            if not segments_text:
                logger.warning(
                    f"Gemini returned no intelligible speech for chat_id={chat_id}"
                )

            logger.info(f"Formatted {len(segments)} segments for diarization")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Formatted {len(segments)} segments for diarization",  # noqa: E501
                        "chat_id": chat_id,
                        "segments_count": len(segments),
                        "segments_text_length": len(segments_text),
                        "model": self._model,
                        "component": "GeminiTranscriptionService",
                        "method": "transcribe_audio_from_url",
                    },
                )
            )
            return chat_id, segments_text

        except TranscriptionFailedException:
            raise
        except Exception as e:
            logger.error(
                f"Error transcribing audio from URL for chat_id {chat_id}: "
                f"{type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Error transcribing audio from URL for chat_id {chat_id}: {type(e).__name__}",  # noqa: E501
                        "chat_id": chat_id,
                        "audio_url": audio_url,
                        "sample_rate": sample_rate,
                        "exception_type": type(e).__name__,
                        "component": "GeminiTranscriptionService",
                        "method": "transcribe_audio_from_url",
                    },
                )
            )
            raise TranscriptionFailedException(
                "Error transcribing audio from URL", status_code=_status_code_of(e)
            )
        finally:
            await self._cleanup_segment_files(segment_paths, chat_id)

    async def _transcribe_segments(
        self, segment_paths: List[str], chat_id: int
    ) -> List[Tuple[float, float, str]]:
        """Transcribe every segment concurrently and stitch them on one clock.

        The offset of segment N is the summed duration of segments 0..N-1 —
        measured, not assumed, since the splitter cuts on size rather than time.
        """
        durations = await asyncio.gather(
            *(get_audio_duration(path) for path in segment_paths)
        )
        offsets: List[float] = []
        running = 0.0
        for duration in durations:
            offsets.append(running)
            running += float(duration or 0.0)

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEGMENTS)

        async def _one(index: int) -> List[Tuple[float, float, str]]:
            async with semaphore:
                return await self._transcribe_segment_with_offset(
                    segment_paths[index],
                    index,
                    offsets[index],
                    float(durations[index] or 0.0),
                    chat_id,
                )

        results = await asyncio.gather(
            *(_one(i) for i in range(len(segment_paths))), return_exceptions=True
        )

        merged: List[Tuple[float, float, str]] = []
        for index, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Segment {index} transcription failed: {type(result).__name__}"
                )
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=str(chat_id),
                        audit_id=None,  # Will be set by external service,
                        details={
                            "error": f"Segment {index} transcription failed: {type(result).__name__}",  # noqa: E501
                            "segment_index": index,
                            "exception_type": type(result).__name__,
                            "component": "GeminiTranscriptionService",
                            "method": "_transcribe_segments",
                        },
                    )
                )
                if isinstance(result, TranscriptionFailedException):
                    raise result
                raise TranscriptionFailedException(
                    f"Segment {index} transcription failed",
                    status_code=_status_code_of(result),
                )
            merged.extend(result)
        return merged

    async def _transcribe_segment_with_offset(
        self,
        segment_path: str,
        segment_index: int,
        offset: float,
        duration: float,
        chat_id: int,
    ) -> List[Tuple[float, float, str]]:
        transcript, usage_metadata = await self._generate(segment_path)
        _emit_stt_usage(self._model, duration, usage_metadata)

        adjusted: List[Tuple[float, float, str]] = []
        dropped = 0
        for segment in transcript.segments:
            start = parse_timestamp(segment.start)
            end = parse_timestamp(segment.end)
            text = (segment.text or "").strip()
            if start is None or not text:
                dropped += 1
                continue
            if end is None or end < start:
                end = start
            # The model occasionally overshoots the clip; clamp so a stitched
            # recording never has an utterance that ends inside the next segment.
            if duration > 0:
                start = min(start, duration)
                end = min(end, duration)
            adjusted.append((start + offset, end + offset, text))

        logger.info(
            f"Segment {segment_index}: transcribed {len(adjusted)} segments with "
            f"offset {offset:.2f}s"
            + (f" ({dropped} unparseable dropped)" if dropped else "")
        )
        return adjusted

    async def _generate(self, segment_path: str):
        """One Gemini call for one audio file. Returns (Transcript, usage).

        Kept small and side-effect free so tests can replace it.
        """
        data = await asyncio.to_thread(_read_bytes, segment_path)
        audio_part = self._types.Part.from_bytes(
            data=data, mime_type=_mime_type_for(segment_path)
        )
        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self._model,
                    contents=[TRANSCRIPTION_PROMPT, audio_part],
                    config=self._config,
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as e:
            raise TranscriptionFailedException(
                f"Gemini transcription timed out after {REQUEST_TIMEOUT_SECONDS}s",
                status_code=504,
            ) from e
        except Exception as e:
            raise TranscriptionFailedException(
                f"Gemini transcription request failed: {type(e).__name__}",
                status_code=_status_code_of(e),
            ) from e

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, Transcript):
            transcript = parsed
        else:
            text = getattr(response, "text", None) or ""
            if not text.strip():
                # A blocked or empty candidate: no speech, not an error. The
                # fallback wrapper decides whether another engine should try.
                transcript = Transcript()
            else:
                try:
                    transcript = Transcript.model_validate_json(text)
                except Exception as e:
                    raise TranscriptionFailedException(
                        "Gemini returned an unparseable transcript: "
                        f"{type(e).__name__}",
                        status_code=502,
                    ) from e
        return transcript, getattr(response, "usage_metadata", None)

    async def _cleanup_segment_files(self, segment_paths: List[str], chat_id: int):
        """Remove the converted/segmented input files. Best-effort, never raises."""
        for segment_path in segment_paths:
            try:
                if await asyncio.to_thread(os.path.exists, segment_path):
                    await asyncio.to_thread(os.remove, segment_path)
            except OSError as e:
                logger.warning(f"Failed to cleanup segment file: {type(e).__name__}")
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=str(chat_id),
                        audit_id=None,  # Will be set by external service,
                        details={
                            "error": f"Failed to cleanup segment file: {type(e).__name__}",  # noqa: E501
                            "segment_path": segment_path,
                            "exception_type": type(e).__name__,
                            "component": "GeminiTranscriptionService",
                            "method": "_cleanup_segment_files",
                        },
                    )
                )


def _status_code_of(error: BaseException) -> int:
    """HTTP status from a google-genai ``APIError`` (its ``code``), else 500."""
    code = getattr(error, "code", None)
    if isinstance(code, int) and 100 <= code <= 599:
        return code
    if isinstance(error, TranscriptionFailedException):
        return error.status_code
    return 500
