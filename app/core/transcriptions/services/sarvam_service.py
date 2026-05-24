"""
Sarvam AI transcription service for Lambda function.
Uses Sarvam Saaras v2.5 for transcription with diarization.
"""

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any, Tuple

from app.core.config import settings
from sarvamai import AsyncSarvamAI
from app.core.transcriptions.utils.audio_converter import convert_and_segment_audio_async, get_audio_duration
from app.core.transcriptions.utils.exceptions import TranscriptionFailedException
from app.core.transcriptions.utils.logger import get_logger
from app.core.transcriptions.utils.phi_events import PHIEvents
from app.core.transcriptions.utils.phi_logger import PHILogEvent, phi_logger

logger = get_logger(__name__)

MAX_SEGMENT_SIZE_MB = 24
POLL_INTERVAL = 7
JOB_TIMEOUT = 600


class SarvamTranscriptionService:
    """Sarvam AI transcription service for processing audio files with diarization."""

    def __init__(self):
        """Initialize the Sarvam transcription service."""
        self.client = AsyncSarvamAI(api_subscription_key=settings.SARVAM.API_KEY)

    async def transcribe_audio_from_url(
        self, audio_url: str, chat_id: int, sample_rate: int = 8000
    ) -> Tuple[int, str]:
        """
        Transcribe audio from URL using Sarvam AI translation service.

        Args:
            audio_url: URL containing the audio file
            chat_id: Chat ID for the transcription session
            sample_rate: Expected sample rate of the audio (default: 8000)

        Returns:
            Tuple[int, str]: (chat_id, segments_text)

        Raises:
            TranscriptionFailedException: If transcription fails
        """
        try:
            # Convert and segment audio
            segment_paths = await convert_and_segment_audio_async(
                audio_url=audio_url,
                sample_rate=sample_rate,
                max_segment_size_mb=MAX_SEGMENT_SIZE_MB,
            )

            logger.info(f"Audio converted into {len(segment_paths)} segments")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "message": f"Audio converted into {len(segment_paths)} segments",
                        "segment_count": len(segment_paths),
                        "component": "SarvamTranscriptionService",
                        "method": "transcribe_audio_from_url",
                    },
                )
            )

            # Transcribe based on segment count
            if len(segment_paths) == 1:
                segments_text = await self._transcribe_with_sarvam_sdk(
                    segment_paths[0], chat_id
                )
            else:
                segments_text = await self._transcribe_multiple_segments(
                    segment_paths, chat_id
                )

            # Log the output transcript
            logger.info(
                f"Transcription completed with {len(segments_text.split(chr(10)))} segments"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "message": "Transcription completed successfully",
                        "transcript_length": len(segments_text),
                        "segment_lines": len(segments_text.split("\n")),
                        "component": "SarvamTranscriptionService",
                        "method": "transcribe_audio_from_url",
                    },
                )
            )

            # Cleanup
            await self._cleanup_segment_files(segment_paths, chat_id)

            return chat_id, segments_text

        except Exception as e:
            logger.error(f"Error transcribing audio: {type(e).__name__}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "error": f"Error transcribing audio: {type(e).__name__}",
                        "exception_type": type(e).__name__,
                        "component": "SarvamTranscriptionService",
                        "method": "transcribe_audio_from_url",
                    },
                )
            )
            raise TranscriptionFailedException("Transcription failed")

    async def _transcribe_with_sarvam_sdk(self, audio_path: str, chat_id: int) -> str:
        """
        Transcribe single audio file using Sarvam SDK.

        Args:
            audio_path: Path to audio file
            chat_id: Chat ID for logging

        Returns:
            Formatted segments text with timing and speaker info
        """
        try:
            logger.info("Starting Sarvam transcription job")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "message": "Creating Sarvam transcription job",
                        "component": "SarvamTranscriptionService",
                        "method": "_transcribe_with_sarvam_sdk",
                    },
                )
            )

            # Create transcription job
            job = await self.client.speech_to_text_translate_job.create_job(
                model="saaras:v2.5",
                with_diarization=True,
                num_speakers=2,
                prompt="Conversation transcription",
            )

            job_id = job._job_id
            logger.info(f"Job created: {job_id}")

            # Upload and start
            await job.upload_files(file_paths=[audio_path], timeout=120.0)
            logger.info("Audio uploaded")

            await job.start()
            logger.info("Job started")

            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "message": f"Transcription job started: {job_id}",
                        "job_id": job_id,
                        "component": "SarvamTranscriptionService",
                        "method": "_transcribe_with_sarvam_sdk",
                    },
                )
            )

            # Wait for completion
            await job.wait_until_complete(
                poll_interval=POLL_INTERVAL, timeout=JOB_TIMEOUT
            )

            if await job.is_failed():
                logger.error(f"Job {job_id} failed")
                raise TranscriptionFailedException("Sarvam job failed")

            # Download results
            # Use a unique per-job directory to avoid clashes across parallel tasks
            output_dir = Path(f"/tmp/sarvam_output_{chat_id}_{job_id}")
            output_dir.mkdir(exist_ok=True)
            await job.download_outputs(output_dir=str(output_dir))
            logger.info(f"Output downloaded to {output_dir}")

            # Parse output
            segments_text = await self._parse_sarvam_output(output_dir, chat_id)

            shutil.rmtree(output_dir, ignore_errors=True)

            return segments_text

        except Exception as e:
            logger.error(f"Sarvam transcription error: {type(e).__name__}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "error": f"Sarvam transcription error: {type(e).__name__}",
                        "exception_type": type(e).__name__,
                        "component": "SarvamTranscriptionService",
                        "method": "_transcribe_with_sarvam_sdk",
                    },
                )
            )
            raise

    async def _transcribe_multiple_segments(
        self, segment_paths: list, chat_id: int
    ) -> str:
        """Transcribe multiple segments in parallel."""
        try:
            logger.info(f"Processing {len(segment_paths)} segments in parallel")

            # Create parallel tasks
            tasks = [
                self._transcribe_segment_with_offset(path, idx, chat_id)
                for idx, path in enumerate(segment_paths)
            ]

            # Execute
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect segments
            all_segments = []
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Segment {idx} failed: {type(result).__name__}")
                    raise result
                all_segments.extend(result)

            # Format
            segments_text = "\n".join(
                [f"{start:.2f}-{end:.2f}: {text}" for start, end, text in all_segments]
            )

            logger.info(f"Combined {len(all_segments)} segments")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "message": f"Combined {len(all_segments)} segments",
                        "total_segments": len(all_segments),
                        "component": "SarvamTranscriptionService",
                        "method": "_transcribe_multiple_segments",
                    },
                )
            )

            return segments_text

        except Exception as e:
            logger.error(f"Multiple segments error: {type(e).__name__}")
            raise

    async def _transcribe_segment_with_offset(
        self, segment_path: str, segment_index: int, chat_id: int
    ) -> list:
        """Transcribe segment and adjust timing."""
        try:
            # Calculate offset
            duration = await get_audio_duration(segment_path)
            offset = segment_index * duration

            # Transcribe
            segments_text = await self._transcribe_with_sarvam_sdk(
                segment_path, chat_id
            )

            # Parse and adjust
            adjusted_segments = []
            for line in segments_text.split("\n"):
                if line.strip() and "-" in line and ":" in line:
                    try:
                        time_part, text_part = line.split(":", 1)
                        start_str, end_str = time_part.split("-")
                        start = float(start_str.strip()) + offset
                        end = float(end_str.strip()) + offset
                        adjusted_segments.append((start, end, text_part.strip()))
                    except (ValueError, IndexError):
                        continue

            logger.info(
                f"Segment {segment_index}: {len(adjusted_segments)} segments, "
                f"offset {offset:.2f}s"
            )
            return adjusted_segments

        except Exception as e:
            logger.error(f"Segment {segment_index} error: {type(e).__name__}")
            raise

    async def _parse_sarvam_output(self, output_dir: Path, chat_id: int) -> str:
        """Parse Sarvam output and format segments."""
        try:
            # Collect all supported files
            json_files = list(output_dir.glob("*.json"))
            txt_files = list(output_dir.glob("*.txt"))
            srt_files = list(output_dir.glob("*.srt"))

            if not (json_files or txt_files or srt_files):
                raise TranscriptionFailedException(
                    "No output files found from Sarvam job"
                )

            all_segments = []  # list of tuples (start, end, text)

            # Helper to append a plain text segment when no timing is available
            def add_plain_text_segment(text: str):
                if text and text.strip():
                    all_segments.append((0.0, 0.0, text.strip()))

            # File reads are wrapped in asyncio.to_thread — these output
            # files may be large and looping blocking-IO on the event loop
            # would stall every other coroutine for the duration.
            def _read_json(path: Path) -> Any:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)

            def _read_text(path: Path) -> str:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()

            # Parse JSON outputs (aggregate all)
            for jf in json_files:
                try:
                    data = await asyncio.to_thread(_read_json, jf)

                    # Case 1: word-level data
                    if (
                        isinstance(data, dict)
                        and "words" in data
                        and isinstance(data["words"], list)
                    ):
                        current_segment = {"words": [], "start": None, "speaker": None}
                        last_end = 0.0
                        for word_data in data["words"]:
                            word = word_data.get("word", "")
                            start = float(word_data.get("start", 0) or 0)
                            end = float(word_data.get("end", 0) or 0)
                            last_end = end or last_end
                            speaker = word_data.get("speaker", "Speaker 0")
                            if (
                                current_segment["speaker"]
                                and current_segment["speaker"] != speaker
                            ):
                                text = " ".join(current_segment["words"]).strip()
                                if text:
                                    all_segments.append(
                                        (
                                            float(current_segment["start"] or 0.0),
                                            last_end,
                                            text,
                                        )
                                    )
                                current_segment = {
                                    "words": [],
                                    "start": None,
                                    "speaker": None,
                                }
                            if current_segment["start"] is None:
                                current_segment["start"] = start
                            current_segment["words"].append(word)
                            current_segment["speaker"] = speaker
                        if current_segment["words"]:
                            text = " ".join(current_segment["words"]).strip()
                            if text:
                                all_segments.append(
                                    (
                                        float(current_segment["start"] or 0.0),
                                        last_end,
                                        text,
                                    )
                                )

                    # Case 2: diarized transcript structure
                    elif isinstance(data, dict) and "diarized_transcript" in data:
                        entries = data["diarized_transcript"].get("entries", []) or []
                        for entry in entries:
                            speaker = entry.get("speaker_id", "Speaker")
                            st = float(entry.get("start_time_seconds", 0) or 0)
                            et = float(entry.get("end_time_seconds", st) or st)
                            tx = entry.get("transcript", "").strip()
                            if tx:
                                all_segments.append((st, et, f"[{speaker}] {tx}"))

                    # Case 3: plain transcript text in dict
                    elif isinstance(data, dict) and (
                        "transcript" in data or "text" in data
                    ):
                        tx = data.get("transcript") or data.get("text") or ""
                        add_plain_text_segment(str(tx))

                    # Case 4: list of items with text
                    elif isinstance(data, list):
                        pieces = []
                        for item in data:
                            if isinstance(item, dict):
                                t = item.get("text") or item.get("transcript") or ""
                                if t:
                                    pieces.append(str(t))
                            elif isinstance(item, str):
                                pieces.append(item)
                        if pieces:
                            add_plain_text_segment(" ".join(pieces))

                    else:
                        # Unknown schema, store a preview to help debugging
                        preview = str(data)[:200]
                        logger.warning(
                            f"Unknown Sarvam JSON schema in {jf.name}: {preview}"
                        )
                except Exception as je:
                    logger.warning(
                        f"Failed to parse JSON output {jf.name}: {type(je).__name__}"
                    )

            # Parse TXT outputs (fallback)
            for tf in txt_files:
                try:
                    txt_content = await asyncio.to_thread(_read_text, tf)
                    lines = txt_content.splitlines()
                    text = " ".join(l for l in lines if l.strip())
                    add_plain_text_segment(text)
                except Exception as te:
                    logger.warning(
                        f"Failed to read TXT output {tf.name}: {type(te).__name__}"
                    )

            # Parse minimal SRT outputs (basic time parsing)
            def _parse_srt_time(ts: str) -> float:
                try:
                    hh, mm, rest = ts.split(":")
                    ss, ms = rest.split(",")
                    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
                except Exception:
                    return 0.0

            for sf in srt_files:
                try:
                    srt_content = await asyncio.to_thread(_read_text, sf)
                    blocks = srt_content.split("\n\n")
                    for block in blocks:
                        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
                        if len(lines) >= 2 and "-->" in lines[0]:
                            # Some SRTs may have an index line; handle both
                            time_line = lines[0]
                            text_lines = lines[1:]
                            if (
                                lines[0].isdigit()
                                and len(lines) >= 3
                                and "-->" in lines[1]
                            ):
                                time_line = lines[1]
                                text_lines = lines[2:]
                            try:
                                start_ts, end_ts = [
                                    p.strip() for p in time_line.split("-->")
                                ]
                                st = _parse_srt_time(start_ts)
                                et = _parse_srt_time(end_ts)
                            except Exception:
                                st = 0.0
                                et = 0.0
                            text = " ".join(text_lines).strip()
                            if text:
                                all_segments.append((st, et, text))
                except Exception as se:
                    logger.warning(
                        f"Failed to parse SRT output {sf.name}: {type(se).__name__}"
                    )

            # Sort by start time and format
            all_segments.sort(key=lambda x: x[0])

            segments_text = "\n".join(
                [f"{s:.2f}-{e:.2f}: {t}" for s, e, t in all_segments if t.strip()]
            )

            logger.info(f"Parsed {len(all_segments)} segments from Sarvam output")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=str(chat_id),
                    audit_id=None,
                    details={
                        "message": f"Parsed {len(all_segments)} segments from output",
                        "segments_count": len(all_segments),
                        "transcript_preview": (
                            segments_text[:200] if segments_text else ""
                        ),
                        "component": "SarvamTranscriptionService",
                        "method": "_parse_sarvam_output",
                    },
                )
            )

            return segments_text

        except Exception as e:
            logger.error(f"Parse output error: {type(e).__name__}")
            raise

    async def _cleanup_segment_files(self, segment_paths: list, chat_id: int):
        """Clean up temporary files."""
        for path in segment_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Cleaned up: {path}")
            except Exception as e:
                logger.warning(f"Cleanup failed for {path}: {type(e).__name__}")
