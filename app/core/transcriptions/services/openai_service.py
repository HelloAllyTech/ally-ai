"""
OpenAI transcription service for Lambda function.
Uses OpenAI Whisper for transcription and GPT for summarization.
"""

import asyncio
import os
from typing import Tuple

from app.core.config import settings
from openai import OpenAI
from app.core.transcriptions.utils.audio_converter import convert_and_segment_audio_async, get_audio_duration
from app.core.transcriptions.utils.exceptions import TranscriptionFailedException
from app.core.transcriptions.utils.logger import get_logger
from app.core.transcriptions.utils.phi_events import PHIEvents
from app.core.transcriptions.utils.phi_logger import PHILogEvent, phi_logger

logger = get_logger(__name__)


def _emit_stt_usage(model: str, audio_seconds: float) -> None:
    """Best-effort batch-STT AI-cost emit. Never raises / blocks transcription."""
    try:
        if not audio_seconds or audio_seconds <= 0:
            return
        from app.core.llm_usage.emitter import emit_ai_usage

        emit_ai_usage(
            "stt",
            "openai",
            model,
            "transcription",
            audio_ms=int(audio_seconds * 1000),
        )
    except Exception:
        logger.debug("batch STT usage emit skipped (best-effort)")


class OpenAITranscriptionService:
    """
    OpenAI transcription service for processing audio files.
    """

    def __init__(self):
        """
        Initialize the OpenAI transcription service.
        """
        self.client = OpenAI(
            api_key=settings.OPENAI.API_KEY,
            organization=settings.OPENAI.ORGANIZATION_ID,
        )

    async def transcribe_audio_from_url(
        self,
        audio_url: str,
        chat_id: int,
        sample_rate: int = 8000,
        is_linear16_encoded: bool = False,
    ) -> Tuple[int, str]:
        """
        Transcribe audio from URL and generate a summary.

        Args:
            audio_url (str): URL containing the audio file
            chat_id (int): Chat ID for the transcription session
            sample_rate (int): Expected sample rate of the audio (default: 8000)
            is_linear16_encoded (bool): Whether the audio is headerless linear16
                (s16le) PCM (mobile uploads), which must be decoded as raw PCM.

        Returns:
            Tuple[int, str]: (chat_id, segments_text)

        Raises:
            Exception: If transcription fails
        """
        try:
            # Transcribe and preprocess audio
            segments_text = await self._transcribe_and_preprocess_audio(
                audio_url, sample_rate, chat_id, is_linear16_encoded
            )

            return chat_id, segments_text

        except Exception as e:
            logger.error(
                f"Error transcribing audio from URL for chat_id {chat_id}: "
                f"{type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Error transcribing audio from URL for chat_id {chat_id}: {type(e).__name__}",  # noqa: E501
                        "chat_id": chat_id,
                        "audio_url": audio_url,
                        "sample_rate": sample_rate,
                        "exception_type": type(e).__name__,
                        "component": "OpenAITranscriptionService",
                        "method": "transcribe_audio_from_url",
                    },
                )
            )
            raise TranscriptionFailedException("Transcription failed")

    async def _transcribe_and_preprocess_audio(
        self,
        audio_url: str,
        sample_rate: int = 8000,
        chat_id: int = None,
        is_linear16_encoded: bool = False,
    ) -> str:
        """
        Transcribe audio and preprocess segments into a formatted
        string for diarization.

        This method supports audio segmentation for large files and processes
        segments in parallel for better performance.

        Args:
            audio_url (str): URL containing the audio file
            sample_rate (int): Expected sample rate of the audio (default: 8000)
            chat_id (int, optional): Chat ID for PHI logging.

        Returns:
            str: Formatted segments text with timing information

        Raises:
            Exception: If transcription fails
        """
        logger.info("Starting audio processing")
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.DATA_ACCESSED,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "message": "Starting audio processing",
                    "audio_url": audio_url,
                    "sample_rate": sample_rate,
                    "component": "OpenAITranscriptionService",
                    "method": "_transcribe_and_preprocess_audio",
                },
            )
        )
        segment_paths = []

        try:
            # Convert and segment audio if needed
            segment_paths = await convert_and_segment_audio_async(
                audio_url,
                sample_rate,
                chat_id=chat_id,
                is_linear16_encoded=is_linear16_encoded,
            )
            logger.info(f"Audio converted to {len(segment_paths)} segment(s)")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Audio converted to {len(segment_paths)} segment(s)",  # noqa: E501
                        "audio_url": audio_url,
                        "sample_rate": sample_rate,
                        "segment_count": len(segment_paths),
                        "component": "OpenAITranscriptionService",
                        "method": "_transcribe_and_preprocess_audio",
                    },
                )
            )

            if len(segment_paths) == 1:
                # Single file - process normally
                segments_text = await self._transcribe_single_file(
                    segment_paths[0], chat_id
                )
            else:
                # Multiple segments - process in parallel
                segments_text = await self._transcribe_multiple_segments(
                    segment_paths, chat_id
                )

            return segments_text

        except Exception as e:
            logger.error(
                f"Error transcribing and preprocessing audio: {type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Error transcribing and preprocessing audio: {type(e).__name__}",  # noqa: E501
                        "audio_url": audio_url,
                        "sample_rate": sample_rate,
                        "segment_count": len(segment_paths),
                        "exception_type": type(e).__name__,
                        "component": "OpenAITranscriptionService",
                        "method": "_transcribe_and_preprocess_audio",
                    },
                )
            )
            raise TranscriptionFailedException(
                "Audio transcription and preprocessing failed"
            )
        finally:
            # Clean up all segment files
            for segment_path in segment_paths:
                if await asyncio.to_thread(os.path.exists, segment_path):
                    try:
                        await asyncio.to_thread(os.remove, segment_path)
                        logger.info("Cleaned up segment file")
                        await phi_logger.log(
                            PHILogEvent(
                                event_type=PHIEvents.DATA_DELETED,
                                chat_id=str(chat_id) if chat_id else None,
                                audit_id=None,  # Will be set by external service,
                                details={
                                    "message": "Cleaned up segment file",
                                    "segment_path": segment_path,
                                    "component": "OpenAITranscriptionService",
                                    "method": "_transcribe_and_preprocess_audio",
                                },
                            )
                        )
                    except OSError as e:
                        logger.warning(
                            f"Failed to cleanup segment file: {type(e).__name__}"
                        )
                        await phi_logger.log(
                            PHILogEvent(
                                event_type=PHIEvents.SYSTEM_ERROR,
                                chat_id=str(chat_id) if chat_id else None,
                                audit_id=None,  # Will be set by external service,
                                details={
                                    "error": f"Failed to cleanup segment file: {type(e).__name__}",  # noqa: E501
                                    "segment_path": segment_path,
                                    "exception_type": type(e).__name__,
                                    "component": "OpenAITranscriptionService",
                                    "method": "_transcribe_and_preprocess_audio",
                                },
                            )
                        )

            logger.info("Audio processing complete")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": "Audio processing complete",
                        "audio_url": audio_url,
                        "sample_rate": sample_rate,
                        "segment_count": len(segment_paths),
                        "component": "OpenAITranscriptionService",
                        "method": "_transcribe_and_preprocess_audio",
                    },
                )
            )

    async def _transcribe_single_file(
        self, wav_file_path: str, chat_id: int = None
    ) -> str:
        """Transcribe a single WAV file"""
        try:
            with open(wav_file_path, "rb") as audio_file:
                transcription_verbose = await asyncio.to_thread(
                    self.client.audio.transcriptions.create,
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                )

            try:
                _emit_stt_usage("whisper-1", await get_audio_duration(wav_file_path))
            except Exception:
                pass

            logger.info("Single file transcription completed successfully")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": "Single file transcription completed successfully",
                        "wav_file_path": wav_file_path,
                        "model": "whisper-1",
                        "response_format": "verbose_json",
                        "component": "OpenAITranscriptionService",
                        "method": "_transcribe_single_file",
                    },
                )
            )

            # Preprocess segments for diarization
            total_segments = len(transcription_verbose.segments)
            segments_text = "\n".join(
                [
                    f"{segment.start:.2f}-{segment.end:.2f}: {segment.text.strip()}"
                    for segment in transcription_verbose.segments
                ]
            )

            logger.info(f"Preprocessed {total_segments} segments for diarization")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Preprocessed {total_segments} segments for diarization",  # noqa: E501
                        "wav_file_path": wav_file_path,
                        "total_segments": total_segments,
                        "segments_text_length": len(segments_text),
                        "component": "OpenAITranscriptionService",
                        "method": "_transcribe_single_file",
                    },
                )
            )

            # Clean up
            del transcription_verbose

            return segments_text

        except Exception as e:
            logger.error(f"Error transcribing single file: {type(e).__name__}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Error transcribing single file: {type(e).__name__}",
                        "wav_file_path": wav_file_path,
                        "exception_type": type(e).__name__,
                        "component": "OpenAITranscriptionService",
                        "method": "_transcribe_single_file",
                    },
                )
            )
            raise TranscriptionFailedException("Error transcribing single file")

    async def _transcribe_multiple_segments(
        self, segment_paths: list, chat_id: int = None
    ) -> str:
        """Transcribe multiple audio segments in parallel and combine results"""
        try:
            logger.info(
                f"Starting parallel transcription of {len(segment_paths)} segments"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Starting parallel transcription of {len(segment_paths)} segments",  # noqa: E501
                        "segment_count": len(segment_paths),
                        "segment_paths": segment_paths,
                        "component": "OpenAITranscriptionService",
                        "method": "_transcribe_multiple_segments",
                    },
                )
            )

            # Create tasks for parallel transcription
            transcription_tasks = []
            for i, segment_path in enumerate(segment_paths):
                task = self._transcribe_segment_with_offset(segment_path, i, chat_id)
                transcription_tasks.append(task)

            # Execute all transcriptions in parallel
            segment_results = await asyncio.gather(
                *transcription_tasks, return_exceptions=True
            )

            # Process results and handle any errors
            all_segments = []
            for i, result in enumerate(segment_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Segment {i} transcription failed: {type(result).__name__}"
                    )
                    await phi_logger.log(
                        PHILogEvent(
                            event_type=PHIEvents.SYSTEM_ERROR,
                            chat_id=str(chat_id) if chat_id else None,
                            audit_id=None,  # Will be set by external service,
                            details={
                                "error": f"Segment {i} transcription failed: {type(result).__name__}",  # noqa: E501
                                "segment_index": i,
                                "segment_path": segment_paths[i],
                                "exception_type": type(result).__name__,
                                "component": "OpenAITranscriptionService",
                                "method": "_transcribe_multiple_segments",
                            },
                        )
                    )
                    raise TranscriptionFailedException(
                        f"Segment {i} transcription failed"
                    )
                all_segments.extend(result)

            # Sort segments by start time to maintain order
            all_segments.sort(key=lambda x: x[0])

            # Combine into final text
            segments_text = "\n".join(
                [
                    f"{start:.2f}-{end:.2f}: {text.strip()}"
                    for start, end, text in all_segments
                ]
            )

            logger.info(
                f"Combined {len(all_segments)} segments from {len(segment_paths)} files"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Combined {len(all_segments)} segments from {len(segment_paths)} files",  # noqa: E501
                        "total_segments": len(all_segments),
                        "segment_files_count": len(segment_paths),
                        "segments_text_length": len(segments_text),
                        "component": "OpenAITranscriptionService",
                        "method": "_transcribe_multiple_segments",
                    },
                )
            )

            return segments_text

        except Exception as e:
            logger.error(f"Error transcribing multiple segments: {type(e).__name__}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Error transcribing multiple segments: {type(e).__name__}",  # noqa: E501
                        "segment_count": len(segment_paths),
                        "exception_type": type(e).__name__,
                        "component": "OpenAITranscriptionService",
                        "method": "_transcribe_multiple_segments",
                    },
                )
            )
            raise TranscriptionFailedException("Error transcribing multiple segments")

    async def _transcribe_segment_with_offset(
        self, segment_path: str, segment_index: int, chat_id: int = None
    ) -> list:
        """Transcribe a single segment and adjust timing based on segment index"""
        try:
            # Get segment duration to calculate offset
            duration = await get_audio_duration(segment_path)
            offset = segment_index * duration  # This is approximate, we'll refine it

            with open(segment_path, "rb") as audio_file:
                transcription_verbose = await asyncio.to_thread(
                    self.client.audio.transcriptions.create,
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                )

            _emit_stt_usage("whisper-1", duration)

            # Adjust timing for this segment
            adjusted_segments = []
            for segment in transcription_verbose.segments:
                adjusted_start = segment.start + offset
                adjusted_end = segment.end + offset
                adjusted_segments.append((adjusted_start, adjusted_end, segment.text))

            logger.info(
                f"Segment {segment_index}: transcribed {len(adjusted_segments)} "
                "segments"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Segment {segment_index}: transcribed {len(adjusted_segments)} segments",  # noqa: E501
                        "segment_path": segment_path,
                        "segment_index": segment_index,
                        "adjusted_segments_count": len(adjusted_segments),
                        "offset_seconds": offset,
                        "duration": duration,
                        "model": "whisper-1",
                        "response_format": "verbose_json",
                        "component": "OpenAITranscriptionService",
                        "method": "_transcribe_segment_with_offset",
                    },
                )
            )

            # Clean up
            del transcription_verbose

            return adjusted_segments

        except Exception as e:
            logger.error(
                f"Error transcribing segment {segment_index}: {type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Error transcribing segment {segment_index}: {type(e).__name__}",  # noqa: E501
                        "segment_path": segment_path,
                        "segment_index": segment_index,
                        "exception_type": type(e).__name__,
                        "component": "OpenAITranscriptionService",
                        "method": "_transcribe_segment_with_offset",
                    },
                )
            )
            raise TranscriptionFailedException(
                f"Error transcribing segment {segment_index}"
            )
