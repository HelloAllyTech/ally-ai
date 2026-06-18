"""
Deepgram transcription service for Lambda function.
Uses Deepgram Nova-3 for transcription and OpenAI GPT for summarization.
"""

import asyncio
import os
from typing import Tuple

from app.core.config import settings
from deepgram import BufferSource, DeepgramClient, PrerecordedOptions
from deepgram.clients.listen.v1.rest import Results
from app.core.transcriptions.utils.audio_converter import convert_and_segment_audio_async, get_audio_duration
from app.core.transcriptions.utils.exceptions import TranscriptionFailedException
from app.core.transcriptions.utils.logger import get_logger
from app.core.transcriptions.utils.phi_events import PHIEvents
from app.core.transcriptions.utils.phi_logger import PHILogEvent, phi_logger

logger = get_logger(__name__)

# Sentence pause threshold in seconds
SENTENCE_PAUSE_THRESHOLD = 0.1

# Message merging threshold in seconds - merge messages from same speaker if gap
# is less than this
MESSAGE_MERGE_THRESHOLD = 3.0

MAX_SEGMENT_SIZE_MB = 24  # Maximum size per segment for Deepgram API compatibility


class DeepgramTranscriptionService:
    """
    Deepgram transcription service for processing audio files.
    """

    def __init__(self):
        """
        Initialize the Deepgram transcription service.
        """
        self.deepgram = DeepgramClient(settings.DEEPGRAM.API_KEY)

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
            # Download and convert audio to WAV format
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
                        "component": "DeepgramTranscriptionService",
                        "method": "transcribe_audio_from_url",
                    },
                )
            )

            # Transcribe using appropriate method based on number of segments
            if len(segment_paths) == 1:
                # Single segment - process directly
                segments_text = await self._transcribe_with_deepgram_sdk(
                    segment_paths[0], chat_id
                )
            else:
                # Multiple segments - process in parallel
                segments_text = await self._transcribe_multiple_segments(
                    segment_paths, chat_id
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
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Error transcribing audio from URL for chat_id {chat_id}: {type(e).__name__}",  # noqa: E501
                        "chat_id": chat_id,
                        "audio_url": audio_url,
                        "sample_rate": sample_rate,
                        "exception_type": type(e).__name__,
                        "component": "DeepgramTranscriptionService",
                        "method": "transcribe_audio_from_url",
                    },
                )
            )
            raise TranscriptionFailedException("Error transcribing audio from URL")

    async def _transcribe_with_deepgram_sdk(
        self, wav_file_path: str, chat_id: int
    ) -> str:
        """
        Transcribe audio using Deepgram SDK and format into
        segments text for OpenAI diarization.

        Args:
            wav_file_path (str): Path to the local WAV file
            chat_id (int): Chat ID for PHI logging.

        Returns:
            str: Formatted segments text with timing information for diarization
        """
        try:
            logger.info("Starting Deepgram SDK transcription")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": "Starting Deepgram SDK transcription",
                        "wav_file_path": wav_file_path,
                        "component": "DeepgramTranscriptionService",
                        "method": "_transcribe_with_deepgram_sdk",
                    },
                )
            )

            # Read the audio file off the event loop — full audio payloads
            # can be many MB and `.read()` on the main thread would stall
            # other coroutines for tens of ms per call.
            def _read_audio_bytes(path: str) -> bytes:
                with open(path, "rb") as f:
                    return f.read()

            buffer_data = await asyncio.to_thread(
                _read_audio_bytes, wav_file_path
            )

            # Create buffer source
            payload = BufferSource(buffer=buffer_data)

            # Configure Deepgram options - disable diarization since we'll use
            # OpenAI for that
            options = PrerecordedOptions(
                model="nova-3",
                language="multi",  # Support multiple languages
                smart_format=True,
                punctuate=True,
                diarize=False,  # Disable Deepgram diarization
                utterances=True,
                utt_split=0.8,  # Split utterances at 0.8 second pauses
                paragraphs=False,
                summarize=False,
            )

            logger.info("Sending request to Deepgram SDK")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": "Sending request to Deepgram SDK",
                        "wav_file_path": wav_file_path,
                        "buffer_size": len(buffer_data),
                        "model": "nova-3",
                        "language": "multi",
                        "component": "DeepgramTranscriptionService",
                        "method": "_transcribe_with_deepgram_sdk",
                    },
                )
            )

            # Make the transcription request
            response = await self.deepgram.listen.asyncrest.v("1").transcribe_file(
                payload,
                options,
                addons={"mip_opt_out": "true"},
            )

            # Best-effort batch-STT AI-cost emit (never blocks transcription).
            try:
                from app.core.llm_usage.emitter import emit_ai_usage

                _dur = await get_audio_duration(wav_file_path)
                if _dur and _dur > 0:
                    emit_ai_usage(
                        "stt",
                        "deepgram",
                        "nova-3",
                        "transcription",
                        audio_ms=int(_dur * 1000),
                    )
            except Exception:
                logger.debug("batch STT usage emit skipped (best-effort)")

            if not response or not hasattr(response, "results"):
                logger.error("Invalid response from Deepgram SDK")
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=str(chat_id),
                        audit_id=None,  # Will be set by external service,
                        details={
                            "error": "Invalid response from Deepgram SDK",
                            "wav_file_path": wav_file_path,
                            "component": "DeepgramTranscriptionService",
                            "method": "_transcribe_with_deepgram_sdk",
                        },
                    )
                )
                raise TranscriptionFailedException("Invalid response from Deepgram SDK")

            logger.info("Received transcription response from Deepgram SDK")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": "Received transcription response from Deepgram SDK",
                        "wav_file_path": wav_file_path,
                        "has_results": bool(response.results),
                        "component": "DeepgramTranscriptionService",
                        "method": "_transcribe_with_deepgram_sdk",
                    },
                )
            )

            # Format the transcription into segments text similar to OpenAI format
            segments_text = await self._format_deepgram_response_for_diarization(
                response.results, chat_id
            )

            return segments_text

        except Exception as e:
            logger.exception(
                f"Error during Deepgram SDK transcription: {type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Error during Deepgram SDK transcription: {type(e).__name__}",  # noqa: E501
                        "wav_file_path": wav_file_path,
                        "exception_type": type(e).__name__,
                        "component": "DeepgramTranscriptionService",
                        "method": "_transcribe_with_deepgram_sdk",
                    },
                )
            )
            raise TranscriptionFailedException("Deepgram SDK transcription error")

    async def _transcribe_multiple_segments(
        self, segment_paths: list, chat_id: int
    ) -> str:
        """Transcribe multiple audio segments in parallel and combine results"""
        try:
            logger.info(
                f"Starting parallel transcription of {len(segment_paths)} segments"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Starting parallel transcription of {len(segment_paths)} segments",  # noqa: E501
                        "segment_count": len(segment_paths),
                        "segment_paths": segment_paths,
                        "component": "DeepgramTranscriptionService",
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
                            chat_id=str(chat_id),
                            audit_id=None,  # Will be set by external service,
                            details={
                                "error": f"Segment {i} transcription failed: {type(result).__name__}",  # noqa: E501
                                "segment_index": i,
                                "segment_path": segment_paths[i],
                                "exception_type": type(result).__name__,
                                "component": "DeepgramTranscriptionService",
                                "method": "_transcribe_multiple_segments",
                            },
                        )
                    )
                    raise result
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
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Combined {len(all_segments)} segments from {len(segment_paths)} files",  # noqa: E501
                        "total_segments": len(all_segments),
                        "segment_files_count": len(segment_paths),
                        "segments_text_length": len(segments_text),
                        "component": "DeepgramTranscriptionService",
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
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Error transcribing multiple segments: {type(e).__name__}",  # noqa: E501
                        "segment_count": len(segment_paths),
                        "exception_type": type(e).__name__,
                        "component": "DeepgramTranscriptionService",
                        "method": "_transcribe_multiple_segments",
                    },
                )
            )
            raise

    async def _transcribe_segment_with_offset(
        self, segment_path: str, segment_index: int, chat_id: int
    ) -> list:
        """Transcribe a single segment and adjust timing based on segment index"""
        try:
            # Get segment duration to calculate offset
            duration = await get_audio_duration(segment_path)
            offset = segment_index * duration  # This is approximate, we'll refine it

            # Transcribe the segment using Deepgram SDK
            segments_text = await self._transcribe_with_deepgram_sdk(
                segment_path, chat_id
            )

            # Parse the segments text to extract timing and content
            adjusted_segments = []
            for line in segments_text.split("\n"):
                if line.strip() and "-" in line and ":" in line:
                    try:
                        # Parse format: "start-end: text"
                        time_part, text_part = line.split(":", 1)
                        start_str, end_str = time_part.split("-")
                        start_time = float(start_str.strip())
                        end_time = float(end_str.strip())
                        text = text_part.strip()

                        # Adjust timing for this segment
                        adjusted_start = start_time + offset
                        adjusted_end = end_time + offset
                        adjusted_segments.append((adjusted_start, adjusted_end, text))

                    except (ValueError, IndexError) as parse_error:
                        logger.warning(
                            f"Failed to parse segment line: "
                            f"{type(parse_error).__name__}"
                        )
                        await phi_logger.log(
                            PHILogEvent(
                                event_type=PHIEvents.SYSTEM_ERROR,
                                chat_id=str(chat_id),
                                audit_id=None,  # Will be set by external service,
                                details={
                                    "error": f"Failed to parse segment line: {type(parse_error).__name__}",  # noqa: E501
                                    "segment_path": segment_path,
                                    "segment_index": segment_index,
                                    "line": line,
                                    "exception_type": type(parse_error).__name__,
                                    "component": "DeepgramTranscriptionService",
                                    "method": "_transcribe_segment_with_offset",
                                },
                            )
                        )
                        continue

            logger.info(
                f"Segment {segment_index}: transcribed {len(adjusted_segments)} "
                f"segments with offset {offset:.2f}s"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Segment {segment_index}: transcribed {len(adjusted_segments)} segments with offset {offset:.2f}s",  # noqa: E501
                        "segment_path": segment_path,
                        "segment_index": segment_index,
                        "adjusted_segments_count": len(adjusted_segments),
                        "offset_seconds": offset,
                        "duration": duration,
                        "component": "DeepgramTranscriptionService",
                        "method": "_transcribe_segment_with_offset",
                    },
                )
            )

            return adjusted_segments

        except Exception as e:
            logger.error(
                f"Error transcribing segment {segment_index}: {type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Error transcribing segment {segment_index}: {type(e).__name__}",  # noqa: E501
                        "segment_path": segment_path,
                        "segment_index": segment_index,
                        "exception_type": type(e).__name__,
                        "component": "DeepgramTranscriptionService",
                        "method": "_transcribe_segment_with_offset",
                    },
                )
            )
            raise

    async def _cleanup_segment_files(self, segment_paths: list, chat_id: int):
        """Clean up temporary segment files"""
        for segment_path in segment_paths:
            try:
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                    logger.info("Cleaned up segment file")
                    await phi_logger.log(
                        PHILogEvent(
                            event_type=PHIEvents.DATA_DELETED,
                            chat_id=str(chat_id),
                            audit_id=None,  # Will be set by external service,
                            details={
                                "message": "Cleaned up segment file",
                                "segment_path": segment_path,
                                "component": "DeepgramTranscriptionService",
                                "method": "_cleanup_segment_files",
                            },
                        )
                    )
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to clean up segment file: {type(cleanup_error).__name__}"
                )
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=str(chat_id),
                        audit_id=None,  # Will be set by external service,
                        details={
                            "error": f"Failed to clean up segment file: {type(cleanup_error).__name__}",  # noqa: E501
                            "segment_path": segment_path,
                            "exception_type": type(cleanup_error).__name__,
                            "component": "DeepgramTranscriptionService",
                            "method": "_cleanup_segment_files",
                        },
                    )
                )

    async def _format_deepgram_response_for_diarization(
        self, results: Results, chat_id: int
    ) -> str:
        """
        Format Deepgram response into segments text similar to
        OpenAI transcription service format.

        Args:
            results: Deepgram Results object from SDK response
            chat_id (int): Chat ID for PHI logging.

        Returns:
            str: Formatted segments text with timing information
        """
        try:
            if not results.channels or len(results.channels) == 0:
                logger.warning("No channels found in Deepgram response")
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=str(chat_id),
                        audit_id=None,  # Will be set by external service,
                        details={
                            "error": "No channels found in Deepgram response",
                            "component": "DeepgramTranscriptionService",
                            "method": "_format_deepgram_response_for_diarization",
                        },
                    )
                )
                return ""

            channel = results.channels[0]
            if not channel.alternatives or len(channel.alternatives) == 0:
                logger.warning("No alternatives found in Deepgram response")
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=str(chat_id),
                        audit_id=None,  # Will be set by external service,
                        details={
                            "error": "No alternatives found in Deepgram response",
                            "channels_count": len(results.channels),
                            "component": "DeepgramTranscriptionService",
                            "method": "_format_deepgram_response_for_diarization",
                        },
                    )
                )
                return ""

            alternative = channel.alternatives[0]
            if not alternative.words:
                logger.warning("No words found in Deepgram response")
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=str(chat_id),
                        audit_id=None,  # Will be set by external service,
                        details={
                            "error": "No words found in Deepgram response",
                            "channels_count": len(results.channels),
                            "alternatives_count": len(channel.alternatives),
                            "component": "DeepgramTranscriptionService",
                            "method": "_format_deepgram_response_for_diarization",
                        },
                    )
                )
                return ""

            # Group words into segments based on utterances or pauses
            segments = []
            current_segment_words = []
            current_start = None

            for word in alternative.words:
                word_start = word.start
                word_end = word.end
                word_text = word.word

                if current_start is None:
                    current_start = word_start

                # Check if we should start a new segment based on pause threshold
                if (
                    current_segment_words
                    and word_start - current_segment_words[-1]["end"]
                    > SENTENCE_PAUSE_THRESHOLD
                ):

                    # Finalize current segment
                    segment_text = " ".join([w["word"] for w in current_segment_words])
                    segment_end = current_segment_words[-1]["end"]
                    segments.append(
                        {
                            "start": current_start,
                            "end": segment_end,
                            "text": segment_text.strip(),
                        }
                    )

                    # Start new segment
                    current_segment_words = []
                    current_start = word_start

                current_segment_words.append(
                    {"word": word_text, "start": word_start, "end": word_end}
                )

            # Add the last segment
            if current_segment_words:
                segment_text = " ".join([w["word"] for w in current_segment_words])
                segment_end = current_segment_words[-1]["end"]
                segments.append(
                    {
                        "start": current_start,
                        "end": segment_end,
                        "text": segment_text.strip(),
                    }
                )

            # Format segments similar to OpenAI transcription service
            segments_text = "\n".join(
                [
                    f"{segment['start']:.2f}-{segment['end']:.2f}: {segment['text']}"
                    for segment in segments
                    if segment["text"].strip()
                ]
            )

            logger.info(f"Formatted {len(segments)} segments for diarization")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Formatted {len(segments)} segments for diarization",  # noqa: E501
                        "segments_count": len(segments),
                        "segments_text_length": len(segments_text),
                        "words_count": len(alternative.words),
                        "sentence_pause_threshold": SENTENCE_PAUSE_THRESHOLD,
                        "component": "DeepgramTranscriptionService",
                        "method": "_format_deepgram_response_for_diarization",
                    },
                )
            )
            return segments_text

        except Exception as e:
            logger.exception(
                f"Error formatting Deepgram response for diarization: "
                f"{type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Error formatting Deepgram response for diarization: {type(e).__name__}",  # noqa: E501
                        "exception_type": type(e).__name__,
                        "component": "DeepgramTranscriptionService",
                        "method": "_format_deepgram_response_for_diarization",
                    },
                )
            )
            raise TranscriptionFailedException("Error formatting response")
