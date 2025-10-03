import asyncio
import math
import os
import subprocess
import tempfile
from typing import List

import httpx

from utils.logger import get_logger
from utils.phi_events import PHIEvents
from utils.phi_logger import PHILogEvent, phi_logger

logger = get_logger(__name__)


async def download_file_to_temp_and_get_details(
    audio_url: str, chat_id: int = None
) -> tuple[str, str]:
    """
    Download audio file to temporary location and identify its actual format.

    Args:
        audio_url (str): URL of the audio file
        chat_id (int, optional): Chat ID for PHI logging.

    Returns:
        tuple[str, str]: Tuple containing path to the temporary downloaded
        file and its detected extension
    """
    try:
        logger.info("Downloading audio file")
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.DATA_ACCESSED,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "message": "Downloading audio file",
                    "audio_url": audio_url,
                    "component": "AudioConverter",
                    "method": "download_file_to_temp_and_get_details",
                },
            )
        )
        # Download the file first with a generic extension
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(audio_url)
            response.raise_for_status()

            # Create temporary input file with generic extension first
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
            temp_input.write(response.content)
            temp_input.close()

            logger.info(f"Downloaded audio file ({len(response.content)} bytes)")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Downloaded audio file ({len(response.content)} bytes)",  # noqa: E501
                        "audio_url": audio_url,
                        "file_size_bytes": len(response.content),
                        "temp_file_path": temp_input.name,
                        "component": "AudioConverter",
                        "method": "download_file_to_temp_and_get_details",
                    },
                )
            )

            # Now identify the actual format
            actual_extension = await identify_audio_format(temp_input.name, chat_id)

            # Rename the file with the correct extension
            correct_path = temp_input.name.replace(".audio", actual_extension)
            try:
                await asyncio.to_thread(os.rename, temp_input.name, correct_path)
                logger.info(f"Renamed file (detected format: {actual_extension})")
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.DATA_MODIFIED,
                        chat_id=str(chat_id) if chat_id else None,
                        audit_id=None,  # Will be set by external service,
                        details={
                            "message": f"Renamed file (detected format: {actual_extension})",  # noqa: E501
                            "audio_url": audio_url,
                            "detected_extension": actual_extension,
                            "original_path": temp_input.name,
                            "correct_path": correct_path,
                            "component": "AudioConverter",
                            "method": "download_file_to_temp_and_get_details",
                        },
                    )
                )
                return correct_path, actual_extension
            except OSError as e:
                logger.warning(
                    f"Failed to rename file, using original: {type(e).__name__}"
                )
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=str(chat_id) if chat_id else None,
                        audit_id=None,  # Will be set by external service,
                        details={
                            "error": f"Failed to rename file, using original: {type(e).__name__}",  # noqa: E501
                            "audio_url": audio_url,
                            "detected_extension": actual_extension,
                            "original_path": temp_input.name,
                            "correct_path": correct_path,
                            "exception_type": type(e).__name__,
                            "component": "AudioConverter",
                            "method": "download_file_to_temp_and_get_details",
                        },
                    )
                )
                return temp_input.name, actual_extension

    except Exception as e:
        logger.error(f"Error downloading audio: {type(e).__name__}")
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.SYSTEM_ERROR,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "error": f"Error downloading audio: {type(e).__name__}",
                    "audio_url": audio_url,
                    "exception_type": type(e).__name__,
                    "component": "AudioConverter",
                    "method": "download_file_to_temp_and_get_details",
                },
            )
        )
        raise


async def convert_and_segment_audio_async(
    audio_url: str,
    sample_rate: int = 8000,
    max_segment_size_mb: int = 24,
    chat_id: int = None,
) -> List[str]:
    """
    Convert audio from URL to audio file format and split into segments if needed.
    Each segment will be under the specified size limit for OpenAI API compatibility.
    If the audio is raw, it will be converted to wav.
    If the audio is not raw, it will be used as is.
    But we are assuming that input will be either raw or mp3.
    In future, we can add support for other formats.

    Args:
        audio_url (str): URL for the audio file
        sample_rate (int): Expected sample rate of the audio (default: 8000)
        max_segment_size_mb (int): Maximum size per segment in MB (default: 24MB)
        chat_id (int, optional): Chat ID for PHI logging.

    Returns:
        List[str]: List of paths to audio file segments
    """
    try:

        temp_downloaded_file_path, detected_extension = (
            await download_file_to_temp_and_get_details(audio_url, chat_id)
        )

        if detected_extension == ".raw":
            file_path = await download_and_convert_raw_audio_to_wav(
                temp_downloaded_file_path, sample_rate, chat_id
            )
        else:
            file_path = temp_downloaded_file_path

        # Check if segmentation is needed
        file_size_mb = await asyncio.to_thread(os.path.getsize, file_path) / (
            1024 * 1024
        )
        logger.info(f"file size: {file_size_mb:.2f} MB")
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.DATA_ACCESSED,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "message": f"file size: {file_size_mb:.2f} MB",
                    "audio_url": audio_url,
                    "file_path": file_path,
                    "file_size_mb": file_size_mb,
                    "detected_extension": detected_extension,
                    "sample_rate": sample_rate,
                    "max_segment_size_mb": max_segment_size_mb,
                    "component": "AudioConverter",
                    "method": "convert_and_segment_audio_async",
                },
            )
        )

        if file_size_mb <= max_segment_size_mb:
            logger.info("File size is within limit, no segmentation needed")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": "File size is within limit, no segmentation needed",
                        "audio_url": audio_url,
                        "file_path": file_path,
                        "file_size_mb": file_size_mb,
                        "max_segment_size_mb": max_segment_size_mb,
                        "component": "AudioConverter",
                        "method": "convert_and_segment_audio_async",
                    },
                )
            )
            return [file_path]

        # Get audio duration to calculate segment duration
        duration = await get_audio_duration(file_path, chat_id)
        if duration <= 0:
            logger.warning("Could not determine audio duration, using single file")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": "Could not determine audio duration, using single file",  # noqa: E501
                        "audio_url": audio_url,
                        "file_path": file_path,
                        "duration": duration,
                        "component": "AudioConverter",
                        "method": "convert_and_segment_audio_async",
                    },
                )
            )
            return [file_path]

        # Calculate number of segments needed
        # For 16kHz mono WAV: ~2.4MB per minute
        bytes_per_second = (file_size_mb * 1024 * 1024) / duration
        segment_duration = (max_segment_size_mb * 1024 * 1024) / bytes_per_second

        num_segments = math.ceil(duration / segment_duration)
        logger.info(
            f"Audio duration: {duration:.2f}s, will split into {num_segments} "
            f"segments of ~{segment_duration:.2f}s each"
        )
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.DATA_MODIFIED,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "message": f"Audio duration: {duration:.2f}s, will split into {num_segments} segments of ~{segment_duration:.2f}s each",  # noqa: E501
                    "audio_url": audio_url,
                    "file_path": file_path,
                    "duration": duration,
                    "num_segments": num_segments,
                    "segment_duration": segment_duration,
                    "bytes_per_second": bytes_per_second,
                    "component": "AudioConverter",
                    "method": "convert_and_segment_audio_async",
                },
            )
        )

        # Split audio into segments
        segment_paths = await split_audio_into_segments(
            file_path, num_segments, duration, chat_id
        )

        # Clean up original file
        try:
            await asyncio.to_thread(os.remove, file_path)
            logger.info("Cleaned up original file")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_DELETED,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": "Cleaned up original file",
                        "audio_url": audio_url,
                        "file_path": file_path,
                        "component": "AudioConverter",
                        "method": "convert_and_segment_audio_async",
                    },
                )
            )
        except OSError as e:
            logger.warning(f"Failed to cleanup original file: {type(e).__name__}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Failed to cleanup original file: {type(e).__name__}",
                        "audio_url": audio_url,
                        "file_path": file_path,
                        "exception_type": type(e).__name__,
                        "component": "AudioConverter",
                        "method": "convert_and_segment_audio_async",
                    },
                )
            )

        return segment_paths

    except Exception as e:
        logger.error(f"Error in convert_and_segment_audio_async: {type(e).__name__}")
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.SYSTEM_ERROR,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "error": f"Error in convert_and_segment_audio_async: {type(e).__name__}",  # noqa: E501
                    "audio_url": audio_url,
                    "sample_rate": sample_rate,
                    "max_segment_size_mb": max_segment_size_mb,
                    "exception_type": type(e).__name__,
                    "component": "AudioConverter",
                    "method": "convert_and_segment_audio_async",
                },
            )
        )
        raise


async def split_audio_into_segments(
    file_path: str, num_segments: int, total_duration: float, chat_id: int = None
) -> List[str]:
    """
    Split a file into multiple segments using FFmpeg.

    Args:
        file_path (str): Path to the input file
        num_segments (int): Number of segments to create
        total_duration (float): Total duration of the audio in seconds
        chat_id (int, optional): Chat ID for PHI logging.

    Returns:
        List[str]: List of paths to segment files
    """
    try:
        segment_duration = total_duration / num_segments
        segment_paths = []

        # Create base path for segments
        base_path = os.path.splitext(file_path)[0]
        file_extension = file_path.lower().split(".")[-1]
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, total_duration)

            # Create segment file path
            segment_path = f"{base_path}_segment_{i:03d}.{file_extension}"

            if file_extension == "wav":
                # Use FFmpeg to extract segment
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    file_path,
                    "-ss",
                    str(start_time),
                    "-t",
                    str(end_time - start_time),
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-y",
                    segment_path,
                ]
        else:
            # For other formats, copy the codec (faster, preserves quality)
            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                file_path,
                "-ss",
                str(start_time),
                "-t",
                str(end_time - start_time),
                "-c",
                "copy",  # Copy without re-encoding (much faster)
                "-y",
                segment_path,
            ]

        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"FFmpeg segmentation failed for segment {i}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"FFmpeg segmentation failed for segment {i}",
                        "file_path": file_path,
                        "segment_index": i,
                        "segment_path": segment_path,
                        "start_time": start_time,
                        "end_time": end_time,
                        "ffmpeg_returncode": process.returncode,
                        "stderr": stderr.decode("utf-8") if stderr else None,
                        "component": "AudioConverter",
                        "method": "split_audio_into_segments",
                    },
                )
            )
            raise subprocess.CalledProcessError(
                process.returncode,
                "ffmpeg",
                f"FFmpeg segmentation failed for segment {i}",
            )

        # Verify segment file size
        segment_size_mb = await asyncio.to_thread(os.path.getsize, segment_path) / (
            1024 * 1024
        )
        logger.info(
            f"Created segment {i} "
            f"({segment_size_mb:.2f} MB, {start_time:.2f}s - {end_time:.2f}s)"
        )
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.DATA_MODIFIED,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "message": f"Created segment {i} ({segment_size_mb:.2f} MB, {start_time:.2f}s - {end_time:.2f}s)",  # noqa: E501
                    "file_path": file_path,
                    "segment_index": i,
                    "segment_path": segment_path,
                    "segment_size_mb": segment_size_mb,
                    "start_time": start_time,
                    "end_time": end_time,
                    "component": "AudioConverter",
                    "method": "split_audio_into_segments",
                },
            )
        )

        segment_paths.append(segment_path)

        logger.info(f"Successfully created {len(segment_paths)} audio segments")
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.DATA_MODIFIED,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "message": f"Successfully created {len(segment_paths)} audio segments",  # noqa: E501
                    "file_path": file_path,
                    "total_segments_created": len(segment_paths),
                    "expected_segments": num_segments,
                    "component": "AudioConverter",
                    "method": "split_audio_into_segments",
                },
            )
        )
        return segment_paths

    except Exception as e:
        logger.exception(f"Error in split_audio_into_segments: {type(e).__name__}")
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.SYSTEM_ERROR,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "error": f"Error in split_audio_into_segments: {type(e).__name__}",
                    "file_path": file_path,
                    "num_segments": num_segments,
                    "total_duration": total_duration,
                    "exception_type": type(e).__name__,
                    "component": "AudioConverter",
                    "method": "split_audio_into_segments",
                },
            )
        )
        raise


async def get_audio_duration(file_path: str, chat_id: int = None) -> float:
    """Get audio duration in seconds using FFprobe"""
    try:
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            file_path,
        ]

        process = await asyncio.create_subprocess_exec(
            *ffprobe_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            duration = float(stdout.decode("utf-8").strip())
            return duration
        else:
            logger.warning("Could not get audio duration")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": "Could not get audio duration",
                        "file_path": file_path,
                        "ffprobe_returncode": process.returncode,
                        "stderr": stderr.decode("utf-8") if stderr else None,
                        "component": "AudioConverter",
                        "method": "get_audio_duration",
                    },
                )
            )
            return 0.0

    except Exception as e:
        logger.warning(f"Error getting audio duration: {type(e).__name__}")
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.SYSTEM_ERROR,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "error": f"Error getting audio duration: {type(e).__name__}",
                    "file_path": file_path,
                    "exception_type": type(e).__name__,
                    "component": "AudioConverter",
                    "method": "get_audio_duration",
                },
            )
        )
        return 0.0


async def download_and_convert_raw_audio_to_wav(
    file_path: str = None, sample_rate: int = 8000, chat_id: int = None
) -> str:
    """Convert raw file to wav audio using pipe streaming."""
    temp_input_path = None
    try:
        temp_input_path = file_path
        output_path = temp_input_path.replace(".raw", ".wav")
        ffmpeg_cmd = build_ffmpeg_command(temp_input_path, output_path, sample_rate)
        logger.info("Running FFmpeg conversion")
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.DATA_MODIFIED,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "message": "Running FFmpeg conversion",
                    "input_path": temp_input_path,
                    "output_path": output_path,
                    "sample_rate": sample_rate,
                    "component": "AudioConverter",
                    "method": "download_and_convert_raw_audio_to_wav",
                },
            )
        )

        # Run FFmpeg process with timeout
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=120.0
            )
        except asyncio.TimeoutError:
            logger.error("FFmpeg process timed out")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": "FFmpeg process timed out",
                        "input_path": temp_input_path,
                        "output_path": output_path,
                        "sample_rate": sample_rate,
                        "timeout_seconds": 120.0,
                        "component": "AudioConverter",
                        "method": "download_and_convert_raw_audio_to_wav",
                    },
                )
            )
            process.kill()
            raise Exception("FFmpeg conversion timed out")

        if process.returncode != 0:
            logger.error("FFmpeg conversion error")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": "FFmpeg conversion error",
                        "input_path": temp_input_path,
                        "output_path": output_path,
                        "sample_rate": sample_rate,
                        "ffmpeg_returncode": process.returncode,
                        "stderr": stderr.decode("utf-8") if stderr else None,
                        "component": "AudioConverter",
                        "method": "download_and_convert_raw_audio_to_wav",
                    },
                )
            )
            raise subprocess.CalledProcessError(
                process.returncode, "ffmpeg", "FFmpeg conversion failed"
            )

        # Check if output file exists and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception("FFmpeg output file is empty or missing")

        logger.info(
            f"Audio converted successfully " f"({os.path.getsize(output_path)} bytes)"
        )
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.DATA_MODIFIED,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "message": f"Audio converted successfully ({os.path.getsize(output_path)} bytes)",  # noqa: E501
                    "input_path": temp_input_path,
                    "output_path": output_path,
                    "output_size_bytes": os.path.getsize(output_path),
                    "sample_rate": sample_rate,
                    "component": "AudioConverter",
                    "method": "download_and_convert_raw_audio_to_wav",
                },
            )
        )

        try:
            # Clean up temp input file since it is raw file and we dont need it anymore
            # We will return the output path of wav file instead of the temp file_path
            os.unlink(temp_input_path)
            logger.info("Cleaned up temp input file")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_DELETED,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": "Cleaned up temp input file",
                        "input_path": temp_input_path,
                        "output_path": output_path,
                        "component": "AudioConverter",
                        "method": "download_and_convert_raw_audio_to_wav",
                    },
                )
            )
        except Exception as e:
            logger.warning(f"Failed to clean up temp input file: {type(e).__name__}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Failed to clean up temp input file: {type(e).__name__}",  # noqa: E501
                        "input_path": temp_input_path,
                        "output_path": output_path,
                        "exception_type": type(e).__name__,
                        "component": "AudioConverter",
                        "method": "download_and_convert_raw_audio_to_wav",
                    },
                )
            )

        return output_path

    except Exception as e:
        logger.error(
            f"Error in download_and_convert_raw_audio_to_wav: {type(e).__name__}"
        )
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.SYSTEM_ERROR,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "error": f"Error in download_and_convert_raw_audio_to_wav: {type(e).__name__}",  # noqa: E501
                    "input_path": temp_input_path,
                    "sample_rate": sample_rate,
                    "exception_type": type(e).__name__,
                    "component": "AudioConverter",
                    "method": "download_and_convert_raw_audio_to_wav",
                },
            )
        )
        raise


async def identify_audio_format(file_path: str, chat_id: int = None) -> str:
    """
    Identify audio format using FFprobe.

    Args:
        file_path (str): Path to the audio file
        chat_id (int, optional): Chat ID for PHI logging.

    Returns:
        str: File extension with dot (e.g., '.mp3', '.wav', '.raw')
    """
    try:
        # Use FFprobe to get format information
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            file_path,
        ]

        process = await asyncio.create_subprocess_exec(
            *ffprobe_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            import json

            try:
                info = json.loads(stdout.decode("utf-8"))

                # Check format name first
                format_name = info.get("format", {}).get("format_name", "").lower()

                # Map common format names to extensions
                format_mapping = {
                    "mp3": ".mp3",
                    "mpeg": ".mp3",
                    "wav": ".wav",
                    "wave": ".wav",
                    "aac": ".aac",
                    "m4a": ".m4a",
                    "mp4": ".m4a",
                    "flac": ".flac",
                    "ogg": ".ogg",
                    "opus": ".opus",
                    "webm": ".webm",
                    "wma": ".wma",
                    "amr": ".amr",
                }

                # Check if format is in our mapping
                for format_key, extension in format_mapping.items():
                    if format_key in format_name:
                        logger.info(f"Detected format: {format_name} -> {extension}")
                        await phi_logger.log(
                            PHILogEvent(
                                event_type=PHIEvents.DATA_ACCESSED,
                                chat_id=str(chat_id) if chat_id else None,
                                audit_id=None,  # Will be set by external service,
                                details={
                                    "message": f"Detected format: {format_name} -> {extension}",  # noqa: E501
                                    "file_path": file_path,
                                    "detected_format": format_name,
                                    "detected_extension": extension,
                                    "component": "AudioConverter",
                                    "method": "identify_audio_format",
                                },
                            )
                        )
                        return extension

                # If no match found, check the first audio stream codec
                streams = info.get("streams", [])
                for stream in streams:
                    if stream.get("codec_type") == "audio":
                        codec_name = stream.get("codec_name", "").lower()
                        for format_key, extension in format_mapping.items():
                            if format_key in codec_name:
                                logger.info(
                                    f"Detected codec: {codec_name} -> {extension}"
                                )
                                await phi_logger.log(
                                    PHILogEvent(
                                        event_type=PHIEvents.DATA_ACCESSED,
                                        chat_id=str(chat_id) if chat_id else None,
                                        audit_id=None,  # Will be set by external service,  # noqa: E501
                                        details={
                                            "message": f"Detected codec: {codec_name} -> {extension}",  # noqa: E501
                                            "file_path": file_path,
                                            "detected_codec": codec_name,
                                            "detected_extension": extension,
                                            "component": "AudioConverter",
                                            "method": "identify_audio_format",
                                        },
                                    )
                                )
                                return extension
                        break

                # If still no match, check file extension from URL as fallback
                logger.warning("Could not detect format from FFprobe, using fallback")
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=str(chat_id) if chat_id else None,
                        audit_id=None,  # Will be set by external service,
                        details={
                            "error": "Could not detect format from FFprobe, using fallback",  # noqa: E501
                            "file_path": file_path,
                            "format_name": format_name,
                            "streams_count": len(streams),
                            "component": "AudioConverter",
                            "method": "identify_audio_format",
                        },
                    )
                )
                return ".raw"

            except json.JSONDecodeError:
                logger.warning("Failed to parse FFprobe JSON output")
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=str(chat_id) if chat_id else None,
                        audit_id=None,  # Will be set by external service,
                        details={
                            "error": "Failed to parse FFprobe JSON output",
                            "file_path": file_path,
                            "component": "AudioConverter",
                            "method": "identify_audio_format",
                        },
                    )
                )
                return ".raw"
        else:
            logger.warning("FFprobe failed")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id) if chat_id else None,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": "FFprobe failed",
                        "file_path": file_path,
                        "ffprobe_returncode": process.returncode,
                        "stderr": stderr.decode("utf-8") if stderr else None,
                        "component": "AudioConverter",
                        "method": "identify_audio_format",
                    },
                )
            )
            return ".raw"

    except Exception as e:
        logger.warning(f"Error identifying audio format: {type(e).__name__}")
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.SYSTEM_ERROR,
                chat_id=str(chat_id) if chat_id else None,
                audit_id=None,  # Will be set by external service,
                details={
                    "error": f"Error identifying audio format: {type(e).__name__}",
                    "file_path": file_path,
                    "exception_type": type(e).__name__,
                    "component": "AudioConverter",
                    "method": "identify_audio_format",
                },
            )
        )
        return ".raw"


def build_ffmpeg_command(
    temp_input_path: str, output_path: str, sample_rate: int = 8000
) -> List[str]:
    """Build FFmpeg command based on sample rate."""
    if sample_rate in [8000, 16000]:
        # For known sample rates, use explicit format specification
        return [
            "ffmpeg",
            "-f",
            "s16le",  # Explicit format: 16-bit signed PCM
            "-ar",
            str(sample_rate),  # Input sample rate
            "-ac",
            "1",  # Explicit channels: mono
            "-i",
            temp_input_path,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",  # Output sample rate: 16kHz
            "-ac",
            "1",  # Output channels: mono
            "-f",
            "wav",
            output_path,
        ]
    else:
        # For other sample rates, use auto-detection
        return [
            "ffmpeg",
            "-i",
            temp_input_path,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",  # Output sample rate: 16kHz
            "-ac",
            "1",  # Output channels: mono
            "-f",
            "wav",
            output_path,
        ]
