import asyncio
import math
import os
import subprocess
import tempfile
from typing import List

import httpx
from utils.logger import get_logger

logger = get_logger(__name__)


async def download_file_to_temp_and_get_details(audio_url: str) -> tuple[str, str]:
    """
    Download audio file to temporary location and identify its actual format.

    Args:
        audio_url (str): URL of the audio file

    Returns:
        tuple[str, str]: Tuple containing path to the temporary downloaded
        file and its detected extension
    """
    try:
        logger.info("Downloading audio file")
        # Download the file first with a generic extension
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(audio_url)
            response.raise_for_status()

            # Create temporary input file with generic extension first
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
            temp_input.write(response.content)
            temp_input.close()

            logger.info(f"Downloaded audio file ({len(response.content)} bytes)")

            # Now identify the actual format
            actual_extension = await identify_audio_format(temp_input.name)

            # Rename the file with the correct extension
            correct_path = temp_input.name.replace(".audio", actual_extension)
            try:
                await asyncio.to_thread(os.rename, temp_input.name, correct_path)
                logger.info(f"Renamed file (detected format: {actual_extension})")
                return correct_path, actual_extension
            except OSError as e:
                logger.warning(
                    f"Failed to rename file, using original: {type(e).__name__}"
                )
                return temp_input.name, actual_extension

    except Exception as e:
        logger.error(f"Error downloading audio: {type(e).__name__}")
        raise


async def convert_and_segment_audio_async(
    audio_url: str, sample_rate: int = 8000, max_segment_size_mb: int = 24
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

    Returns:
        List[str]: List of paths to audio file segments
    """
    try:

        temp_downloaded_file_path, detected_extension = (
            await download_file_to_temp_and_get_details(audio_url)
        )

        if detected_extension == ".raw":
            file_path = await download_and_convert_raw_audio_to_wav(
                temp_downloaded_file_path, sample_rate
            )
        else:
            file_path = temp_downloaded_file_path

        # Check if segmentation is needed
        file_size_mb = await asyncio.to_thread(os.path.getsize, file_path) / (
            1024 * 1024
        )
        logger.info(f"file size: {file_size_mb:.2f} MB")

        if file_size_mb <= max_segment_size_mb:
            logger.info("File size is within limit, no segmentation needed")
            return [file_path]

        # Get audio duration to calculate segment duration
        duration = await get_audio_duration(file_path)
        if duration <= 0:
            logger.warning("Could not determine audio duration, using single file")
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

        # Split audio into segments
        segment_paths = await split_audio_into_segments(
            file_path, num_segments, duration
        )

        # Clean up original file
        try:
            await asyncio.to_thread(os.remove, file_path)
            logger.info("Cleaned up original file")
        except OSError as e:
            logger.warning(f"Failed to cleanup original file: {type(e).__name__}")

        return segment_paths

    except Exception as e:
        logger.error(f"Error in convert_and_segment_audio_async: {type(e).__name__}")
        raise


async def split_audio_into_segments(
    file_path: str, num_segments: int, total_duration: float
) -> List[str]:
    """
    Split a file into multiple segments using FFmpeg.

    Args:
        file_path (str): Path to the input file
        num_segments (int): Number of segments to create
        total_duration (float): Total duration of the audio in seconds

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

        segment_paths.append(segment_path)

        logger.info(f"Successfully created {len(segment_paths)} audio segments")
        return segment_paths

    except Exception as e:
        logger.exception(f"Error in split_audio_into_segments: {type(e).__name__}")
        raise


async def get_audio_duration(file_path: str) -> float:
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
            return 0.0

    except Exception as e:
        logger.warning(f"Error getting audio duration: {type(e).__name__}")
        return 0.0


async def download_and_convert_raw_audio_to_wav(
    file_path: str = None, sample_rate: int = 8000
) -> str:
    """Convert raw file to wav audio using pipe streaming."""
    temp_input_path = None
    try:
        temp_input_path = file_path
        output_path = temp_input_path.replace(".raw", ".wav")
        ffmpeg_cmd = build_ffmpeg_command(temp_input_path, output_path, sample_rate)
        logger.info("Running FFmpeg conversion")

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
            process.kill()
            raise Exception("FFmpeg conversion timed out")

        if process.returncode != 0:
            logger.error("FFmpeg conversion error")
            raise subprocess.CalledProcessError(
                process.returncode, "ffmpeg", "FFmpeg conversion failed"
            )

        # Check if output file exists and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception("FFmpeg output file is empty or missing")

        logger.info(
            f"Audio converted successfully " f"({os.path.getsize(output_path)} bytes)"
        )

        try:
            # Clean up temp input file since it is raw file and we dont need it anymore
            # We will return the output path of wav file instead of the temp file_path
            os.unlink(temp_input_path)
            logger.info("Cleaned up temp input file")
        except Exception as e:
            logger.warning(f"Failed to clean up temp input file: {type(e).__name__}")

        return output_path

    except Exception as e:
        logger.error(
            f"Error in download_and_convert_raw_audio_to_wav: {type(e).__name__}"
        )
        raise


async def identify_audio_format(file_path: str) -> str:
    """
    Identify audio format using FFprobe.

    Args:
        file_path (str): Path to the audio file

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
                                return extension
                        break

                # If still no match, check file extension from URL as fallback
                logger.warning("Could not detect format from FFprobe, using fallback")
                return ".raw"

            except json.JSONDecodeError:
                logger.warning("Failed to parse FFprobe JSON output")
                return ".raw"
        else:
            logger.warning("FFprobe failed")
            return ".raw"

    except Exception as e:
        logger.warning(f"Error identifying audio format: {type(e).__name__}")
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
