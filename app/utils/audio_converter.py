import asyncio
from app.utils.logger import get_logger
import httpx
import subprocess
from datetime import datetime
import uuid
import os
import math
from typing import List

logger = get_logger(__name__)

def write_file(path: str, data: bytes):
    """Synchronous file write helper"""
    with open(path, 'wb') as f:
        f.write(data)

async def convert_and_store_raw_to_wav_with_ffmpeg_async(presigned_url: str, sample_rate: int = 8000):
    """
    Convert audio from presigned URL to WAV format using FFmpeg with pipe streaming.
    Uses pipe streaming for both 8kHz and higher sample rates for memory optimization.
    
    Args:
        presigned_url (str): URL for the audio file
        sample_rate (int): Expected sample rate of the audio (default: 8000)
        
    Returns:
        str: Path to the saved WAV file for transcription
    """
    
    try:
        if sample_rate == 8000:
            # 8kHz microphone audio - use explicit format specification
            logger.info("Using 8kHz pipe streaming with explicit format")
            return await _convert_with_pipe_streaming(presigned_url, frequency=8000)
        elif sample_rate == 16000:
            # 16kHz microphone audio - use explicit format specification
            logger.info("Using 16kHz pipe streaming with explicit format")
            return await _convert_with_pipe_streaming(presigned_url, frequency=16000)
        else:
            # Higher sample rate audio - use auto-detection
            logger.info(f"Using pipe streaming with auto-detection for {sample_rate} Hz audio")
            return await _convert_with_pipe_streaming(presigned_url, frequency=48000)
            
    except Exception as e:
        logger.error(f"Error in convert_and_store_raw_to_wav_with_ffmpeg_async: {e}")
        raise

async def convert_and_segment_audio_async(presigned_url: str, sample_rate: int = 8000, max_segment_size_mb: int = 24) -> List[str]:
    """
    Convert audio from presigned URL to WAV format and split into segments if needed.
    Each segment will be under the specified size limit for OpenAI API compatibility.
    
    Args:
        presigned_url (str): URL for the audio file
        sample_rate (int): Expected sample rate of the audio (default: 8000)
        max_segment_size_mb (int): Maximum size per segment in MB (default: 24MB)
        
    Returns:
        List[str]: List of paths to WAV file segments
    """
    try:
        # First convert to WAV
        wav_file_path = await convert_and_store_raw_to_wav_with_ffmpeg_async(presigned_url, sample_rate)
        
        # Check if segmentation is needed
        file_size_mb = await asyncio.to_thread(os.path.getsize, wav_file_path) / (1024 * 1024)
        logger.info(f"WAV file size: {file_size_mb:.2f} MB")
        
        if file_size_mb <= max_segment_size_mb:
            logger.info("File size is within limit, no segmentation needed")
            return [wav_file_path]
        
        # Get audio duration to calculate segment duration
        duration = await get_audio_duration(wav_file_path)
        if duration <= 0:
            logger.warning("Could not determine audio duration, using single file")
            return [wav_file_path]
        
        # Calculate number of segments needed
        # For 16kHz mono WAV: ~2.4MB per minute
        bytes_per_second = (file_size_mb * 1024 * 1024) / duration
        segment_duration = (max_segment_size_mb * 1024 * 1024) / bytes_per_second
        
        num_segments = math.ceil(duration / segment_duration)
        logger.info(f"Audio duration: {duration:.2f}s, will split into {num_segments} segments of ~{segment_duration:.2f}s each")
        
        # Split audio into segments
        segment_paths = await split_audio_into_segments(wav_file_path, num_segments, duration)
        
        # Clean up original file
        try:
            # await asyncio.to_thread(os.remove, wav_file_path)
            logger.info(f"Cleaned up original WAV file: {wav_file_path}")
        except OSError as e:
            logger.warning(f"Failed to cleanup original WAV file {wav_file_path}: {e}")
        
        return segment_paths
        
    except Exception as e:
        logger.error(f"Error in convert_and_segment_audio_async: {e}")
        raise

async def split_audio_into_segments(wav_file_path: str, num_segments: int, total_duration: float) -> List[str]:
    """
    Split a WAV file into multiple segments using FFmpeg.
    
    Args:
        wav_file_path (str): Path to the input WAV file
        num_segments (int): Number of segments to create
        total_duration (float): Total duration of the audio in seconds
        
    Returns:
        List[str]: List of paths to segment files
    """
    try:
        segment_duration = total_duration / num_segments
        segment_paths = []
        
        # Create base path for segments
        base_path = os.path.splitext(wav_file_path)[0]
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, total_duration)
            
            # Create segment file path
            segment_path = f"{base_path}_segment_{i:03d}.wav"
            
            # Use FFmpeg to extract segment
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', wav_file_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',  # Overwrite output file
                segment_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                logger.error(f"FFmpeg segmentation failed for segment {i}: {error_msg}")
                raise subprocess.CalledProcessError(
                    process.returncode, 'ffmpeg', 
                    f"FFmpeg segmentation failed for segment {i}: {error_msg}"
                )
            
            # Verify segment file size
            segment_size_mb = await asyncio.to_thread(os.path.getsize, segment_path) / (1024 * 1024)
            logger.info(f"Created segment {i}: {segment_path} ({segment_size_mb:.2f} MB, {start_time:.2f}s - {end_time:.2f}s)")
            
            segment_paths.append(segment_path)
        
        logger.info(f"Successfully created {len(segment_paths)} audio segments")
        return segment_paths
        
    except Exception as e:
        logger.error(f"Error in split_audio_into_segments: {e}")
        raise

async def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds using FFprobe"""
    try:
        ffprobe_cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0',
            file_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *ffprobe_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            duration = float(stdout.decode('utf-8').strip())
            return duration
        else:
            logger.warning(f"Could not get audio duration: {stderr.decode('utf-8')}")
            return 0.0
            
    except Exception as e:
        logger.warning(f"Error getting audio duration: {e}")
        return 0.0

async def _convert_with_pipe_streaming(presigned_url: str, frequency: int = 8000) -> str:
    """Convert audio using pipe streaming with format detection inside"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", presigned_url) as response:
                response.raise_for_status()
                
                # Choose FFmpeg command based on sample rate
                if frequency == 8000:
                    # 8kHz microphone audio - explicit format specification
                    ffmpeg_cmd = [
                        'ffmpeg', 
                        '-f', 's16le',        # Explicit format: 16-bit signed PCM
                        '-ar', '8000',        # Explicit sample rate: 8kHz
                        '-ac', '1',           # Explicit channels: mono
                        '-i', 'pipe:0',       # Input from pipe
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',       # Output sample rate: 16kHz
                        '-ac', '1',           # Output channels: mono
                        '-f', 'wav', 
                        'pipe:1'
                    ]
                elif frequency == 16000:
                    # 16kHz microphone audio - explicit format specification
                    ffmpeg_cmd = [
                        'ffmpeg', 
                        '-f', 's16le',        # Explicit format: 16-bit signed PCM
                        '-ar', '16000',        # Explicit sample rate: 16kHz
                        '-ac', '1',           # Explicit channels: mono
                        '-i', 'pipe:0',       # Input from pipe
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',       # Output sample rate: 16kHz
                        '-ac', '1',           # Output channels: mono
                        '-f', 'wav', 
                        'pipe:1'
                    ]
                else:
                    # Higher sample rate audio - auto-detection
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-i', 'pipe:0',       # Auto-detect input format
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',       # Output sample rate: 16kHz
                        '-ac', '1',           # Output channels: mono
                        '-f', 'wav',
                        'pipe:1'
                    ]
                
                # Stream directly to FFmpeg process
                process = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Stream audio data to FFmpeg and collect WAV output
                wav_chunks = []
                
                async def feed_ffmpeg():
                    """Feed audio data to FFmpeg stdin"""
                    try:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            process.stdin.write(chunk)
                            await process.stdin.drain()
                        process.stdin.close()
                        await process.stdin.wait_closed()
                    except Exception as e:
                        logger.error(f"Error feeding data to FFmpeg: {e}")
                        raise
                
                async def read_ffmpeg():
                    """Read WAV output from FFmpeg stdout"""
                    try:
                        while True:
                            chunk = await process.stdout.read(8192)
                            if not chunk:
                                break
                            wav_chunks.append(chunk)
                    except Exception as e:
                        logger.error(f"Error reading from FFmpeg: {e}")
                        raise
                
                # Run both tasks concurrently
                logger.info(f"Starting pipe streaming conversion: {' '.join(ffmpeg_cmd)}")
                await asyncio.gather(
                    feed_ffmpeg(),
                    read_ffmpeg()
                )
                
                # Wait for FFmpeg process to complete
                await process.wait()
                
                # Check if FFmpeg process was successful
                if process.returncode != 0:
                    stderr = await process.stderr.read()
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    logger.error(f"FFmpeg stderr: {error_msg}")
                    raise subprocess.CalledProcessError(
                        process.returncode, 'ffmpeg', 
                        f"FFmpeg conversion failed: {error_msg}"
                    )
                
                # Combine WAV chunks
                wav_data = b''.join(wav_chunks)
                
                # Validate WAV data
                if not wav_data or len(wav_data) < 44:  # WAV header is 44 bytes
                    raise ValueError("Invalid WAV data generated by FFmpeg")
                
                logger.info(f"Pipe streaming conversion completed: {len(wav_data)} bytes WAV")
                
                # Save WAV data to file
                timestamp = await asyncio.to_thread(datetime.now().strftime, "%Y%m%d_%H%M%S_%f")
                unique_id = str(uuid.uuid4())[:8]
                output_path = f"/tmp/audio_{timestamp}_{unique_id}.wav"
                
                await asyncio.to_thread(write_file, output_path, wav_data)
                logger.info(f"WAV file saved for transcription: {output_path}")
                
                return output_path
                
    except Exception as e:
        logger.error(f"Error in _convert_with_pipe_streaming: {e}")
        raise
