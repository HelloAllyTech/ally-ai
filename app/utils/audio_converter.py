import asyncio
from app.utils.logger import get_logger
import httpx
import subprocess
from datetime import datetime
import uuid

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
            return await _convert_with_pipe_streaming(presigned_url, is_8khz=True)
        else:
            # Higher sample rate audio - use auto-detection
            logger.info(f"Using pipe streaming with auto-detection for {sample_rate} Hz audio")
            return await _convert_with_pipe_streaming(presigned_url, is_8khz=False)
            
    except Exception as e:
        logger.error(f"Error in convert_and_store_raw_to_wav_with_ffmpeg_async: {e}")
        raise

async def _convert_with_pipe_streaming(presigned_url: str, is_8khz: bool = True) -> str:
    """Convert audio using pipe streaming with format detection inside"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", presigned_url) as response:
                response.raise_for_status()
                
                # Choose FFmpeg command based on sample rate
                if is_8khz:
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
