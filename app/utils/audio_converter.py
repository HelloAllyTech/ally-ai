import asyncio
from app.utils.logger import get_logger
import httpx
import subprocess
import os
from datetime import datetime
import uuid

logger = get_logger(__name__)

async def save_wav_to_file(wav_data: bytes) -> str:
    """
    Save WAV data to a file with guaranteed unique filename.
    
    Args:
        wav_data (bytes): WAV audio data
                                   
    Returns:
        str: Path to the saved file
        
    Raises:
        IOError: If file writing fails
    """
    try:
        # Create guaranteed unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_id = str(uuid.uuid4())[:8]  # First 8 characters of UUID
        
        output_path = f"/tmp/audio_{timestamp}_{unique_id}.wav"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write WAV data to file
        with open(output_path, 'wb') as f:
            f.write(wav_data)
        
        logger.info(f"WAV file saved to: {output_path} ({len(wav_data)} bytes)")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving WAV file: {e}")
        raise IOError(f"Failed to save WAV file: {e}")

# Direct streaming to FFmpeg 
async def convert_and_store_raw_to_wav_with_ffmpeg_async(presigned_url: str):
    """
    Stream audio directly from URL to FFmpeg for optimal memory usage.
    
    This method downloads audio from a presigned URL and streams it directly
    to FFmpeg for conversion to WAV format, minimizing memory usage by
    avoiding storing the entire audio file in memory.
    
    Args:
        presigned_url (str): URL for the audio file
        
    Returns:
        str: Path to the saved WAV file for transcription
        
    Raises:
        httpx.HTTPError: If the HTTP request to download audio fails
        asyncio.TimeoutError: If the download or FFmpeg conversion times out
        subprocess.CalledProcessError: If FFmpeg conversion fails
        ValueError: If the audio data is invalid or corrupted
        Exception: For any other unexpected errors during processing
    """
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
            async with client.stream("GET", presigned_url) as response:
                response.raise_for_status()  # Raise exception for HTTP errors
                
                # Stream directly to FFmpeg process
                process = await asyncio.create_subprocess_exec(
                    'ffmpeg', '-i', 'pipe:0', '-acodec', 'pcm_s16le',
                    '-ar', '8000', '-ac', '1', '-f', 'wav', 'pipe:1',
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Stream audio data to FFmpeg and collect WAV output
                wav_chunks = []
                
                async def feed_ffmpeg():
                    """Feed audio data to FFmpeg stdin"""
                    try:
                        async for chunk in response.aiter_bytes():
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
                            chunk = await process.stdout.read(8192)  # 8KB chunks
                            if not chunk:
                                break
                            wav_chunks.append(chunk)
                    except Exception as e:
                        logger.error(f"Error reading from FFmpeg: {e}")
                        raise
                
                # Run both tasks concurrently
                logger.info("Starting streaming conversion to FFmpeg")
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
                    raise subprocess.CalledProcessError(
                        process.returncode, 'ffmpeg', 
                        f"FFmpeg conversion failed: {error_msg}"
                    )
                
                # Combine WAV chunks
                wav_data = b''.join(wav_chunks)
                
                # Validate WAV data
                if not wav_data or len(wav_data) < 44:  # WAV header is 44 bytes
                    raise ValueError("Invalid WAV data generated by FFmpeg")
                
                logger.info(f"Streaming conversion completed: {len(wav_data)} bytes WAV")
                
                # Save WAV data to file and get the path
                saved_file_path = await save_wav_to_file(wav_data)
                logger.info(f"WAV file saved for transcription: {saved_file_path}")
                
                # Return file path for further processing
                return saved_file_path
                
    except httpx.HTTPError as e:
        logger.error(f"HTTP error downloading audio from {presigned_url}: {e}")
        raise
    except asyncio.TimeoutError as e:
        logger.error(f"Timeout error during audio download/conversion: {e}")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid audio data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in convert_and_store_raw_to_wav_with_ffmpeg_async: {e}")
        raise
