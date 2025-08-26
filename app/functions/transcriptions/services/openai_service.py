"""
OpenAI transcription service for Lambda function.
Uses OpenAI Whisper for transcription and GPT for summarization.
"""

import asyncio
import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI

from utils.logger import get_logger
from core.config import settings
from utils.exceptions import TranscriptionFailedException
from utils.audio_converter import convert_and_segment_audio_async, get_audio_duration

logger = get_logger(__name__)

class OpenAITranscriptionService:
    """
    OpenAI transcription service for processing audio files.
    """
    
    def __init__(self):
        """
        Initialize the OpenAI transcription service.
        """
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            organization=settings.OPENAI_ORGANIZATION_ID
        )

    async def transcribe_audio_from_url(
        self, 
        audio_url: str,
        chat_id: int,
        sample_rate: int = 8000
    ) -> Tuple[int, str]:
        """
        Transcribe audio from URL and generate a summary.
        
        Args:
            audio_url (str): URL containing the audio file
            chat_id (int): Chat ID for the transcription session
            sample_rate (int): Expected sample rate of the audio (default: 8000)
            
        Returns:
            Tuple[int, str]: (chat_id, segments_text)
            
        Raises:
            Exception: If transcription fails
        """
        try:
            # Transcribe and preprocess audio
            segments_text = await self._transcribe_and_preprocess_audio(audio_url, sample_rate)
            
            return chat_id, segments_text
            
        except Exception as e:
            logger.error(f"Error transcribing audio from URL for chat_id {chat_id}: {str(e)}")
            raise TranscriptionFailedException(f"Transcription failed: {str(e)}")
    
    async def _transcribe_and_preprocess_audio(self, audio_url: str, sample_rate: int = 8000) -> str:
        """
        Transcribe audio and preprocess segments into a formatted string for diarization.
        
        This method supports audio segmentation for large files and processes
        segments in parallel for better performance.
        
        Args:
            audio_url (str): URL containing the audio file
            sample_rate (int): Expected sample rate of the audio (default: 8000)
            
        Returns:
            str: Formatted segments text with timing information
            
        Raises:
            Exception: If transcription fails
        """
        logger.info(f"Starting audio processing")
        segment_paths = []
        
        try:
            # Convert and segment audio if needed
            segment_paths = await convert_and_segment_audio_async(audio_url, sample_rate)
            logger.info(f"Audio converted to {len(segment_paths)} segment(s)")
            
            if len(segment_paths) == 1:
                # Single file - process normally
                segments_text = await self._transcribe_single_file(segment_paths[0])
            else:
                # Multiple segments - process in parallel
                segments_text = await self._transcribe_multiple_segments(segment_paths)
            
            return segments_text
                    
        except Exception as e:
            logger.error(f"Error transcribing and preprocessing audio: {str(e)}")
            raise TranscriptionFailedException(f"Audio transcription and preprocessing failed: {str(e)}")
        finally:
            # Clean up all segment files
            for segment_path in segment_paths:
                if await asyncio.to_thread(os.path.exists, segment_path):
                    try:
                        await asyncio.to_thread(os.remove, segment_path)
                        logger.info(f"Cleaned up segment file: {segment_path}")
                    except OSError as e:
                        logger.warning(f"Failed to cleanup segment file {segment_path}: {e}")
            
            logger.info(f"Audio processing complete")

    async def _transcribe_single_file(self, wav_file_path: str) -> str:
        """Transcribe a single WAV file"""
        try:
            with open(wav_file_path, 'rb') as audio_file:
                transcription_verbose = await asyncio.to_thread(
                    self.client.audio.transcriptions.create,
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                )
            
            logger.info("Single file transcription completed successfully")
            
            # Preprocess segments for diarization
            total_segments = len(transcription_verbose.segments)
            segments_text = "\n".join([
                f"{segment.start:.2f}-{segment.end:.2f}: {segment.text.strip()}"
                for segment in transcription_verbose.segments
            ])
            
            logger.info(f"Preprocessed {total_segments} segments for diarization")
            
            # Clean up
            del transcription_verbose
            
            return segments_text
            
        except Exception as e:
            logger.error(f"Error transcribing single file: {e}")
            raise TranscriptionFailedException(f"Error transcribing single file: {e}")

    async def _transcribe_multiple_segments(self, segment_paths: list) -> str:
        """Transcribe multiple audio segments in parallel and combine results"""
        try:
            logger.info(f"Starting parallel transcription of {len(segment_paths)} segments")
            
            # Create tasks for parallel transcription
            transcription_tasks = []
            for i, segment_path in enumerate(segment_paths):
                task = self._transcribe_segment_with_offset(segment_path, i)
                transcription_tasks.append(task)
            
            # Execute all transcriptions in parallel
            segment_results = await asyncio.gather(*transcription_tasks, return_exceptions=True)
            
            # Process results and handle any errors
            all_segments = []
            for i, result in enumerate(segment_results):
                if isinstance(result, Exception):
                    logger.error(f"Segment {i} transcription failed: {result}")
                    raise TranscriptionFailedException(f"Segment {i} transcription failed: {result}")
                all_segments.extend(result)
            
            # Sort segments by start time to maintain order
            all_segments.sort(key=lambda x: x[0])
            
            # Combine into final text
            segments_text = "\n".join([
                f"{start:.2f}-{end:.2f}: {text.strip()}"
                for start, end, text in all_segments
            ])
            
            logger.info(f"Combined {len(all_segments)} segments from {len(segment_paths)} files")
            
            return segments_text
            
        except Exception as e:
            logger.error(f"Error transcribing multiple segments: {e}")
            raise TranscriptionFailedException(f"Error transcribing multiple segments: {e}")

    async def _transcribe_segment_with_offset(self, segment_path: str, segment_index: int) -> list:
        """Transcribe a single segment and adjust timing based on segment index"""
        try:
            # Get segment duration to calculate offset
            duration = await get_audio_duration(segment_path)
            offset = segment_index * duration  # This is approximate, we'll refine it
            
            with open(segment_path, 'rb') as audio_file:
                transcription_verbose = await asyncio.to_thread(
                    self.client.audio.transcriptions.create,
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                )
            
            # Adjust timing for this segment
            adjusted_segments = []
            for segment in transcription_verbose.segments:
                adjusted_start = segment.start + offset
                adjusted_end = segment.end + offset
                adjusted_segments.append((adjusted_start, adjusted_end, segment.text))
            
            logger.info(f"Segment {segment_index}: transcribed {len(adjusted_segments)} segments")
            
            # Clean up
            del transcription_verbose
            
            return adjusted_segments
            
        except Exception as e:
            logger.error(f"Error transcribing segment {segment_index}: {e}")
            raise TranscriptionFailedException(f"Error transcribing segment {segment_index}: {e}")