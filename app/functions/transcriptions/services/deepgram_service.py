"""
Deepgram transcription service for Lambda function.
Uses Deepgram Nova-3 for transcription and OpenAI GPT for summarization.
"""

import asyncio
import os
from typing import List, Dict, Any, Tuple
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from deepgram.clients.listen.v1.rest import Results
from openai import OpenAI

from utils.logger import get_logger
from core.config import settings
from core.schemas import ChatMessage
from utils.exceptions import TranscriptionFailedException
from utils.audio_converter import convert_and_segment_audio_async, get_audio_duration

logger = get_logger(__name__)

# Sentence pause threshold in seconds
SENTENCE_PAUSE_THRESHOLD = 0.1

# Message merging threshold in seconds - merge messages from same speaker if gap is less than this
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
        self.deepgram = DeepgramClient(settings.DEEPGRAM_API_KEY)
    
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
            # Download and convert audio to WAV format
            segment_paths = await convert_and_segment_audio_async(
                audio_url=audio_url, 
                sample_rate=sample_rate,
                max_segment_size_mb=MAX_SEGMENT_SIZE_MB
            )
            
            logger.info(f"Audio converted into {len(segment_paths)} segments")
            
            # Transcribe using appropriate method based on number of segments
            if len(segment_paths) == 1:
                # Single segment - process directly
                segments_text = await self._transcribe_with_deepgram_sdk(segment_paths[0])
            else:
                # Multiple segments - process in parallel
                segments_text = await self._transcribe_multiple_segments(segment_paths)
            
            return chat_id, segments_text
            
            # Clean up temporary WAV file
            try:
                if os.path.exists(wav_file_path):
                    os.remove(wav_file_path)
                    logger.info(f"Cleaned up temporary file: {wav_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file {wav_file_path}: {cleanup_error}")
            
        except Exception as e:
            logger.error(f"Error transcribing audio from URL for chat_id {chat_id}: {str(e)}")
            raise TranscriptionFailedException(f"Error transcribing audio from URL for chat_id {chat_id}: {str(e)}")
    
    async def _transcribe_with_deepgram_sdk(self, wav_file_path: str) -> str:
        """
        Transcribe audio using Deepgram SDK and format into segments text for OpenAI diarization.
        
        Args:
            wav_file_path (str): Path to the local WAV file
            
        Returns:
            str: Formatted segments text with timing information for diarization
        """
        try:
            logger.info(f"Starting Deepgram SDK transcription for file: {wav_file_path}")
            
            # Read the audio file
            with open(wav_file_path, "rb") as audio_file:
                buffer_data = audio_file.read()

            # Create file source
            payload: FileSource = {
                "buffer": buffer_data,
            }

            # Configure Deepgram options - disable diarization since we'll use OpenAI for that
            options = PrerecordedOptions(
                model="nova-3",
                language="multi",  # Support multiple languages
                smart_format=True,
                punctuate=True,
                diarize=False,  # Disable Deepgram diarization
                utterances=True,
                utt_split=0.8,  # Split utterances at 0.8 second pauses
                paragraphs=False,
                summarize=False
            )

            logger.info("Sending request to Deepgram SDK")
            
            # Make the transcription request
            response = await self.deepgram.listen.asyncrest.v("1").transcribe_file(
                payload, options
            )

            if not response or not hasattr(response, 'results'):
                logger.error("Invalid response from Deepgram SDK")
                raise TranscriptionFailedException("Invalid response from Deepgram SDK")
            
            logger.info("Received transcription response from Deepgram SDK")
            
            # Format the transcription into segments text similar to OpenAI format
            segments_text = self._format_deepgram_response_for_diarization(response.results)
            
            return segments_text
                
        except Exception as e:
            logger.exception(f"Error during Deepgram SDK transcription: {e}")
            raise TranscriptionFailedException(f"Deepgram SDK transcription error: {str(e)}")

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
                    raise result
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
            raise

    async def _transcribe_segment_with_offset(self, segment_path: str, segment_index: int) -> list:
        """Transcribe a single segment and adjust timing based on segment index"""
        try:
            # Get segment duration to calculate offset
            duration = await get_audio_duration(segment_path)
            offset = segment_index * duration  # This is approximate, we'll refine it
            
            # Transcribe the segment using Deepgram SDK
            segments_text = await self._transcribe_with_deepgram_sdk(segment_path)
            
            # Parse the segments text to extract timing and content
            adjusted_segments = []
            for line in segments_text.split('\n'):
                if line.strip() and '-' in line and ':' in line:
                    try:
                        # Parse format: "start-end: text"
                        time_part, text_part = line.split(':', 1)
                        start_str, end_str = time_part.split('-')
                        start_time = float(start_str.strip())
                        end_time = float(end_str.strip())
                        text = text_part.strip()
                        
                        # Adjust timing for this segment
                        adjusted_start = start_time + offset
                        adjusted_end = end_time + offset
                        adjusted_segments.append((adjusted_start, adjusted_end, text))
                        
                    except (ValueError, IndexError) as parse_error:
                        logger.warning(f"Failed to parse segment line: {line} - {parse_error}")
                        continue
            
            logger.info(f"Segment {segment_index}: transcribed {len(adjusted_segments)} segments with offset {offset:.2f}s")
            
            return adjusted_segments
            
        except Exception as e:
            logger.error(f"Error transcribing segment {segment_index}: {e}")
            raise

    async def _cleanup_segment_files(self, segment_paths: list):
        """Clean up temporary segment files"""
        for segment_path in segment_paths:
            try:
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                    logger.info(f"Cleaned up segment file: {segment_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up segment file {segment_path}: {cleanup_error}")

    def _format_deepgram_response_for_diarization(self, results: Results) -> str:
        """
        Format Deepgram response into segments text similar to OpenAI transcription service format.
        
        Args:
            results: Deepgram Results object from SDK response
            
        Returns:
            str: Formatted segments text with timing information
        """
        try:
            if not results.channels or len(results.channels) == 0:
                logger.warning("No channels found in Deepgram response")
                return ""
            
            channel = results.channels[0]
            if not channel.alternatives or len(channel.alternatives) == 0:
                logger.warning("No alternatives found in Deepgram response")
                return ""
            
            alternative = channel.alternatives[0]
            if not alternative.words:
                logger.warning("No words found in Deepgram response")
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
                if (current_segment_words and 
                    word_start - current_segment_words[-1]['end'] > SENTENCE_PAUSE_THRESHOLD):
                    
                    # Finalize current segment
                    segment_text = " ".join([w['word'] for w in current_segment_words])
                    segment_end = current_segment_words[-1]['end']
                    segments.append({
                        'start': current_start,
                        'end': segment_end,
                        'text': segment_text.strip()
                    })
                    
                    # Start new segment
                    current_segment_words = []
                    current_start = word_start
                
                current_segment_words.append({
                    'word': word_text,
                    'start': word_start,
                    'end': word_end
                })
            
            # Add the last segment
            if current_segment_words:
                segment_text = " ".join([w['word'] for w in current_segment_words])
                segment_end = current_segment_words[-1]['end']
                segments.append({
                    'start': current_start,
                    'end': segment_end,
                    'text': segment_text.strip()
                })
            
            # Format segments similar to OpenAI transcription service
            segments_text = "\n".join([
                f"{segment['start']:.2f}-{segment['end']:.2f}: {segment['text']}"
                for segment in segments if segment['text'].strip()
            ])
            
            logger.info(f"Formatted {len(segments)} segments for diarization")
            return segments_text
            
        except Exception as e:
            logger.exception(f"Error formatting Deepgram response for diarization: {e}")
            raise TranscriptionFailedException(f"Error formatting response: {str(e)}")
