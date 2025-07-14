import asyncio
from openai import OpenAI
from app.core.config import settings
from app.utils.logger import get_logger
from app.schemas.common import ChatMessage
from app.core.transcriptions.base import BaseTranscriptionService
from app.core.constants import TranscriptionConstants
from app.exceptions.custom_exceptions import TranscriptionFailedException
from app.utils.audio_converter import convert_and_store_raw_to_wav_with_ffmpeg_async
import os

logger = get_logger(__name__)

# Semaphore to limit concurrent transcription requests
TRANSCRIPTION_SEMAPHORE = asyncio.Semaphore(2)


class OpenAITranscriptionService(BaseTranscriptionService[OpenAI]):
    """OpenAI transcription service for transcribing audio files using OpenAI's audio API."""
    
    def __init__(self, text_generation_service):
        """
        Initialize the OpenAI transcription service.
        """
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            organization=settings.OPENAI_ORGANIZATION_ID
        )
        super().__init__(self.client, text_generation_service)

    async def transcribe_audio_from_url(
        self, 
        presigned_url: str,
        chat_id: int,
        sample_rate: int = 8000
    ) -> bool:
        """
        Transcribe audio from URL and generate a summary.
        
        Args:
            presigned_url (str): URL containing the audio file
            chat_id (int): Chat ID for the transcription session
            sample_rate (int): Expected sample rate of the audio (default: 8000)
            
        Returns:
            bool: True if transcription and summarization was successful
            
        Raises:
            Exception: If transcription fails
        """
        try:
            # Transcribe and preprocess audio
            segments_text = await self._transcribe_and_preprocess_audio(presigned_url, sample_rate)
            # Use OpenAI structured output to diarize the transcription
            diarization_result = await self.text_generation_service.diarize_from_transcription(transcription=segments_text)
            messages = [
                ChatMessage(
                    role=msg.role.upper(),  # Convert to uppercase for consistency
                    content=msg.content,
                    start_time=msg.start_time,
                    end_time=msg.end_time
                )
                for msg in diarization_result.messages
            ]
            # Send transcript to core
            await self._send_transcript_to_core(chat_id, messages)

            # Generate summary
            summary = await self._generate_summary(messages, chat_id)
            # Send summary to core
            await self._send_summary_to_core(chat_id, summary)
            return True
            
        except Exception as e:
            logger.error(f"Error transcribing audio from URL for chat_id {chat_id}: {str(e)}")
            raise Exception(f"Transcription failed: {str(e)}")
        
    async def _transcribe_and_preprocess_audio(self, presigned_url: str, sample_rate: int = 8000) -> str:
        """
        Transcribe audio and preprocess segments into a formatted string for diarization.
        
        This method uses a semaphore to limit concurrent audio processing to 2 at a time
        to prevent resource exhaustion during download and FFmpeg conversion.
        
        Args:
            presigned_url (str): URL containing the audio file
            sample_rate (int): Expected sample rate of the audio (default: 8000)
            
        Returns:
            str: Formatted segments text with timing information
            
        Raises:
            Exception: If transcription fails
        """
        async with TRANSCRIPTION_SEMAPHORE:
            logger.info(f"Starting audio processing (semaphore acquired)")
            wav_file_path = None
            transcription_verbose = None
            
            try:
                
                # Convert raw audio buffer to WAV format using FFmpeg (async)
                wav_file_path = await convert_and_store_raw_to_wav_with_ffmpeg_async(presigned_url, sample_rate)
                logger.debug(f"WAV file created at: {wav_file_path}")
                
                # Create transcription asynchronously using asyncio.to_thread
                with open(wav_file_path, 'rb') as audio_file:
                    transcription_verbose = await asyncio.to_thread(
                        self.client.audio.transcriptions.create,
                        model=TranscriptionConstants.MODEL,
                        file=audio_file,
                        response_format="verbose_json",
                    )
                logger.info("Audio transcription completed successfully")
                
                # Preprocess segments for diarization efficiently
                total_segments = len(transcription_verbose.segments)
                
                # Use list comprehension for better performance
                segments_text = "\n".join([
                    f"{segment.start:.2f}-{segment.end:.2f}: {segment.text.strip()}"
                    for segment in transcription_verbose.segments
                ])
                
                logger.info(f"Preprocessed {total_segments} segments for diarization")
                
                # Clean up transcription object
                del transcription_verbose
                transcription_verbose = None
                
                return segments_text
                        
            except Exception as e:
                logger.error(f"Error transcribing and preprocessing audio: {str(e)}")
                raise TranscriptionFailedException(
                    message="Audio transcription and preprocessing failed",
                    audio_source=presigned_url,
                    error_details=str(e)
                )
            finally:
                # Clean up WAV file after transcription is complete
                if wav_file_path and os.path.exists(wav_file_path):
                    try:
                        os.remove(wav_file_path)
                        logger.info(f"Cleaned up WAV file: {wav_file_path}")
                    except OSError as e:
                        logger.warning(f"Failed to cleanup WAV file {wav_file_path}: {e}")
                
                # Clean up transcription object if it exists
                if transcription_verbose is not None:
                    del transcription_verbose
                
                logger.info(f"Released semaphore (audio processing complete)")


    # The following methods are inherited from BaseTranscriptionService:
    # - _send_transcript_to_core
    # - _send_summary_to_core

