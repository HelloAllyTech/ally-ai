import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from deepgram.clients.listen.v1.rest import Results

from app.core.config import settings
from app.utils.logger import get_logger
from app.schemas.common import ChatMessage
from app.core.transcriptions.base import BaseTranscriptionService
from app.exceptions.custom_exceptions import TranscriptionFailedException
from app.utils.audio_converter import convert_and_store_raw_to_wav_with_ffmpeg_async, get_audio_duration

logger = get_logger(__name__)

# Semaphore to limit concurrent transcription requests
TRANSCRIPTION_SEMAPHORE = asyncio.Semaphore(3)

# Sentence pause threshold in seconds
SENTENCE_PAUSE_THRESHOLD = 0.1

# Message merging threshold in seconds - merge messages from same speaker if gap is less than this
MESSAGE_MERGE_THRESHOLD = 3.0


class DeepgramTranscriptionService(BaseTranscriptionService):
    """
    Deepgram transcription service for transcribing audio files using Deepgram's SDK.
    """

    def __init__(self, text_generation_service):
        """
        Initialize the Deepgram transcription service.
        """
        self.deepgram = DeepgramClient(settings.DEEPGRAM_API_KEY)
        super().__init__(None, text_generation_service)

    async def transcribe_audio_from_url(
        self, 
        audio_url: str,
        chat_id: int,
        sample_rate: int = 8000
    ) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Transcribe audio from URL and generate a summary.
        
        Args:
            audio_url (str): URL containing the audio file
            chat_id (int): Chat ID for the transcription session
            sample_rate (int): Expected sample rate of the audio (default: 8000)
            
        Returns:
            Tuple[int, List[Dict[str, Any]], Dict[str, Any]]: (chat_id, transcription_data, summary_data)
            
        Raises:
            Exception: If transcription fails
        """
        try:
            # Download and convert audio to WAV format
            wav_file_path = await convert_and_store_raw_to_wav_with_ffmpeg_async(
                audio_url, 
                sample_rate=sample_rate
            )
            
            logger.info(f"Audio converted and saved to: {wav_file_path}")
            
            # Transcribe using Deepgram SDK and get formatted segments
            segments_text = await self._transcribe_with_deepgram_sdk(wav_file_path)
            
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
            
            # Merge consecutive messages from the same speaker if they're close together in time
            messages = self._merge_consecutive_messages(messages)

            # Clean up temporary WAV file
            try:
                if os.path.exists(wav_file_path):
                    os.remove(wav_file_path)
                    logger.info(f"Cleaned up temporary file: {wav_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file {wav_file_path}: {cleanup_error}")
            
            # Generate summary
            summary = await self._generate_summary(messages, chat_id)
            
            # Return the data instead of sending to core
            transcription_data = [msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in messages]
            summary_data = summary.model_dump() if hasattr(summary, 'model_dump') else summary
            return chat_id, transcription_data, summary_data
            
        except Exception as e:
            logger.error(f"Error transcribing audio from URL for chat_id {chat_id}: {str(e)}")
            raise Exception(f"Transcription failed: {str(e)}")

    async def _transcribe_with_deepgram_sdk(self, wav_file_path: str) -> str:
        """
        Transcribe audio using Deepgram SDK and format into segments text for OpenAI diarization.
        
        Args:
            wav_file_path (str): Path to the local WAV file
            
        Returns:
            str: Formatted segments text with timing information for diarization
        """
        async with TRANSCRIPTION_SEMAPHORE:
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

    def _merge_consecutive_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Merge consecutive messages from the same speaker if they're close together in time.
        
        Args:
            messages: List of ChatMessage objects
            
        Returns:
            List of merged ChatMessage objects
        """
        merged_messages = []
        current_message = None
        
        for message in messages:
            if current_message is None:
                current_message = message
            elif (message.role == current_message.role and 
                  message.start_time - current_message.end_time <= MESSAGE_MERGE_THRESHOLD):
                # Merge messages
                current_message.content += " " + message.content
                current_message.end_time = message.end_time
            else:
                # Add current message to list and start new one
                merged_messages.append(current_message)
                current_message = message
        
        # Add the last message
        if current_message:
            merged_messages.append(current_message)
        
        return merged_messages
