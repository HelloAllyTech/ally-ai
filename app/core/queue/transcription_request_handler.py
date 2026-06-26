import asyncio
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import boto3
from app.core.ally_core import AllyCoreService
from app.core.config import settings
from app.core.constants import PipelineStage
from app.core.queue.message_models import (
    TranscribeAndSummarizeRequestMessage,
    TranscriptionResultMessage,
)
from app.core.text_generations.openai_text_generation_service import (
    OpenAITextGenerationService,
)
from app.core.transcriptions.services import (
    DeepgramTranscriptionService,
    OpenAITranscriptionService,
    SarvamTranscriptionService,
)
from app.schemas.common import ChatMessage
from app.core.transcriptions.utils.logger import get_logger
from app.core.transcriptions.utils.phi_events import PHIEvents
from app.core.transcriptions.utils.phi_logger import PHILogEvent, log_sync, phi_logger

logger = get_logger(__name__)

class TranscriptionServiceProvider(str, Enum):
    """Enumeration of supported transcription providers."""

    OPENAI = "openai"
    DEEPGRAM = "deepgram"
    SARVAM = "sarvam"


class PipelineStageError(Exception):
    """A processing failure attributed to a specific pipeline stage.

    Carries the stage and a human-readable upstream reason so the single
    error-reporting path can forward both to ally-core (and on to Slack)
    instead of the generic "Processing failed".
    """

    def __init__(self, stage: PipelineStage, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(f"[{stage.value}] {reason}")


class TranscriptionRequestHandler:
    """
    Handler for transcription processing.
    """

    def __init__(
        self,
        ally_core_service: AllyCoreService = None,
        text_generation_service: Optional[OpenAITextGenerationService] = None,
        transcription_service: SarvamTranscriptionService | DeepgramTranscriptionService | OpenAITranscriptionService = None
    ):
        """
        Initialize the transcription request worker.

        """
        logger.info("Initializing transcription services...")
        self.transcription_service = transcription_service
        self.ally_core_service = ally_core_service
        self.text_generation_service = text_generation_service

        logger.info("Transcription services initialized successfully")

    async def process_transcription_request(
        self, request_data: Dict[str, Any]
    ) -> bool:
        """
        Process a single transcription request.

        Args:
            request_data: The request data from SQS message

        Returns:
            bool: True if the message is fully handled and can be deleted from
            the queue (result delivered, or a processing error was reported);
            False if it should be left for SQS redrive (delivery/error-report
            could not reach ally-core, or an unexpected failure occurred).
        """
        # Tracked so a failure at any point is attributed to the stage it
        # happened in (forwarded to ally-core / Slack). Correlation id ties all
        # of this chat's log lines together across both services.
        stage = PipelineStage.REQUEST_PARSE
        correlation_id = (
            request_data.get("correlation_id")
            if isinstance(request_data, dict)
            else None
        )
        started_at = time.time()
        try:
            # Parse request message
            request = TranscribeAndSummarizeRequestMessage(**request_data)
            chat_id = request.chat_id
            correlation_id = request.correlation_id or correlation_id

            logger.info(
                f"Processing transcription request for chat_id: {chat_id} "
                f"correlation_id={correlation_id}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=request.chat_id,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "message": f"Processing transcription request for chat_id: {chat_id}",  # noqa: E501
                        "chat_id": chat_id,
                        "correlation_id": correlation_id,
                        "audio_url": request.audio_url,
                        "sample_rate": request.sample_rate,
                        "component": "TranscriptionRequestHandler",
                        "method": "process_transcription_request",
                    },
                )
            )

            # Transcribe Audio (covers download + convert + transcribe; finer
            # stage granularity is captured in the audio-converter logs).
            stage = PipelineStage.TRANSCRIBE
            logger.info(
                f"Transcribing audio for chat_id: {chat_id} "
                f"correlation_id={correlation_id}"
            )
            try:
                _, segments_text = (
                    await self.transcription_service.transcribe_audio_from_url(
                        audio_url=request.audio_url,
                        chat_id=request.chat_id,
                        sample_rate=request.sample_rate,
                        is_linear16_encoded=bool(request.is_linear16_encoded),
                    )
                )
            except Exception as e:
                raise PipelineStageError(PipelineStage.TRANSCRIBE, str(e)) from e

            # An empty transcript is a failure, not a success. STT can return
            # nothing without raising (silent/garbled audio, wrong sample rate, a
            # bad mobile linear16 upload, or a provider returning no words) — see
            # DeepgramTranscriptionService._format_deepgram_response_for_diarization.
            # Without this guard the empty result flows on to diarization+summary
            # and the chat is marked SUCCESS with a blank summary, which is the
            # "summary silently not generated" symptom. Fail it at the TRANSCRIBE
            # stage so it is reported, alertable, and (with a fallback provider
            # configured) recoverable, instead of disappearing.
            if not segments_text or not segments_text.strip():
                raise PipelineStageError(
                    PipelineStage.TRANSCRIBE,
                    "Transcription produced no text (no speech detected or "
                    "unintelligible audio)",
                )

            # Create result message
            result_message = TranscriptionResultMessage(
                chat_id=chat_id,
                segments_text=segments_text,
                timestamp=int(time.time() * 1000),
                correlation_id=correlation_id,
            )

            # Diarize + summarize + deliver. Raises PipelineStageError on a
            # genuine processing failure; a *delivery* failure of a successful
            # result is swallowed inside (never reported as an error) because
            # the result may already have landed — see
            # send_combined_result_to_ally_core. `delivered` is False when the
            # callback could not be reached after retries, in which case we
            # leave the SQS message for redrive.
            delivered = await self._process_transcription_result(
                result_message,
                session_mode=request.mode,
                correlation_id=correlation_id,
            )

            logger.info(
                f"Transcription processing completed for chat_id: "
                f"{chat_id} correlation_id={correlation_id} "
                f"delivered={delivered} "
                f"elapsed_ms={int((time.time() - started_at) * 1000)}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_MODIFIED,
                    chat_id=chat_id,
                    audit_id=None,  # Will be set by caller
                    details={
                        "message": f"Transcription processing completed successfully for chat_id: {chat_id}",  # noqa: E501
                        "chat_id": chat_id,
                        "correlation_id": correlation_id,
                        "elapsed_ms": int((time.time() - started_at) * 1000),
                        "component": "TranscriptionRequestHandler",
                        "method": "process_transcription_request",
                        "status": "success",
                    },
                )
            )
            # Delete the message only if the result actually reached ally-core;
            # otherwise leave it for SQS redrive.
            return delivered

        except PipelineStageError as e:
            chat_id = request_data.get("chat_id", "unknown")
            logger.error(
                f"Transcription failed at stage={e.stage.value} for "
                f"chat_id={chat_id} correlation_id={correlation_id}: {e.reason}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=chat_id,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Transcription failed at stage {e.stage.value}: {e.reason}",  # noqa: E501
                        "chat_id": chat_id,
                        "correlation_id": correlation_id,
                        "stage": e.stage.value,
                        "reason": e.reason,
                        "request_queue_url": settings.QUEUE.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL,  # noqa: E501
                        "component": "TranscriptionRequestHandler",
                        "method": "process_transcription_request",
                    },
                )
            )
            # A deterministic processing failure that we successfully reported
            # is terminal — delete the message. If we couldn't even report it,
            # leave it for redrive.
            reported = await self._send_error_response(
                chat_id, e.reason, stage=e.stage, correlation_id=correlation_id
            )
            return reported

        except Exception as e:
            chat_id = request_data.get("chat_id", "unknown")
            logger.exception(
                f"Error processing transcription request: chat_id={chat_id} "
                f"correlation_id={correlation_id} stage={stage.value}: {e}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=chat_id,
                    audit_id=None,  # Will be set by external service,
                    details={
                        "error": f"Error processing transcription request: {chat_id} {e}",  # noqa: E501
                        "chat_id": chat_id,
                        "correlation_id": correlation_id,
                        "stage": stage.value,
                        "reason": str(e),
                        "exception_type": type(e).__name__,
                        "request_queue_url": settings.QUEUE.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL,  # noqa: E501
                        "component": "TranscriptionRequestHandler",
                        "method": "process_transcription_request",
                    },
                )
            )
            reported = await self._send_error_response(
                chat_id, str(e), stage=stage, correlation_id=correlation_id
            )
            return reported

    async def _process_transcription_result(
        self,
        request: TranscriptionResultMessage,
        session_mode: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """
        Process the transcription result and do diarization + summary, then
        deliver the combined result to ally-core.

        Returns:
            bool: whether the combined result was delivered to ally-core
            (propagated from send_combined_result_to_ally_core; a False lets the
            caller leave the SQS message for redrive).

        Raises:
            PipelineStageError: on a genuine diarization/summary failure, tagged
                with the failing stage so it can be reported precisely. A
                *delivery* failure of a successful result is NOT raised here —
                it is handled (with retries) inside
                send_combined_result_to_ally_core, because re-posting an error
                could clobber a result that already landed.
        """
        chat_id = request.chat_id
        segments_text = request.segments_text

        logger.info(
            f"Processing diarization and summary for chat_id: {chat_id} "
            f"correlation_id={correlation_id}"
        )
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.DATA_MODIFIED,
                chat_id=str(chat_id),
                audit_id=None,  # Will be set by caller
                details={
                    "message": f"Processing diarization and summary for chat_id: {chat_id}",
                    "chat_id": chat_id,
                    "correlation_id": correlation_id,
                    "component": "TranscriptionRequestHandler",
                    "method": "_process_transcription_result",
                    "segments_text_length": (
                        len(segments_text) if segments_text else 0
                    ),
                },
            )
        )

        # Do diarization
        try:
            diarization_result = (
                await self.text_generation_service.diarize_from_transcription(
                    transcription=segments_text
                )
            )
        except Exception as e:
            logger.error(
                f"Diarization failed for chat_id {chat_id} "
                f"correlation_id={correlation_id}: {e}"
            )
            raise PipelineStageError(PipelineStage.DIARIZE, str(e)) from e

        # Convert to ChatMessage objects
        messages = [
            ChatMessage(
                role=msg.role.upper(),  # Convert to uppercase for consistency
                content=msg.content,
                start_time=msg.start_time,
                end_time=msg.end_time,
            )
            for msg in diarization_result.messages
        ]

        # Guard: a transcript with no messages (STT produced only noise, or
        # diarization yielded nothing) must NOT be delivered. If it were, the
        # backend would persist an empty transcript, summarise nothing, mark the
        # chat SUCCESS, and then DELETE the recording — leaving no transcript, no
        # summary, and no audio to recover from. Failing here keeps the chat
        # FAILED with the audio intact so a retry/reprocess can re-transcribe.
        if not messages:
            raise PipelineStageError(
                PipelineStage.TRANSCRIBE,
                "Transcript is empty after diarization (no usable speech "
                "captured); not delivering so the recording is preserved for "
                "retry",
            )

        # The transcript is ready at this point — capture it BEFORE attempting
        # the summary so a summary failure can't discard it.
        transcription_data = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg
            for msg in messages
        ]

        # PHASE 1: deliver the transcript on its own, immediately, BEFORE the
        # summary is attempted. This guarantees the transcript is persisted even
        # if summary generation is slow, hangs, or the worker/message dies before
        # the combined result is sent. The backend stores it and keeps the chat
        # IN_PROGRESS. Best-effort (never raises); the transcript is sent again
        # with the summary in phase 2, so a phase-1 delivery blip self-heals.
        phase1_delivered = await self.send_combined_result_to_ally_core(
            chat_id,
            transcription_data,
            None,
            correlation_id=correlation_id,
        )
        logger.info(
            f"Phase 1 (transcript-only) delivery for chat_id {chat_id} "
            f"correlation_id={correlation_id} delivered={phase1_delivered}"
        )

        # Generate summary. If it fails we deliver the transcript anyway (with a
        # summary_error) so the user can read the transcript and retry summary
        # generation later — instead of losing the whole session to a FAILED.
        summary_data = None
        summary_error = None
        try:
            summary = await self._generate_summary(
                messages, chat_id, session_mode=session_mode
            )
            summary_data = (
                summary.model_dump() if hasattr(summary, "model_dump") else summary
            )
        except PipelineStageError as e:
            logger.error(
                f"Summary generation failed for chat_id {chat_id} "
                f"correlation_id={correlation_id}; delivering transcript without "
                f"a summary for later retry: {e.reason}"
            )
            summary_error = e.reason

        # Deliver to ally-core. Retries internally on transient delivery
        # failure and never raises (the result is idempotent on the receiver),
        # so a slow/blipping callback can't turn a good summary into a FAILED.
        # When summary_error is set, the transcript is still delivered and the
        # backend marks the summary retryable.
        delivered = await self.send_combined_result_to_ally_core(
            chat_id,
            transcription_data,
            summary_data,
            correlation_id=correlation_id,
            summary_error=summary_error,
        )

        logger.info(
            f"Diarization and summary completed for chat_id {chat_id} "
            f"correlation_id={correlation_id} delivered={delivered}"
        )
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.DATA_MODIFIED,
                chat_id=chat_id,
                audit_id=None,
                details={
                    "message": f"Diarization and summary completed for chat_id {chat_id}",
                    "chat_id": chat_id,
                    "correlation_id": correlation_id,
                    "delivered": delivered,
                    "component": "TranscriptionRequestHandler",
                    "method": "_process_transcription_result",
                    "messages_count": len(messages),
                    "transcription_data_length": len(transcription_data),
                },
            )
        )
        return delivered

    async def _generate_summary(
        self,
        messages: List[ChatMessage],
        chat_id: int,
        session_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate summary from diarized messages.

        Parameters:
            messages (List[ChatMessage]): The diarized messages.
            chat_id (int): The chat ID.

        Returns:
            Dict[str, Any]: The summary data.
        """
        try:
            # Generate summary using text generation service
            summary = await self.text_generation_service.generate_summary_notes(
                chat_history=messages,
                keys=None,
                session_mode=session_mode,
            )

            return summary

        except Exception as e:
            logger.error(
                f"Error generating summary for chat_id {chat_id}: {e}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=str(chat_id),
                    audit_id=None,  # Will be set by caller
                    details={
                        "error": f"Error generating summary for chat_id {chat_id}: {e}",
                        "chat_id": chat_id,
                        "component": "TranscriptionRequestHandler",
                        "method": "_generate_summary",
                        "exception_type": type(e).__name__,
                        "messages_count": len(messages),
                    },
                )
            )
            # Raise tagged; the single error-reporting path in
            # process_transcription_request posts one error to ally-core. (Do
            # not post here — that produced duplicate error callbacks before.)
            raise PipelineStageError(PipelineStage.SUMMARIZE, str(e)) from e

    # Delivering a *successful* result is retried because a slow or blipping
    # callback is a transient delivery problem, NOT a processing failure. The
    # receiver is idempotent (it ignores a result once the chat is SUCCESS), so
    # re-posting the same success is safe. Critically we never downgrade a
    # delivery failure into an error POST — that is what previously flipped good
    # summaries to FAILED under load.
    _DELIVERY_MAX_ATTEMPTS = 4
    _DELIVERY_BACKOFF_SECONDS = 2

    async def send_combined_result_to_ally_core(
        self,
        chat_id: int,
        transcription: List[Dict[str, Any]],
        summary: Optional[Dict[str, Any]],
        correlation_id: Optional[str] = None,
        summary_error: Optional[str] = None,
    ) -> bool:
        """
        Send transcription (and summary when available) to ally-core's
        process-transcript API, retrying on transient delivery failures.

        When summary is None and summary_error is set, the transcript is still
        delivered and the error/stage is forwarded so the backend persists the
        transcript and marks the summary FAILED-but-retryable.

        Never raises and never posts a standalone error response: a delivery
        failure is ambiguous (the result may already have landed), so on
        exhaustion we log loudly and let SQS redrive / the backend reaper
        handle it.

        Returns:
            bool: True if the result was delivered, False if all attempts were
            exhausted (caller leaves the SQS message for redrive).
        """
        last_exception: Optional[Exception] = None
        # Forwarded only when the summary failed but the transcript is good, so
        # the backend can save the transcript and mark the summary retryable.
        stage = PipelineStage.SUMMARIZE.value if summary_error else None

        for attempt in range(1, self._DELIVERY_MAX_ATTEMPTS + 1):
            try:
                await self.ally_core_service.process_transcript(
                    chat_id=chat_id,
                    transcription=transcription,
                    summary=summary,
                    error=summary_error,
                    stage=stage,
                    correlation_id=correlation_id,
                )

                logger.info(
                    f"Sent transcription and summary to core for chat_id: "
                    f"{chat_id} correlation_id={correlation_id} (attempt {attempt})"
                )
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.DATA_MODIFIED,
                        chat_id=str(chat_id),
                        audit_id=None,
                        details={
                            "message": f"Sent transcription+summary to ally-core for chat_id: {chat_id}",  # noqa: E501
                            "chat_id": chat_id,
                            "correlation_id": correlation_id,
                            "component": "TranscriptionRequestHandler",
                            "method": "send_combined_result_to_ally_core",
                            "attempt": attempt,
                            "transcription_count": len(transcription),
                            "summary_keys": list(summary.keys()) if summary else [],
                        },
                    )
                )
                return True

            except Exception as e:
                last_exception = e
                logger.error(
                    f"Delivery of result to ally-core failed for chat_id "
                    f"{chat_id} correlation_id={correlation_id} "
                    f"(attempt {attempt}/{self._DELIVERY_MAX_ATTEMPTS}): "
                    f"{type(e).__name__}: {e}"
                )
                if attempt < self._DELIVERY_MAX_ATTEMPTS:
                    await asyncio.sleep(self._DELIVERY_BACKOFF_SECONDS * attempt)

        # Exhausted: do NOT post an error (would risk clobbering a result that
        # actually landed). Leave the chat IN_PROGRESS for the backend reaper,
        # and surface loudly so it is alertable as a delivery problem.
        logger.error(
            f"Exhausted retries delivering result to ally-core for chat_id "
            f"{chat_id} correlation_id={correlation_id}; leaving for backend "
            f"reaper. Last error: {type(last_exception).__name__ if last_exception else None}"
        )
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.SYSTEM_ERROR,
                chat_id=str(chat_id),
                audit_id=None,
                details={
                    "error": f"Failed to deliver result to ally-core for chat_id {chat_id} after {self._DELIVERY_MAX_ATTEMPTS} attempts: {last_exception}",  # noqa: E501
                    "chat_id": chat_id,
                    "correlation_id": correlation_id,
                    "stage": PipelineStage.DELIVER.value,
                    "component": "TranscriptionRequestHandler",
                    "method": "send_combined_result_to_ally_core",
                    "exception_type": type(last_exception).__name__
                    if last_exception
                    else None,
                },
            )
        )
        return False

    # Reporting the terminal FAILED state back to ally-core is the only thing
    # that moves a chat off "Processing". If this call is dropped the chat is
    # stranded until the backend's stale-chat reaper catches it, so retry a few
    # times before giving up.
    _ERROR_RESPONSE_MAX_ATTEMPTS = 3
    _ERROR_RESPONSE_BACKOFF_SECONDS = 2

    async def _send_error_response(
        self,
        chat_id: Any,
        error_message: str,
        stage: Optional[PipelineStage] = None,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """Report a genuine processing failure to ally-core, retrying on
        failure. Forwards the failing stage and upstream reason so the backend
        can categorise the failure and raise an actionable alert.

        Returns:
            bool: True if the error was reported, False if all attempts were
            exhausted (caller leaves the SQS message for redrive).
        """
        last_exception: Optional[Exception] = None
        stage_value = stage.value if stage else None

        for attempt in range(1, self._ERROR_RESPONSE_MAX_ATTEMPTS + 1):
            try:
                await self.ally_core_service.process_transcript(
                    chat_id=chat_id,
                    error=error_message,
                    stage=stage_value,
                    correlation_id=correlation_id,
                )

                logger.info(
                    f"Error response sent for chat_id: {chat_id} "
                    f"stage={stage_value} correlation_id={correlation_id}"
                )
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.DATA_MODIFIED,
                        chat_id=str(chat_id),
                        audit_id=None,
                        details={
                            "message": f"Error response sent for chat_id: {chat_id}",
                            "chat_id": chat_id,
                            "correlation_id": correlation_id,
                            "stage": stage_value,
                            "component": "TranscriptionRequestHandler",
                            "method": "_send_error_response",
                            "error_message": error_message,
                            "attempt": attempt,
                        },
                    )
                )
                return True

            except Exception as e:
                last_exception = e
                logger.error(
                    f"Failed to send error response for chat_id {chat_id} "
                    f"(attempt {attempt}/{self._ERROR_RESPONSE_MAX_ATTEMPTS}): "
                    f"{type(e).__name__}"
                )
                if attempt < self._ERROR_RESPONSE_MAX_ATTEMPTS:
                    await asyncio.sleep(self._ERROR_RESPONSE_BACKOFF_SECONDS * attempt)

        # All attempts failed: the chat will remain PENDING until the backend
        # reaper times it out (and the SQS message is left for redrive).
        # Surface this loudly so it is alertable.
        logger.error(
            f"Exhausted retries sending error response for chat_id {chat_id}; "
            f"chat will stay PENDING until the backend reaper fails it"
        )
        await phi_logger.log(
            PHILogEvent(
                event_type=PHIEvents.SYSTEM_ERROR,
                chat_id=str(chat_id),
                audit_id=None,
                details={
                    "error": f"Failed to send error response for chat_id {chat_id} after {self._ERROR_RESPONSE_MAX_ATTEMPTS} attempts: {last_exception}",
                    "chat_id": chat_id,
                    "correlation_id": correlation_id,
                    "stage": stage_value,
                    "component": "TranscriptionRequestHandler",
                    "method": "_send_error_response",
                    "exception_type": type(last_exception).__name__
                    if last_exception
                    else None,
                    "error_message": error_message,
                },
            )
        )
        return False
