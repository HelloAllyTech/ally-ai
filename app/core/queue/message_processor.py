import asyncio
import json
from typing import Any, Awaitable, Callable, Dict

from app.core.phi_events import PHIEvents
from app.core.phi_logger import PHILogEvent, phi_logger
from app.core.queue.sqs_queue_service import SQSQueueService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MessageProcessor:
    """
    Simplified message processor for continuously polling a queue for messages and
    processing them.
    """

    def __init__(
        self,
        queue_service: SQSQueueService,
        handler: Callable[[Dict[str, Any]], Awaitable[None]],
        queue_url: str,
        max_messages: int = 10,
        wait_time_seconds: int = 20,
        visibility_timeout: int = 30,
        polling_interval: int = 0,
        delete_after_processing: bool = True,
        max_concurrent_messages: int = 3,
    ):
        """
        Initialize the message processor.

        Parameters:
            queue_service (SQSQueueService): The queue service to use.
            handler (Callable): The handler function to process messages.
            queue_url (str): The URL of the queue to poll.
            max_messages (int): The maximum number of messages to receive in each batch.
            wait_time_seconds (int): The duration (in seconds) for which the call
                waits for a message to arrive.
            visibility_timeout (int): The duration (in seconds) that the received
                messages are hidden from subsequent retrieve requests.
            polling_interval (int): The interval (in seconds) between polling
                attempts. 0 means continuous polling.
            delete_after_processing (bool): Whether to automatically delete messages
            after successful processing.
            max_concurrent_messages (int): Upper bound on messages processed in
                parallel per poll cycle. Each processed message may hold an
                audio blob + LLM call in memory; without this bound the worker
                memory-spikes linearly with max_messages.
        """
        self.queue_service = queue_service
        self.handler = handler
        self.queue_url = queue_url
        self.max_messages = max_messages
        self.wait_time_seconds = wait_time_seconds
        self.visibility_timeout = visibility_timeout
        self.polling_interval = polling_interval
        self.delete_after_processing = delete_after_processing
        self.max_concurrent_messages = max(1, max_concurrent_messages)
        self._running = False
        self._task = None
        self._process_semaphore = asyncio.Semaphore(self.max_concurrent_messages)

    async def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a single message.

        Parameters:
            message (Dict[str, Any]): The message to process.
        """
        chat_id = None
        receipt_handle = message.get("receipt_handle")
        # Whether the message was fully handled and is safe to delete. A handler
        # may return False to signal "leave for SQS redrive" (e.g. the result
        # could not be delivered to the downstream service); a handler that
        # returns None/True (or has no opinion) is treated as handled. An
        # unhandled exception leaves this False so the message redrives.
        handled = False

        try:
            # Extract the message body
            body = message.get("body", {})

            # Extract chat_id early, before any processing
            if not isinstance(body, dict):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    # Set chat_id to unknown if JSON parsing fails
                    chat_id = "unknown"
                    logger.error("Failed to parse message body as JSON")
                    await phi_logger.log(
                        PHILogEvent(
                            event_type=PHIEvents.SYSTEM_ERROR,
                            chat_id=chat_id,
                            audit_id=None,  # Will be set by external service
                            details={
                                "error": "Failed to parse message body as JSON",
                                "message_id": message.get("message_id", "unknown"),
                                "receipt_handle": (
                                    receipt_handle[:20] + "..."
                                    if receipt_handle
                                    else None
                                ),
                                "queue_url": self.queue_url,
                            },
                        )
                    )

            # Proceed only if body is a valid dict; otherwise skip handler
            if isinstance(body, dict):
                # Extract chat_id from valid body
                chat_id = body.get("chat_id", "unknown")
                correlation_id = body.get("correlation_id")
                logger.info(
                    f"Processing message for chat_id: {chat_id} "
                    f"correlation_id={correlation_id}"
                )

                # Process the message using the handler
                result = await self.handler(body)
                # Only an explicit False means "leave for redrive".
                handled = result is not False
            else:
                # Malformed body (already logged above): nothing to retry, drop
                # it so it doesn't loop forever.
                handled = True

        except Exception as e:
            chat_id = (
                body.get("chat_id", "unknown") if "body" in locals() else "unknown"
            )
            logger.exception(
                f"Error processing message for chat_id {chat_id}: {type(e).__name__}"
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=chat_id,
                    audit_id=None,  # Will be set by external service
                    details={
                        "error": (
                            f"Error processing message for chat_id {chat_id}: "
                            f"{type(e).__name__}"
                        ),
                        "chat_id": chat_id,
                        "exception_type": type(e).__name__,
                        "message_id": message.get("message_id", "unknown"),
                        "receipt_handle": (
                            receipt_handle[:20] + "..." if receipt_handle else None
                        ),
                        "queue_url": self.queue_url,
                    },
                )
            )

        finally:
            # Delete only when the message was fully handled. An unhandled
            # failure (handler returned False or raised) is left on the queue so
            # SQS redelivers it after the visibility timeout and, after
            # maxReceiveCount, routes it to the DLQ — instead of silently
            # dropping work, which is what the old unconditional delete did.
            if not handled and receipt_handle:
                logger.warning(
                    f"Leaving message on queue for redrive (chat_id={chat_id}); "
                    f"not deleting after a failed/incomplete handling"
                )
            if self.delete_after_processing and receipt_handle and handled:
                try:
                    await self.queue_service.delete_message(
                        queue_url=self.queue_url, receipt_handle=receipt_handle
                    )
                    logger.info(f"Message deleted from queue for chat_id: {chat_id}")
                except Exception as delete_error:
                    logger.error(
                        f"Failed to delete message from queue for chat_id {chat_id}: "
                        f"{delete_error}"
                    )
                    await phi_logger.log(
                        PHILogEvent(
                            event_type=PHIEvents.SYSTEM_ERROR,
                            chat_id=chat_id,
                            audit_id=None,  # Will be set by external service
                            details={
                                "error": f"Failed to delete message from queue: {str(delete_error)}",  # noqa: E501
                                "chat_id": chat_id,
                                "message_id": message.get("message_id", "unknown"),
                                "queue_url": self.queue_url,
                            },
                        )
                    )

    async def poll_queue(self) -> None:
        """
        Poll the queue for messages and process them.
        """
        try:
            messages = await self.queue_service.receive_messages(
                queue_url=self.queue_url,
                max_messages=self.max_messages,
                wait_time_seconds=self.wait_time_seconds,
                visibility_timeout=self.visibility_timeout,
            )

            if messages:
                # Concurrency/backlog visibility: how big was this batch, how
                # many are we allowed to run at once, and what is the queue
                # depth behind us. This is what surfaces concurrency-driven
                # backlog (the "2-3 recording at once" scenario) in the logs.
                depth = {}
                get_depth = getattr(self.queue_service, "get_queue_depth", None)
                if callable(get_depth):
                    depth = await get_depth(self.queue_url)
                logger.info(
                    "poll: received batch_size=%s max_concurrent=%s "
                    "queue_visible=%s queue_in_flight=%s queue=%s",
                    len(messages),
                    self.max_concurrent_messages,
                    depth.get("visible", "n/a"),
                    depth.get("in_flight", "n/a"),
                    self.queue_url,
                )

                # Bound concurrency: SQS may hand us up to max_messages (default 10)
                # per poll, but processing each may load an audio blob and call an
                # LLM. Without this semaphore the worker's memory spikes linearly
                # with batch size.
                async def _bounded(msg: Dict[str, Any]) -> None:
                    # Log when a message has to wait for a concurrency slot, so
                    # saturation (more concurrent recordings than slots) is
                    # visible rather than silent.
                    if self._process_semaphore.locked():
                        logger.info(
                            "All %s concurrency slots busy; message queued for a slot",
                            self.max_concurrent_messages,
                        )
                    async with self._process_semaphore:
                        await self.process_message(msg)

                await asyncio.gather(
                    *[_bounded(message) for message in messages],
                    return_exceptions=True,
                )

        except Exception as e:
            logger.exception(f"Error polling queue: {type(e).__name__}")
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.SYSTEM_ERROR,
                    chat_id=None,  # No specific chat for polling errors
                    audit_id=None,  # Will be set by external service
                    details={
                        "error": f"Error polling queue: {type(e).__name__}",
                        "exception_type": type(e).__name__,
                        "queue_url": self.queue_url,
                        "max_messages": self.max_messages,
                        "wait_time_seconds": self.wait_time_seconds,
                        "visibility_timeout": self.visibility_timeout,
                    },
                )
            )

    async def run(self) -> None:
        """
        Run the message processor continuously.
        """
        self._running = True
        logger.info(f"Starting message processor for queue: {self.queue_url}")

        while self._running:
            try:
                await self.poll_queue()

                # Add polling interval if specified
                if self.polling_interval > 0:
                    await asyncio.sleep(self.polling_interval)

            except asyncio.CancelledError:
                logger.info("Message processor cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in message processor: {type(e).__name__}")
                await phi_logger.log(
                    PHILogEvent(
                        event_type=PHIEvents.SYSTEM_ERROR,
                        chat_id=None,  # No specific chat for system errors
                        audit_id=None,  # Will be set by external service
                        details={
                            "error": f"Error in message processor: {type(e).__name__}",
                            "exception_type": type(e).__name__,
                            "queue_url": self.queue_url,
                        },
                    )
                )
                # Continue running even if there's an error
                await asyncio.sleep(1)

        logger.info(f"Message processor stopped for queue: {self.queue_url}")

    async def start(self) -> None:
        """
        Start the message processor in the background.
        """
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run())
            logger.info("Message processor started")

    async def stop(self) -> None:
        """
        Stop the message processor.
        """
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Message processor stopped")
