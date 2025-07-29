import asyncio
from typing import Dict, Any

from app.core.queue.base import BaseQueueService
from app.core.queue.message_router import MessageRouter
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MessageProcessor:
    """
    Simple message processor for continuously polling a queue for messages and processing them.
    """

    def __init__(
            self,
            queue_service: BaseQueueService,
            router: MessageRouter,
            queue_url: str,
            max_messages: int = 10,
            wait_time_seconds: int = 20,
            visibility_timeout: int = 30,
            polling_interval: int = 0,
            auto_delete: bool = True
    ):
        """
        Initialize the message processor.

        Parameters:
            queue_service (BaseQueueService): The queue service to use.
            router (MessageRouter): The message router to use for routing messages.
            queue_url (str): The URL of the queue to poll.
            max_messages (int): The maximum number of messages to receive in each batch.
            wait_time_seconds (int): The duration (in seconds) for which the call waits for a message to arrive.
            visibility_timeout (int): The duration (in seconds) that the received messages are hidden from subsequent retrieve requests.
            polling_interval (int): The interval (in seconds) between polling attempts. 0 means continuous polling.
            auto_delete (bool): Whether to automatically delete messages after successful processing.
        """
        self.queue_service = queue_service
        self.router = router
        self.queue_url = queue_url
        self.max_messages = max_messages
        self.wait_time_seconds = wait_time_seconds
        self.visibility_timeout = visibility_timeout
        self.polling_interval = polling_interval
        self.auto_delete = auto_delete
        self._running = False
        self._task = None

    async def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a single message.

        Parameters:
            message (Dict[str, Any]): The message to process.
        """
        try:
            # Route the message to the appropriate handlers
            await self.router.route_message(message)

            # Delete the message if auto_delete is True
            if self.auto_delete:
                await self.queue_service.delete_message(
                    queue_url=self.queue_url,
                    receipt_handle=message['receipt_handle']
                )

        except Exception as e:
            chat_id = message.get('body', {}).get('chat_id', 'unknown')
            logger.exception(f"Error processing message for chat_id {chat_id}: {str(e)}")

    async def poll_queue(self) -> None:
        """
        Poll the queue for messages and process them.
        """
        try:
            messages = await self.queue_service.receive_messages(
                queue_url=self.queue_url,
                max_messages=self.max_messages,
                wait_time_seconds=self.wait_time_seconds,
                visibility_timeout=self.visibility_timeout
            )

            if messages:
                logger.info(f"Received {len(messages)} messages from queue {self.queue_url}")

                # Process each message
                await asyncio.gather(
                    *[self.process_message(message) for message in messages],
                    return_exceptions=True
                )

        except Exception as e:
            logger.exception(f"Error polling queue {self.queue_url}: {str(e)}")

    async def run(self) -> None:
        """
        Run the message processor continuously.
        """
        self._running = True
        logger.info(f"Starting message processor for queue {self.queue_url}")

        try:
            while self._running:
                await self.poll_queue()

                # Sleep if a polling interval is specified
                if self.polling_interval > 0:
                    await asyncio.sleep(self.polling_interval)

        except KeyboardInterrupt:
            logger.info("Message processor interrupted by user")
        except asyncio.CancelledError:
            logger.info("Message processor task cancelled")
        finally:
            self._running = False
            logger.info(f"Message processor for queue {self.queue_url} stopped")

    async def start(self) -> None:
        """
        Start the message processor in a background task.
        """
        if self._task is not None and not self._task.done():
            logger.warning("Message processor is already running")
            return

        self._task = asyncio.create_task(self.run())
        logger.info(f"Message processor for queue {self.queue_url} started in background task")

    async def stop(self) -> None:
        """
        Stop the message processor.
        """
        if not self._running:
            logger.info("Message processor is not running")
            return

        logger.info(f"Stopping message processor for queue {self.queue_url}")
        self._running = False

        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info(f"Message processor for queue {self.queue_url} stopped")