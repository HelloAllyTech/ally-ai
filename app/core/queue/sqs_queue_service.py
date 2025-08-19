import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import boto3
from botocore.client import BaseClient

from app.utils.logger import get_logger

logger = get_logger(__name__)


class SQSQueueService:
    """
    Simplified SQS queue service for sending and receiving messages.
    Queues are expected to be created manually by DevOps.
    """

    def __init__(
        self,
        client: BaseClient,
        max_workers: int = 10
    ):
        """
        Initialize the SQS queue service.

        Parameters:
            client (BaseClient): The SQS client.
            max_workers (int): Maximum number of worker threads for async operations.
        """
        self.client = client
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def _run_in_executor(self, func, *args, **kwargs):
        """
        Run a function in the executor.

        Parameters:
            func: The function to run.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs)
        )

    async def send_message(
        self,
        queue_url: str,
        message_body: str,
        delay_seconds: int = 0,
        message_attributes: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a message to an SQS queue.

        Parameters:
            queue_url (str): The URL of the queue to send the message to.
            message_body (str): The message to send.
            delay_seconds (int): The time in seconds to delay the message.
            message_attributes (Optional[Dict[str, Any]]): Message attributes.

        Returns:
            str: The ID of the message.

        Raises:
            Exception: If there is an error sending the message.
        """
        try:
            # Prepare the message attributes if provided
            formatted_attributes = {}
            if message_attributes:
                for key, value in message_attributes.items():
                    if isinstance(value, str):
                        formatted_attributes[key] = {
                            'DataType': 'String',
                            'StringValue': value
                        }
                    elif isinstance(value, (int, float)):
                        formatted_attributes[key] = {
                            'DataType': 'Number',
                            'StringValue': str(value)
                        }
                    elif isinstance(value, bytes):
                        formatted_attributes[key] = {
                            'DataType': 'Binary',
                            'BinaryValue': value
                        }

            # Send the message
            kwargs = {
                'QueueUrl': queue_url,
                'MessageBody': message_body,
                'DelaySeconds': delay_seconds
            }
            
            if formatted_attributes:
                kwargs['MessageAttributes'] = formatted_attributes

            response = await self._run_in_executor(
                self.client.send_message,
                **kwargs
            )
            
            message_id = response.get('MessageId')
            logger.info(f"Message sent to queue {queue_url} with ID {message_id}")
            return message_id
            
        except Exception as e:
            logger.exception(f"Error sending message to queue {queue_url}: {str(e)}")
            raise

    async def receive_messages(
        self,
        queue_url: str,
        max_messages: int = 10,
        wait_time_seconds: int = 20,
        visibility_timeout: int = 30,
        message_attribute_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Receive messages from an SQS queue.

        Parameters:
            queue_url (str): The URL of the queue to receive messages from.
            max_messages (int): The maximum number of messages to receive.
            wait_time_seconds (int): The duration (in seconds) for which the call waits for a message to arrive.
            visibility_timeout (int): The duration (in seconds) that the received messages are hidden from subsequent retrieve requests.
            message_attribute_names (Optional[List[str]]): The names of the message attributes to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of received messages.

        Raises:
            Exception: If there is an error receiving messages.
        """
        try:
            kwargs = {
                'QueueUrl': queue_url,
                'MaxNumberOfMessages': max_messages,
                'WaitTimeSeconds': wait_time_seconds,
                'VisibilityTimeout': visibility_timeout
            }
            
            if message_attribute_names:
                kwargs['MessageAttributeNames'] = message_attribute_names

            response = await self._run_in_executor(
                self.client.receive_message,
                **kwargs
            )
            
            messages = response.get('Messages', [])
            processed_messages = []
            
            for message in messages:
                processed_message = {
                    'message_id': message.get('MessageId'),
                    'receipt_handle': message.get('ReceiptHandle'),
                    'body': message.get('Body'),
                    'attributes': message.get('Attributes', {}),
                    'message_attributes': message.get('MessageAttributes', {})
                }
                processed_messages.append(processed_message)
            
            if messages:
                logger.info(f"Received {len(messages)} messages from queue {queue_url}")
            
            return processed_messages
            
        except Exception as e:
            logger.exception(f"Error receiving messages from queue {queue_url}: {str(e)}")
            raise

    async def delete_message(self, queue_url: str, receipt_handle: str) -> None:
        """
        Delete a message from an SQS queue.

        Parameters:
            queue_url (str): The URL of the queue to delete the message from.
            receipt_handle (str): The receipt handle of the message to delete.

        Raises:
            Exception: If there is an error deleting the message.
        """
        try:
            await self._run_in_executor(
                self.client.delete_message,
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle
            )
            
            logger.info(f"Message with receipt handle {receipt_handle} deleted from queue {queue_url}")
            
        except Exception as e:
            logger.exception(f"Error deleting message from queue {queue_url}: {str(e)}")
            raise


