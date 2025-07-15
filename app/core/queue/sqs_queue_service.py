import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable, Awaitable

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from app.core.queue.base import BaseQueueService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SQSQueueService(BaseQueueService[BaseClient]):
    """
    Implementation of the BaseQueueService using AWS SQS.
    Uses boto3 with asyncio for asynchronous operations.
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
        super().__init__(client)
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
            message_attribute_names (Optional[List[str]]): A list of message attribute names to receive.

        Returns:
            List[Dict[str, Any]]: A list of messages.

        Raises:
            Exception: If there is an error receiving messages.
        """
        try:
            # Prepare the request parameters
            kwargs = {
                'QueueUrl': queue_url,
                'MaxNumberOfMessages': max_messages,
                'WaitTimeSeconds': wait_time_seconds,
                'VisibilityTimeout': visibility_timeout,
                'AttributeNames': ['All']
            }
            
            if message_attribute_names:
                kwargs['MessageAttributeNames'] = message_attribute_names
            else:
                kwargs['MessageAttributeNames'] = ['All']

            # Receive messages
            response = await self._run_in_executor(
                self.client.receive_message,
                **kwargs
            )
            
            messages = response.get('Messages', [])
            
            # Process the messages
            processed_messages = []
            for message in messages:
                try:
                    # Parse the message body as JSON
                    body = json.loads(message.get('Body', '{}'))
                    
                    # Add the message metadata
                    processed_message = {
                        'message_id': message.get('MessageId'),
                        'receipt_handle': message.get('ReceiptHandle'),
                        'body': body,
                        'attributes': message.get('Attributes', {}),
                        'message_attributes': message.get('MessageAttributes', {})
                    }
                    
                    processed_messages.append(processed_message)
                    
                except json.JSONDecodeError:
                    # If the message body is not valid JSON, include it as a string
                    processed_message = {
                        'message_id': message.get('MessageId'),
                        'receipt_handle': message.get('ReceiptHandle'),
                        'body': message.get('Body', ''),
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

    async def create_queue(self, queue_name: str, attributes: Optional[Dict[str, str]] = None) -> str:
        """
        Check if an SQS queue exists. The queue is expected to be created by DevOps.

        Parameters:
            queue_name (str): The name of the queue to check.
            attributes (Optional[Dict[str, str]]): Not used, kept for backward compatibility.

        Returns:
            str: The URL of the existing queue.

        Raises:
            ValueError: If the queue does not exist.
        """
        try:
            # Get the queue URL to check if it exists
            queue_url = await self.get_queue_url(queue_name)
            logger.info(f"Queue {queue_name} exists with URL {queue_url}")
            return queue_url
            
        except ValueError as e:
            # Queue doesn't exist
            logger.error(f"Queue {queue_name} does not exist.")
            raise ValueError(f"Queue {queue_name} does not exist.") from e
        
        except Exception as e:
            logger.exception(f"Error checking queue {queue_name}: {str(e)}")
            raise

    async def get_queue_url(self, queue_name: str) -> str:
        """
        Get the URL of an SQS queue.

        Parameters:
            queue_name (str): The name of the queue.

        Returns:
            str: The URL of the queue.

        Raises:
            Exception: If there is an error getting the queue URL.
        """
        try:
            response = await self._run_in_executor(
                self.client.get_queue_url,
                QueueName=queue_name
            )
            
            queue_url = response.get('QueueUrl')
            logger.info(f"Got URL for queue {queue_name}: {queue_url}")
            return queue_url
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'AWS.SimpleQueueService.NonExistentQueue':
                logger.warning(f"Queue {queue_name} does not exist")
                raise ValueError(f"Queue {queue_name} does not exist")
            else:
                logger.exception(f"Error getting URL for queue {queue_name}: {str(e)}")
                raise
        except Exception as e:
            logger.exception(f"Error getting URL for queue {queue_name}: {str(e)}")
            raise

    async def process_messages(
        self,
        queue_url: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]],
        max_messages: int = 10,
        wait_time_seconds: int = 20,
        visibility_timeout: int = 30,
        auto_delete: bool = True
    ) -> int:
        """
        Process messages from an SQS queue using a handler function.

        Parameters:
            queue_url (str): The URL of the queue to process messages from.
            handler (Callable[[Dict[str, Any]], Awaitable[None]]): The handler function to process messages.
            max_messages (int): The maximum number of messages to receive.
            wait_time_seconds (int): The duration (in seconds) for which the call waits for a message to arrive.
            visibility_timeout (int): The duration (in seconds) that the received messages are hidden from subsequent retrieve requests.
            auto_delete (bool): Whether to automatically delete messages after successful processing.

        Returns:
            int: The number of messages processed.

        Raises:
            Exception: If there is an error processing messages.
        """
        try:
            # Receive messages
            messages = await self.receive_messages(
                queue_url=queue_url,
                max_messages=max_messages,
                wait_time_seconds=wait_time_seconds,
                visibility_timeout=visibility_timeout
            )
            
            if not messages:
                return 0
                
            # Process each message
            for message in messages:
                try:
                    # Process the message
                    await handler(message)
                    
                    # Delete the message if auto_delete is True
                    if auto_delete:
                        await self.delete_message(
                            queue_url=queue_url,
                            receipt_handle=message['receipt_handle']
                        )
                        
                except Exception as e:
                    logger.exception(f"Error processing message {message.get('message_id', 'unknown')}: {str(e)}")
                    # Don't delete the message so it can be retried after visibility timeout
            
            return len(messages)
            
        except Exception as e:
            logger.exception(f"Error processing messages from queue {queue_url}: {str(e)}")
            raise
