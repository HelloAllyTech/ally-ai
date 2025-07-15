from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, TypeVar

# Type variable for the client
ClientT = TypeVar('ClientT')


class BaseQueueService[ClientT](ABC):
    """
    Abstract base class for queue services using a client to send and receive messages.

    Parameters:
        client (ClientT): An instance of a queue client.
    """

    def __init__(self, client: ClientT) -> None:
        """
        Initialize the queue service with a client.

        Parameters:
            client (ClientT): An instance of a queue client.
        """
        self.client = client

    @abstractmethod
    async def send_message(self, queue_url: str, message: Dict[str, Any], 
                          message_attributes: Optional[Dict[str, Any]] = None,
                          delay_seconds: int = 0) -> Dict[str, Any]:
        """
        Send a message to the specified queue.

        Parameters:
            queue_url (str): The URL of the queue to send the message to.
            message (Dict[str, Any]): The message body to send.
            message_attributes (Optional[Dict[str, Any]]): Optional message attributes.
            delay_seconds (int): The number of seconds to delay the message.

        Returns:
            Dict[str, Any]: The response from the queue service.

        Raises:
            QueueOperationException: If the message sending fails.
        """
        pass

    @abstractmethod
    async def receive_messages(self, queue_url: str, max_messages: int = 1, 
                              wait_time_seconds: int = 20,
                              visibility_timeout: int = 30) -> List[Dict[str, Any]]:
        """
        Receive messages from the specified queue.

        Parameters:
            queue_url (str): The URL of the queue to receive messages from.
            max_messages (int): The maximum number of messages to receive.
            wait_time_seconds (int): The duration (in seconds) for which the call waits for a message to arrive.
            visibility_timeout (int): The duration (in seconds) that the received messages are hidden from subsequent retrieve requests.

        Returns:
            List[Dict[str, Any]]: A list of received messages.

        Raises:
            QueueOperationException: If the message receiving fails.
        """
        pass

    @abstractmethod
    async def delete_message(self, queue_url: str, receipt_handle: str) -> None:
        """
        Delete a message from the queue.

        Parameters:
            queue_url (str): The URL of the queue.
            receipt_handle (str): The receipt handle of the message to delete.

        Raises:
            QueueOperationException: If the message deletion fails.
        """
        pass

    @abstractmethod
    async def create_queue(self, queue_name: str, attributes: Optional[Dict[str, str]] = None) -> str:
        """
        Create a new queue.

        Parameters:
            queue_name (str): The name of the queue to create.
            attributes (Optional[Dict[str, str]]): Queue attributes.

        Returns:
            str: The URL of the created queue.

        Raises:
            QueueOperationException: If the queue creation fails.
        """
        pass

    @abstractmethod
    async def get_queue_url(self, queue_name: str) -> str:
        """
        Get the URL of an existing queue.

        Parameters:
            queue_name (str): The name of the queue.

        Returns:
            str: The URL of the queue.

        Raises:
            QueueOperationException: If the queue does not exist or the operation fails.
        """
        pass

    @abstractmethod
    async def process_messages(self, queue_url: str, handler: Callable[[Dict[str, Any]], Any], 
                              max_messages: int = 10, wait_time_seconds: int = 20,
                              visibility_timeout: int = 30, auto_delete: bool = True) -> None:
        """
        Process messages from the queue using the provided handler function.

        Parameters:
            queue_url (str): The URL of the queue to process messages from.
            handler (Callable[[Dict[str, Any]], Any]): The function to process each message.
            max_messages (int): The maximum number of messages to receive in each batch.
            wait_time_seconds (int): The duration (in seconds) for which the call waits for a message to arrive.
            visibility_timeout (int): The duration (in seconds) that the received messages are hidden from subsequent retrieve requests.
            auto_delete (bool): Whether to automatically delete messages after successful processing.

        Raises:
            QueueOperationException: If the message processing fails.
        """
        pass
