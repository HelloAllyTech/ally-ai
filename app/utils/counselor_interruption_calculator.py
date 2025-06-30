from typing import List
import logging

from app.schemas.common import ChatMessage

logger = logging.getLogger(__name__)


def calculate_counselor_interruptions(chat_messages: List[ChatMessage]) -> int:
    """
    Calculate the count of interruptions by the counselor.
    
    An interruption is defined as a counselor message that starts during a client's speaking time,
    i.e., when a counselor's message start_time falls within a client's message start_time to end_time range.
    
    This implementation uses a sort and linear scan approach for efficiency:
    1. Filter messages with valid timestamps
    2. Sort all messages by start_time (with client messages prioritized in case of ties)
    3. Track active client messages as we scan through the sorted list
    4. Count interruptions when a counselor message starts while a client message is active
    
    Parameters:
        chat_messages (List[ChatMessage]): List of chat messages with role, content, start_time, and end_time
        
    Returns:
        int: Count of interruptions by the counselor
    """
    # Filter messages with valid timestamps
    valid_messages = [msg for msg in chat_messages if msg.start_time is not None and msg.end_time is not None]

    if len(valid_messages) < 2:
        logger.debug("Not enough messages with valid timestamps for interruption calculation")
        return 0

    # Sort messages by start_time, with client messages coming first in case of ties
    sorted_messages = sorted(
        valid_messages,
        key=lambda msg: (msg.start_time, 0 if msg.role.lower() == "client" else 1)
    )

    active_client_messages = []  # List of currently active client messages
    interruption_count = 0

    for msg in sorted_messages:
        # First, remove any client messages that have ended before this message starts
        active_client_messages = [client_msg for client_msg in active_client_messages
                                  if client_msg.end_time > msg.start_time]

        if msg.role.lower() == "client":
            # Add this client message to active list
            active_client_messages.append(msg)
        elif msg.role.lower() == "counselor" and active_client_messages:
            # If this is a counselor message and there are active client messages,
            # check if it interrupts any of them
            for client_msg in active_client_messages:
                if client_msg.start_time < msg.start_time < client_msg.end_time:
                    interruption_count += 1
                    logger.debug(f"Detected counselor interruption at time {msg.start_time}")
                    break  # Count only once per counselor message

    logger.info(f"Total counselor interruptions: {interruption_count}")
    return interruption_count
