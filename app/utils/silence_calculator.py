from typing import List
import logging

from app.schemas.common import ChatMessage

logger = logging.getLogger(__name__)


def calculate_silence_by_counselor(chat_messages: List[ChatMessage]) -> int:
    """
    Calculate the count of silence moments by counselor.
    
    Silence is defined as a time frame during which neither the counselor nor client has spoken.
    A silence moment is only counted if:
    1. The duration is 3 seconds or more
    2. The silence occurs after a client message and before a counselor message OR
       between two client messages
    
    Parameters:
        chat_messages (List[ChatMessage]): List of chat messages with role, content, start_time, and end_time
        
    Returns:
        int: Count of silence moments by counselor
    """

    # Sort messages by start_time to ensure chronological order
    sorted_messages = sorted(
        [msg for msg in chat_messages if msg.start_time is not None and msg.end_time is not None],
        key=lambda x: x.start_time
    )

    if len(sorted_messages) < 2:
        logger.debug("Not enough messages with valid timestamps for silence calculation")
        return 0

    silence_count = 0

    # Iterate through messages to find silence periods
    for i in range(len(sorted_messages) - 1):
        current_msg = sorted_messages[i]
        next_msg = sorted_messages[i + 1]

        # Skip if either message doesn't have valid timestamps
        if (current_msg.start_time is None or current_msg.end_time is None or
                next_msg.start_time is None or next_msg.end_time is None):
            continue

        # Calculate silence duration between messages
        silence_duration = next_msg.start_time - current_msg.end_time

        # Only count silence if it's 3 seconds or more
        if silence_duration < 3:
            continue

        # Check if silence is after client message and before counselor message
        if current_msg.role == "client" and next_msg.role == "counselor":
            silence_count += 1
            logger.debug(f"Adding silence moment (client → counselor)")

        # Check if silence is between two client messages
        elif current_msg.role == "client" and next_msg.role == "client":
            silence_count += 1
            logger.debug(f"Adding silence moment (client → client)")

    logger.info(f"Total silence moments by counselor: {silence_count}")
    return silence_count
