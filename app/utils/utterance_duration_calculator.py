from typing import List, Optional
from app.schemas.common import ChatMessage


def calculate_avg_client_utterance_duration(chat_messages: List[ChatMessage]) -> Optional[float]:
    """
    Calculate the average duration of client utterances in seconds.
    
    This function calculates the average time duration of messages sent by clients
    based on the start_time and end_time fields in ChatMessage objects.
    
    Args:
        chat_messages: List of chat messages with timing information
        
    Returns:
        Optional[float]: Average duration of client utterances in seconds rounded to 2 decimal places,
                        or None if no valid client utterances with timing data are found
    """
    client_durations = []

    for message in chat_messages:
        # Check if this is a client message and has timing information
        if (message.role.lower() == "client" and
                message.start_time is not None and
                message.end_time is not None and
                message.end_time > message.start_time):
            duration = message.end_time - message.start_time
            client_durations.append(duration)

    # Return average duration if we have valid client utterances, otherwise None
    if client_durations:
        return round(sum(client_durations) / len(client_durations),2)

    return None
