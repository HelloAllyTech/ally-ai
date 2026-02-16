from typing import Any, Dict, List, Set, Union

from app.schemas.common import ChatMessage


def convert_chat_messages_to_string(chat_messages: List[ChatMessage]) -> str:
    """
    Convert a list of ChatMessage objects to a formatted string.

    Each chat message is converted to a string in the format "role: content",
    and individual messages are separated by newlines.

    Parameters:
        chat_messages (List[ChatMessage]): A list of ChatMessage objects,
            where each object is expected to have 'role' and 'content' attributes.

    Returns:
        str: A single string with each message formatted as "role: content"
             on its own line.
    """
    return "\n".join(f"{msg.role}: {msg.content}" for msg in chat_messages)



def filter_message_tags(
    message_tags: List[Any],
    valid_message_ids: Set[str],
) -> List[Dict[str, Any]]:
    """
    Filter and serialise message tags, keeping only entries whose ID
    is in the valid set.

    Useful for post-processing LLM output to strip tags for messages
    that shouldn't have been tagged (e.g. client messages).

    Parameters:
        message_tags: List of MessageTagItemOutput objects from LLM response.
        valid_message_ids: Set of message IDs that are allowed to have tags.

    Returns:
        List of dicts with "id" and "tags" ready for the API response.
    """
    return [
        {
            "id": item.id,
            "tags": [
                {"label": t.label, "category": t.category.value}
                for t in item.tags
            ],
        }
        for item in message_tags
        if item.id in valid_message_ids
    ]


def filter_emotional_movement(
    emotional_movement: List[Any],
    client_message_ids: List[str],
    default_level: int = 0,
) -> List[Dict[str, Any]]:
    """
    Filter, serialise, and back-fill emotional movement so every client
    message is guaranteed to have a rating in conversation order.

    - Strips entries for non-client messages (e.g. counselor).
    - Back-fills any missing client messages with *default_level* (neutral).

    Parameters:
        emotional_movement: List of EmotionalMovementItemOutput objects from LLM response.
        client_message_ids: Ordered list of client message IDs (conversation order).
        default_level: Level to assign to missing messages (default 0 = neutral).

    Returns:
        List of dicts with "message_id" and "level" for every client message,
        in conversation order.
    """
    valid_ids = set(client_message_ids)

    # Build lookup from LLM response, keeping only valid client messages
    rated = {
        item.message_id: item.level
        for item in emotional_movement
        if item.message_id in valid_ids
    }

    # Ensure every client message has a rating, preserving conversation order
    return [
        {"message_id": msg_id, "level": rated.get(msg_id, default_level)}
        for msg_id in client_message_ids
    ]
