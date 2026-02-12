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


def filter_valid_ids(ids: List[str], valid_ids: Set[str]) -> List[str]:
    """
    Filter a list of IDs, keeping only those present in the valid set.

    Useful for post-processing LLM output to prevent hallucinated IDs.

    Parameters:
        ids: List of IDs to validate.
        valid_ids: Set of allowed IDs.

    Returns:
        List of IDs that exist in the valid set.
    """
    return [id_ for id_ in ids if id_ in valid_ids]


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
