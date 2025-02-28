from typing import List

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
    return '\n'.join(f'{msg.role}: {msg.content}' for msg in chat_messages)
