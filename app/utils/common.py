import re
from typing import Any, Dict, List, Set, Tuple

from app.core.text_generations.structured_output_models import (
    MessageTagLabelEnum,
    get_message_tag_category,
)
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


def build_id_mapping(
    chat_messages: List[ChatMessage],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build bidirectional mappings between original message UUIDs
    and compact sequential keys (m1, m2, ...) to reduce token usage in LLM
    prompts and responses.

    Parameters:
        chat_messages: List of ChatMessage objects with original IDs.

    Returns:
        A tuple of two dicts:
        - uuid_to_key: Maps original UUID → key  (e.g. "abc-123..." → "m1")
        - key_to_uuid: Maps key → original UUID  (e.g. "m1" → "abc-123...")
    """
    uuid_to_key: Dict[str, str] = {}
    key_to_uuid: Dict[str, str] = {}
    for idx, msg in enumerate(chat_messages, start=1):
        if msg.id and msg.id not in uuid_to_key:
            key = f"m{idx}"
            uuid_to_key[msg.id] = key
            key_to_uuid[key] = msg.id
    return uuid_to_key, key_to_uuid


def filter_message_tags(
    message_tags: List[Any],
    valid_keys: Set[str],
    key_to_uuid: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Filter and serialise message tags, keeping only entries whose key
    is in the valid set.

    Useful for post-processing LLM output to strip tags for messages
    that shouldn't have been tagged (e.g. client messages).

    Also validates that tag labels are in the allowed enum set, filtering out
    any hallucinated or invalid labels.

    Parameters:
        message_tags: List of MessageTagItemOutput objects from LLM response.
        valid_keys: Set of message keys that are allowed to have tags.
        key_to_uuid: Mapping of keys back to original UUIDs.

    Returns:
        List of dicts with "id" (original UUID) and "tags" ready for the API response.
    """
    valid_labels = set(MessageTagLabelEnum)

    return [
        {
            "id": key_to_uuid[item.id],
            "tags": [
                {
                    "label": t.label.value,
                    "category": get_message_tag_category(t.label).value,
                }
                for t in item.tags
                if t.label in valid_labels
            ],
        }
        for item in message_tags
        if item.id in valid_keys
    ]


def filter_emotional_movement(
    emotional_movement: List[Any],
    client_keys: List[str],
    key_to_uuid: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Filter, serialise, and back-fill emotional movement so every client
    message is guaranteed to have a rating in conversation order.

    - Strips entries for non-client messages (e.g. counselor).
    - Back-fills any missing client messages with 0 (neutral).

    Parameters:
        emotional_movement: List of EmotionalMovementItemOutput objects
            from LLM response.
        client_keys: Ordered list of client message keys (conversation order).
        key_to_uuid: Mapping of keys back to original UUIDs.

    Returns:
        List of dicts with "message_id" (original UUID) and "level" for every
        client message, in conversation order.
    """
    valid_keys = set(client_keys)

    # Build lookup from LLM response, keeping only valid client messages
    rated = {
        item.message_id: item.level
        for item in emotional_movement
        if item.message_id in valid_keys
    }

    # Ensure every client message has a rating, preserving conversation order
    return [
        {
            "message_id": key_to_uuid[msg_key],
            "level": rated.get(msg_key, 0),
        }
        for msg_key in client_keys
    ]


# Matches the [[msg:KEY]] anchors the supervisor note uses to point at a
# transcript moment. Tolerant of stray whitespace and casing, because the
# model writes these by hand inside prose.
_NOTE_REFERENCE_PATTERN = re.compile(r"\[\[\s*msg\s*:\s*([^\]]*?)\s*\]\]", re.I)


def remap_note_references(note: str, key_to_uuid: Dict[str, str]) -> str:
    """
    Rewrite [[msg:KEY]] anchors in a supervisor note to original message UUIDs.

    The LLM sees compact keys (m1, m2, ...) to save tokens, so anchors come back
    referring to those. Clients need the real message UUID to link an anchor to
    a transcript message.

    Anchors naming a key that isn't in the transcript are dropped rather than
    kept, so a hallucinated reference degrades to plain prose instead of a link
    that goes nowhere. The prompt requires each sentence to read correctly
    without its anchor, which is what makes dropping safe.

    Parameters:
        note: The supervisor note markdown as returned by the LLM.
        key_to_uuid: Mapping of compact keys back to original UUIDs.

    Returns:
        The note with valid anchors remapped to UUIDs and invalid ones removed.
    """
    if not note:
        return note

    def _replace(match: re.Match) -> str:
        uuid = key_to_uuid.get(match.group(1).strip())
        return f"[[msg:{uuid}]]" if uuid else ""

    remapped = _NOTE_REFERENCE_PATTERN.sub(_replace, note)

    # Dropping an anchor can leave a doubled space or a space before
    # punctuation; tidy those so the prose still reads cleanly.
    remapped = re.sub(r"[ \t]{2,}", " ", remapped)
    remapped = re.sub(r"[ \t]+([,.;:!?])", r"\1", remapped)
    return remapped.strip()
