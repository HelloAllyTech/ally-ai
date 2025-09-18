from typing import List, Optional

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.schemas.common import ChatMessage
from app.utils.logger import get_logger

# Constants

# Total range of VADER sentiment scores (from -1 to 1), used for
# normalizing percentage changes
VADER_RANGE = 2.0
SEGMENTS_COUNT = 10

logger = get_logger(__name__)


def calculate_client_positivity_lift(
    chat_messages: List[ChatMessage],
) -> Optional[float]:
    """
    Calculate the client positivity lift as a percentage change in sentiment scores.

    This function:
    1. Filters for client messages only
    2. Divides client messages into 10 equal segments
    3. Calculates VADER sentiment score for each segment
    4. Computes percentage change between consecutive segments
    5. Returns the average of these percentage changes

    Formula:
    1. For each segment pair (i-1, i):
       change = (curr_score - prev_score) / VADER_RANGE * 100
       where VADER_RANGE = 2.0 (the full range of VADER scores from -1 to 1)

    2. client_positivity_lift = average of all percentage changes

    Args:
        chat_messages: List of chat messages

    Returns:
        Optional[float]: Average percentage lift in client positivity,
                        or None if fewer than 10 client messages are found
    """
    # Filter for client messages only
    client_messages = [msg for msg in chat_messages if msg.role.lower() == "client"]

    # If fewer than 10 client messages, return None
    if len(client_messages) < 10:
        logger.debug(
            "Fewer than 10 client messages found, cannot calculate positivity lift"
        )
        return None

    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Split the client messages into SEGMENTS_COUNT nearly equal parts
    total_messages = len(client_messages)
    base_segment_size = total_messages // SEGMENTS_COUNT
    extra_messages = (
        total_messages % SEGMENTS_COUNT
    )  # Used to distribute leftover messages

    segments = []

    start_idx = 0
    for i in range(SEGMENTS_COUNT):

        # Distribute the remainder: first 'extra_messages' segments get
        # one extra item to fit total messages
        extra = 1 if i < extra_messages else 0
        end_index = start_idx + base_segment_size + extra

        # Slice the current segment from client_messages
        segment = client_messages[start_idx:end_index]
        segments.append(segment)

        # Move the start index forward for the next segment
        start_idx = end_index

    # Calculate sentiment score for each segment
    segment_scores = []
    for segment in segments:
        segment_text = " ".join([msg.content for msg in segment])
        sentiment_score = analyzer.polarity_scores(segment_text)["compound"]
        sentiment_score = round(sentiment_score, 4)
        segment_scores.append(sentiment_score)

    # Calculate percentage change between consecutive segments
    percentage_changes = []
    for i in range(1, len(segment_scores)):
        prev_score = segment_scores[i - 1]
        curr_score = segment_scores[i]

        # Calculate percentage change using VADER_RANGE as denominator
        # This normalizes the change relative to the full range of possible scores
        change = (curr_score - prev_score) / VADER_RANGE * 100
        percentage_changes.append(change)

    # Return average of percentage changes if any, otherwise None
    if percentage_changes:
        # Filter out any NaN or infinite values that might have occurred
        valid_changes = [change for change in percentage_changes if np.isfinite(change)]
        if valid_changes:
            return round(sum(valid_changes) / len(valid_changes), 2)

    logger.warning("Could not calculate valid positivity lift values")
    return None
