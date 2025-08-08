from typing import List, Set
import re
from rapidfuzz import fuzz, process, utils
from app.schemas.common import ChatMessage

# List of common affirmation phrases
AFFIRMATION_PHRASES = [
    "That makes complete sense.",
    "I can completely understand why you'd feel this way.",
    "It's okay to feel conflicted about this.",
    "You're not wrong to feel this way.",
    "Your feelings are absolutely valid.",
    "This must be very heavy for you.",
    "What you're saying is important.",
    "You've been very patient.",
    "That must have been very hurtful, especially coming from family.",
    "It's okay to question things, even traditions.",
    "A lot of people in our culture face this too.",
    "In many families, this kind of pressure is quite common.",
    "It's normal to feel anxious when there's so much expectation.",
    "You're not alone in going through this.",
    "Others from similar backgrounds have also experienced this.",
    "Feeling like this after what you've been through is very natural.",
    "A lot of Indian parents do have a hard time understanding mental health.",
    "This is a common issue in arranged marriage scenarios.",
    "You're not overthinking - it's a valid concern.",
    "You've handled a very difficult situation with grace.",
    "It's brave of you to speak about this - especially in our culture.",
    "You're doing the right thing for yourself, and that matters.",
    "You have a right to make your own choices.",
    "You're not being selfish for thinking about your needs.",
    "You've come a long way.",
    "You are worthy - no matter what society says.",
    "It's okay to take your time to figure things out.",
    "You're not 'too sensitive' - your feelings are genuine.",
    "What you want for yourself is important."
]

# Build regex pattern once at import time for better performance
FUZZY_THRESHOLD = 80
MIN_SENTENCE_LENGTH = 5

SENTENCE_SPLITTER = re.compile(r"[.!?]")

PROCESSED_PHRASES = [utils.default_process(phrase) for phrase in AFFIRMATION_PHRASES]


def extract_sentences(content : str) -> List[str]:
    sentences = SENTENCE_SPLITTER.split(content)
    return [s.strip() for s in sentences if len(s.strip()) > MIN_SENTENCE_LENGTH]


def count_affirmations(chat_messages: List[ChatMessage]) -> int:
    """
    Count the number of affirmations used by the counselor in the chat history.
    
    Args:
        chat_messages: List of chat messages
        
    Returns:
        int: Count of affirmations used by the counselor
    """
    if not chat_messages:
        return 0

    count = 0


    for msg in chat_messages:
        if msg.role.lower() != "counselor":
            continue

        sentences = extract_sentences(msg.content)

        for sentence in sentences:
            best_match = process.extractOne(
                sentence,
                PROCESSED_PHRASES,
                scorer=fuzz.token_set_ratio,
                processor=utils.default_process,
                score_cutoff=FUZZY_THRESHOLD,
            )

            if best_match:
                count+=1

    return count
