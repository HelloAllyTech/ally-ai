import re
from typing import List, Dict
from collections import defaultdict

# Step 1: Unicode block mapping
UNICODE_SCRIPT_RANGES = {
    "Hindi": (0x0900, 0x097F), # Devanagari is the script used for Hindi. This will actually detect Marathi and Hindi.
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Oriya": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "English": (0x0041, 0x007A),  # Basic English/Latin block 
}

def get_script_for_char(char: str) -> str:
    """
    Determine the script for a single character using Unicode ranges.
    
    Args:
        char: A single character
        
    Returns:
        The script name (e.g., 'Hindi', 'Malayalam', etc.)
    """
    code_point = ord(char)
    for script, (start, end) in UNICODE_SCRIPT_RANGES.items():
        if start <= code_point <= end:
            return script
    return "English"  # Default to English/Latin instead of Unknown

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text by removing punctuation and splitting into words.
    
    Args:
        text: Input text string
        
    Returns:
        List of preprocessed words
    """
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text.split()

def detect_script_for_word(word: str) -> str:
    """
    Detect script for a word based on majority of characters.
    
    Args:
        word: Input word
        
    Returns:
        The most common script in the word
    """
    script_counter = defaultdict(int)
    for char in word:
        script = get_script_for_char(char)
        script_counter[script] += 1
    return max(script_counter.items(), key=lambda x: x[1])[0] if script_counter else "English"

def parse_chat_messages(chat_history: str) -> List[str]:
    """
    Parse chat history into a list of message contents.
    
    Args:
        chat_history: A string containing the chat history
        
    Returns:
        List of message contents
    """
    messages = []
    for line in chat_history.split('\n'):
        if ':' in line:
            _, content = line.split(':', 1)
            messages.append(content.strip())
    return messages

def detect_languages(chat_history: str) -> List[Dict[str, float]]:
    """
    Detect languages in chat history using Unicode script ranges.
    
    Args:
        chat_history: A string containing the chat history
        
    Returns:
        List of dictionaries with language and percentage
    """

    # Parse messages from chat history
    messages = parse_chat_messages(chat_history)
    if not messages:
        return []
    
    # First loop: Collect all words from all messages
    all_words = []
    for message in messages:
        words = preprocess_text(message)
        all_words.extend(words)
    
    total_words = len(all_words)
    
    # Second loop: Count scripts for all words
    script_counter = defaultdict(int)
    for word in all_words:
        script = detect_script_for_word(word)
        script_counter[script] += 1
    
    # Convert counts to percentages
    result = []
    for script, count in script_counter.items():
        if total_words > 0:
            percentage = (count / total_words) * 100
            result.append({
                "language": script,
                "percentage": round(percentage, 1)
            })
    
    # Sort by percentage in descending order
    return sorted(result, key=lambda x: x["percentage"], reverse=True)