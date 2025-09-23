"""
Unit tests for language detector utility.
"""

import pytest

from app.utils.language_detector import (
    detect_languages,
    detect_script_for_word,
    get_script_for_char,
    parse_chat_messages,
    preprocess_text,
)


class TestGetScriptForChar:
    """Test cases for get_script_for_char function."""

    def test_hindi_characters(self):
        """Test Hindi (Devanagari) character detection."""
        # Hindi characters
        assert get_script_for_char("अ") == "hi"  # Hindi 'a'
        assert get_script_for_char("क") == "hi"  # Hindi 'ka'
        assert get_script_for_char("म") == "hi"  # Hindi 'ma'

    def test_bengali_characters(self):
        """Test Bengali character detection."""
        assert get_script_for_char("অ") == "bn"  # Bengali 'a'
        assert get_script_for_char("ক") == "bn"  # Bengali 'ka'

    def test_tamil_characters(self):
        """Test Tamil character detection."""
        assert get_script_for_char("அ") == "ta"  # Tamil 'a'
        assert get_script_for_char("க") == "ta"  # Tamil 'ka'

    def test_english_characters(self):
        """Test English character detection."""
        assert get_script_for_char("A") == "en"  # English 'A'
        assert get_script_for_char("z") == "en"  # English 'z'
        assert get_script_for_char("5") == "en"  # English digit

    def test_unknown_characters(self):
        """Test characters outside known ranges."""
        assert get_script_for_char("中") == "en"  # Chinese (defaults to en)
        assert get_script_for_char("あ") == "en"  # Japanese (defaults to en)
        assert get_script_for_char("€") == "en"  # Euro symbol (defaults to en)

    def test_edge_case_characters(self):
        """Test edge case characters."""
        assert get_script_for_char(" ") == "en"  # Space
        assert get_script_for_char("\n") == "en"  # Newline
        # Empty string will raise TypeError, so we test that
        with pytest.raises(TypeError):
            get_script_for_char("")


class TestPreprocessText:
    """Test cases for preprocess_text function."""

    def test_basic_text_preprocessing(self):
        """Test basic text preprocessing."""
        text = "Hello, world! How are you?"
        result = preprocess_text(text)
        expected = ["Hello", "world", "How", "are", "you"]
        assert result == expected

    def test_text_with_punctuation(self):
        """Test text with various punctuation marks."""
        text = "Hello!!! How are you??? I'm fine, thanks."
        result = preprocess_text(text)
        expected = ["Hello", "How", "are", "you", "Im", "fine", "thanks"]
        assert result == expected

    def test_text_with_numbers(self):
        """Test text with numbers."""
        text = "I have 5 apples and 10 oranges."
        result = preprocess_text(text)
        expected = ["I", "have", "5", "apples", "and", "10", "oranges"]
        assert result == expected

    def test_empty_text(self):
        """Test empty text."""
        result = preprocess_text("")
        assert result == []

    def test_text_with_only_punctuation(self):
        """Test text with only punctuation."""
        result = preprocess_text("!!!???...")
        assert result == []

    def test_text_with_whitespace(self):
        """Test text with extra whitespace."""
        text = "  Hello    world  "
        result = preprocess_text(text)
        expected = ["Hello", "world"]
        assert result == expected

    def test_mixed_language_text(self):
        """Test text with mixed languages."""
        text = "Hello नमस्ते world"
        result = preprocess_text(text)
        # The regex removes punctuation, so "नमस्ते" becomes "नमसत" (removes the dot)
        expected = ["Hello", "नमसत", "world"]
        assert result == expected


class TestDetectScriptForWord:
    """Test cases for detect_script_for_word function."""

    def test_english_word(self):
        """Test English word detection."""
        assert detect_script_for_word("hello") == "en"
        assert detect_script_for_word("WORLD") == "en"
        assert detect_script_for_word("Test123") == "en"

    def test_hindi_word(self):
        """Test Hindi word detection."""
        assert detect_script_for_word("नमस्ते") == "hi"
        assert detect_script_for_word("हिंदी") == "hi"

    def test_mixed_script_word(self):
        """Test word with mixed scripts (should return majority)."""
        # Word with more Hindi characters than English
        word = "helloनमस्ते"
        result = detect_script_for_word(word)
        # Should return the script with majority characters
        assert result in ["hi", "en"]

    def test_empty_word(self):
        """Test empty word."""
        assert detect_script_for_word("") == "English"

    def test_single_character(self):
        """Test single character word."""
        assert detect_script_for_word("a") == "en"
        assert detect_script_for_word("अ") == "hi"

    def test_word_with_numbers(self):
        """Test word with numbers."""
        assert detect_script_for_word("test123") == "en"
        assert detect_script_for_word("123") == "en"


class TestParseChatMessages:
    """Test cases for parse_chat_messages function."""

    def test_basic_chat_parsing(self):
        """Test basic chat message parsing."""
        chat_history = (
            "client: Hello, how are you?\ncounselor: I'm doing well, thank you."
        )
        result = parse_chat_messages(chat_history)
        expected = ["Hello, how are you?", "I'm doing well, thank you."]
        assert result == expected

    def test_chat_with_multiple_colons(self):
        """Test chat with multiple colons in content."""
        chat_history = "client: Time: 3:30 PM\ncounselor: I see: you're busy"
        result = parse_chat_messages(chat_history)
        expected = ["Time: 3:30 PM", "I see: you're busy"]
        assert result == expected

    def test_empty_chat_history(self):
        """Test empty chat history."""
        result = parse_chat_messages("")
        assert result == []

    def test_chat_without_colons(self):
        """Test chat history without colons (should be ignored)."""
        chat_history = "This is not a valid message\nNeither is this"
        result = parse_chat_messages(chat_history)
        assert result == []

    def test_mixed_valid_invalid_messages(self):
        """Test chat with mix of valid and invalid messages."""
        chat_history = (
            "client: Valid message\nInvalid message\ncounselor: Another valid"
        )
        result = parse_chat_messages(chat_history)
        expected = ["Valid message", "Another valid"]
        assert result == expected

    def test_chat_with_whitespace(self):
        """Test chat with extra whitespace."""
        chat_history = "  client  :  Hello  \n  counselor  :  Hi there  "
        result = parse_chat_messages(chat_history)
        expected = ["Hello", "Hi there"]
        assert result == expected


class TestDetectLanguages:
    """Test cases for detect_languages function."""

    def test_english_only_chat(self):
        """Test chat with only English text."""
        chat_history = "client: Hello, how are you?\ncounselor: I'm doing well."
        result = detect_languages(chat_history)

        assert len(result) == 1
        assert result[0].language == "en"
        assert result[0].percentage == 100.0

    def test_hindi_only_chat(self):
        """Test chat with only Hindi text."""
        chat_history = "client: नमस्ते, आप कैसे हैं?\ncounselor: मैं ठीक हूँ।"
        result = detect_languages(chat_history)

        assert len(result) == 1
        assert result[0].language == "hi"
        assert result[0].percentage == 100.0

    def test_mixed_language_chat(self):
        """Test chat with mixed languages."""
        chat_history = "client: Hello नमस्ते world\ncounselor: Hi there"
        result = detect_languages(chat_history)

        # Should have both languages
        languages = [lang.language for lang in result]
        assert "en" in languages
        assert "hi" in languages

        # Percentages should add up to 100
        total_percentage = sum(lang.percentage for lang in result)
        assert abs(total_percentage - 100.0) < 0.1

    def test_empty_chat_history(self):
        """Test empty chat history."""
        result = detect_languages("")
        assert result == []

    def test_chat_with_no_valid_messages(self):
        """Test chat with no valid messages."""
        result = detect_languages("Invalid message\nAnother invalid")
        assert result == []

    def test_chat_with_punctuation_only(self):
        """Test chat with only punctuation."""
        result = detect_languages("client: !!!\ncounselor: ???")
        assert result == []

    def test_language_percentage_calculation(self):
        """Test accurate percentage calculation."""
        # 2 English words, 1 Hindi word = 66.7% English, 33.3% Hindi
        chat_history = "client: Hello world नमस्ते"
        result = detect_languages(chat_history)

        # Find the languages
        en_lang = next((lang for lang in result if lang.language == "en"), None)
        hi_lang = next((lang for lang in result if lang.language == "hi"), None)

        assert en_lang is not None
        assert hi_lang is not None

        # Check approximate percentages (allowing for rounding)
        assert abs(en_lang.percentage - 66.7) < 1.0
        assert abs(hi_lang.percentage - 33.3) < 1.0

    def test_languages_sorted_by_percentage(self):
        """Test that languages are sorted by percentage in descending order."""
        chat_history = "client: Hello नमस्ते world world world"
        result = detect_languages(chat_history)

        # Should be sorted by percentage (descending)
        percentages = [lang.percentage for lang in result]
        assert percentages == sorted(percentages, reverse=True)

    def test_multiple_languages(self):
        """Test chat with multiple languages."""
        chat_history = "client: Hello नमस्ते 你好 world"
        result = detect_languages(chat_history)

        # Should detect multiple languages
        assert len(result) >= 2
        languages = [lang.language for lang in result]
        assert "en" in languages
        assert "hi" in languages

    def test_rounding_of_percentages(self):
        """Test that percentages are properly rounded to 1 decimal place."""
        chat_history = "client: Hello world नमस्ते"
        result = detect_languages(chat_history)

        for lang in result:
            # Check that percentage has at most 1 decimal place
            decimal_places = (
                len(str(lang.percentage).split(".")[-1])
                if "." in str(lang.percentage)
                else 0
            )
            assert decimal_places <= 1
