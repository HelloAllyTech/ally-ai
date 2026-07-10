"""Unit tests for the round-trip WER/CER computation (deterministic part)."""

from app.core.round_trip.wer import error_rate_pct, normalize


class TestNormalize:
    def test_strips_punctuation_and_case(self):
        assert normalize("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize("  a\n b\t c ") == "a b c"

    def test_preserves_indic_script(self):
        assert normalize("நான் வீட்டில் இருக்கிறேன்.") == "நான் வீட்டில் இருக்கிறேன்"


class TestErrorRate:
    def test_identical_is_zero(self):
        assert error_rate_pct("hello world", "Hello, world!") == 0.0

    def test_one_substitution_of_four_words(self):
        assert error_rate_pct("a b c d", "a b x d") == 25.0

    def test_deletion_and_insertion(self):
        assert error_rate_pct("a b c", "a c") == round(1 / 3 * 100, 2)
        assert error_rate_pct("a c", "a b c") == 50.0

    def test_empty_hypothesis_is_total_error(self):
        assert error_rate_pct("a b", "") == 100.0

    def test_empty_reference(self):
        assert error_rate_pct("", "") == 0.0
        assert error_rate_pct("", "x") == 100.0

    def test_capped_at_100(self):
        assert error_rate_pct("a", "x y z w v") == 100.0

    def test_cer_counts_characters(self):
        # ref "abcd" vs hyp "abed": 1 char substitution of 4 = 25%
        assert error_rate_pct("abcd", "abed", unit="cer") == 25.0

    def test_cer_for_indic(self):
        text = "ನಾನು ಮನೆಯಲ್ಲಿದ್ದೇನೆ"
        assert error_rate_pct(text, text, unit="cer") == 0.0
