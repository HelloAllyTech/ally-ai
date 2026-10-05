"""Unit tests for extract_usage_details: the cache-hit and reasoning detail
read off LangChain's normalised usage, for the calls that go through
`_invoke_llm` rather than the structured-generation dispatch."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.core.llm_usage.extract import extract_usage_details


def _cb(per_model):
    return SimpleNamespace(usage_metadata=per_model)


class TestExtractUsageDetails:
    def test_reads_cache_hits_and_reasoning_from_the_callback(self):
        cb = _cb(
            {
                "gpt-5-mini": {
                    "input_tokens": 800,
                    "output_tokens": 500,
                    "total_tokens": 1300,
                    "input_token_details": {"cache_read": 512, "audio": 0},
                    "output_token_details": {"reasoning": 320, "audio": 0},
                }
            }
        )
        assert extract_usage_details(cb) == (512, {"reasoning_tokens": 320})

    def test_sums_across_models_like_the_token_counts_do(self):
        cb = _cb(
            {
                "a": {"input_token_details": {"cache_read": 100}},
                "b": {
                    "input_token_details": {"cache_read": 50},
                    "output_token_details": {"reasoning": 7},
                },
            }
        )
        assert extract_usage_details(cb) == (150, {"reasoning_tokens": 7})

    def test_falls_back_to_the_aimessage(self):
        response = SimpleNamespace(
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_token_details": {"cache_read": 4},
            }
        )
        assert extract_usage_details(None, response) == (4, None)

    def test_an_empty_callback_defers_to_the_aimessage(self):
        response = SimpleNamespace(
            usage_metadata={"input_token_details": {"cache_read": 4}}
        )
        assert extract_usage_details(_cb({}), response) == (4, None)

    def test_reads_the_raw_openai_shape(self):
        response = SimpleNamespace(
            usage_metadata=None,
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                }
            },
        )
        assert extract_usage_details(None, response) == (3, {"reasoning_tokens": 2})

    def test_unreported_detail_is_none_not_zero(self):
        response = SimpleNamespace(
            usage_metadata={"input_tokens": 10, "output_tokens": 5}
        )
        assert extract_usage_details(None, response) == (None, None)

    def test_zero_reasoning_carries_no_metadata(self):
        cb = _cb(
            {
                "gpt-4o-mini": {
                    "input_token_details": {"cache_read": 0},
                    "output_token_details": {"reasoning": 0},
                }
            }
        )
        # A measured miss stays 0; zero reasoning is not worth a blob.
        assert extract_usage_details(cb) == (0, None)

    def test_never_invents_numbers_from_a_mock(self):
        """A MagicMock answers every attribute and coerces to 1 under int(),
        so a lax reader would report phantom cache hits in tests and on any
        response shape it does not recognise."""
        assert extract_usage_details(MagicMock(), MagicMock()) == (None, None)

    def test_nothing_at_all(self):
        assert extract_usage_details() == (None, None)
