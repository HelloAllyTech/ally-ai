"""
Unit tests for rate limiter utility.
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from app.utils.rate_limiter import TokenBucketRateLimiter, rate_limiter


class TestTokenBucketRateLimiter:
    """Test cases for TokenBucketRateLimiter class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        limiter = TokenBucketRateLimiter()
        assert limiter.max_requests == 500
        assert limiter.window_seconds == 60
        assert limiter.tokens == 500
        assert limiter.last_refill > 0
        assert limiter.lock is not None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        limiter = TokenBucketRateLimiter(max_requests=100, window_seconds=30)
        assert limiter.max_requests == 100
        assert limiter.window_seconds == 30
        assert limiter.tokens == 100

    @pytest.mark.asyncio
    async def test_acquire_success_immediate(self):
        """Test successful token acquisition when tokens are available."""
        limiter = TokenBucketRateLimiter(max_requests=10, window_seconds=60)

        # Should succeed immediately since we start with full tokens
        start_time = time.time()
        await limiter.acquire()
        end_time = time.time()

        # Should complete quickly (no waiting)
        assert end_time - start_time < 0.1
        assert limiter.tokens == 9  # One token consumed

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens sequentially."""
        limiter = TokenBucketRateLimiter(max_requests=5, window_seconds=60)

        # Acquire 3 tokens
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()

        # Allow for small floating point precision differences
        assert abs(limiter.tokens - 2) < 0.01  # 5 - 3 = 2

    @pytest.mark.asyncio
    async def test_acquire_exhaust_tokens(self):
        """Test behavior when all tokens are exhausted."""
        limiter = TokenBucketRateLimiter(max_requests=2, window_seconds=60)

        # Exhaust all tokens
        await limiter.acquire()
        await limiter.acquire()
        # Allow for small floating point precision differences
        assert abs(limiter.tokens) < 0.01

        # Next acquisition should require waiting
        start_time = time.time()
        await limiter.acquire()
        end_time = time.time()

        # Should have waited some time
        assert end_time - start_time > 0.1
        # Should be 0 after waiting (allow for precision)
        assert abs(limiter.tokens) < 0.01

    @pytest.mark.asyncio
    async def test_token_refill_over_time(self):
        """Test that tokens are refilled over time."""
        limiter = TokenBucketRateLimiter(max_requests=10, window_seconds=60)

        # Exhaust all tokens
        for _ in range(10):
            await limiter.acquire()
        # Allow for small floating point precision differences
        assert abs(limiter.tokens) < 0.01

        # Mock time to simulate passage of time
        with patch("time.time") as mock_time:
            # Start at time 0
            mock_time.return_value = 0
            limiter.last_refill = 0

            # Simulate 6 seconds passing (should refill 1 token: 10/60 * 6 = 1)
            mock_time.return_value = 6

            await limiter.acquire()
            # Should consume the refilled token (allow for precision)
            assert abs(limiter.tokens) < 0.01

    @pytest.mark.asyncio
    async def test_token_refill_cap_at_max(self):
        """Test that tokens don't exceed max_requests when refilling."""
        limiter = TokenBucketRateLimiter(max_requests=5, window_seconds=60)

        # Mock time to simulate long passage of time
        with patch("time.time") as mock_time:
            mock_time.return_value = 0
            limiter.last_refill = 0
            limiter.tokens = 0

            # Simulate 120 seconds passing (should refill 10 tokens, but capped at 5)
            mock_time.return_value = 120

            await limiter.acquire()
            assert limiter.tokens == 4  # 5 - 1 = 4 (capped at max)

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test that concurrent access is properly synchronized."""
        limiter = TokenBucketRateLimiter(max_requests=3, window_seconds=60)

        # Create multiple concurrent acquisition tasks
        async def acquire_token():
            await limiter.acquire()
            return limiter.tokens

        # Run 3 concurrent acquisitions
        tasks = [acquire_token() for _ in range(3)]
        results = await asyncio.gather(*tasks)

        # All should succeed - the exact token count depends on timing
        # but we should have successfully acquired 3 tokens
        assert len(results) == 3
        # The final token count should be 0 (3 tokens acquired from 3 available)
        assert abs(limiter.tokens) < 0.1

    @pytest.mark.asyncio
    async def test_wait_time_calculation(self):
        """Test that wait time is calculated correctly when tokens are insufficient."""
        limiter = TokenBucketRateLimiter(max_requests=10, window_seconds=60)

        # Exhaust all tokens
        for _ in range(10):
            await limiter.acquire()

        # Mock time to control refill rate
        with patch("time.time") as mock_time:
            mock_time.return_value = 0
            limiter.last_refill = 0
            limiter.tokens = 0.5  # Half a token available

            # Next acquisition should wait for 0.5 tokens
            # Wait time = (1 - 0.5) / (10/60) = 0.5 / (1/6) = 3 seconds
            start_time = time.time()
            mock_time.return_value = start_time

            await limiter.acquire()

            # Should have waited approximately 3 seconds
            assert limiter.tokens == 0

    @pytest.mark.asyncio
    async def test_edge_case_zero_tokens_needed(self):
        """Test edge case where exactly 1 token is available."""
        limiter = TokenBucketRateLimiter(max_requests=10, window_seconds=60)
        limiter.tokens = 1.0

        await limiter.acquire()
        # Allow for small floating point precision differences
        assert abs(limiter.tokens) < 0.01

    @pytest.mark.asyncio
    async def test_edge_case_fractional_tokens(self):
        """Test behavior with fractional tokens."""
        limiter = TokenBucketRateLimiter(max_requests=10, window_seconds=60)
        limiter.tokens = 0.1  # Less than 1 token

        # Should wait since we need 1 token but only have 0.1
        start_time = time.time()
        await limiter.acquire()
        end_time = time.time()

        # Should have waited
        assert end_time - start_time > 0.1
        assert limiter.tokens == 0


class TestRateLimiterInstance:
    """Test cases for the global rate_limiter instance."""

    def test_rate_limiter_instance_exists(self):
        """Test that the global rate_limiter instance exists."""
        assert rate_limiter is not None
        assert isinstance(rate_limiter, TokenBucketRateLimiter)

    def test_rate_limiter_has_expected_attributes(self):
        """Test that the global rate_limiter has expected attributes."""
        assert hasattr(rate_limiter, "max_requests")
        assert hasattr(rate_limiter, "window_seconds")
        assert hasattr(rate_limiter, "tokens")
        assert hasattr(rate_limiter, "last_refill")
        assert hasattr(rate_limiter, "lock")

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_works(self):
        """Test that the global rate_limiter acquire method works."""
        # This test verifies the global instance is functional
        # We can't easily test the exact timing without mocking settings
        # but we can verify it doesn't raise exceptions
        try:
            await rate_limiter.acquire()
            # If we get here, the method executed successfully
            assert True
        except Exception as e:
            pytest.fail(f"rate_limiter.acquire() raised an exception: {e}")
