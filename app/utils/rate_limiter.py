import asyncio
import time

from app.core.config import settings


class TokenBucketRateLimiter:
    def __init__(self, max_requests: int = 500, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.tokens = max_requests
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_refill

            # refill tokens
            tokens_to_add = elapsed * (self.max_requests / self.window_seconds)
            self.tokens = min(self.max_requests, self.tokens + tokens_to_add)
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return
            else:
                # need to wait
                wait_time = (1 - self.tokens) / (
                    self.max_requests / self.window_seconds
                )
                await asyncio.sleep(wait_time)
                self.tokens = 0


rate_limiter = TokenBucketRateLimiter(
    max_requests=settings.OPENAI.RATE_LIMIT,
    window_seconds=settings.OPENAI.WINDOW_SECONDS,
)
