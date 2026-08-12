import asyncio
from typing import Awaitable, Callable, List, TypeVar

from langchain_openai import OpenAIEmbeddings
from openai import APIConnectionError, RateLimitError

from app.core.constants import EmbeddingConstants
from app.core.embeddings.base import BaseEmbeddingService
from app.exceptions.custom_exceptions import EmbeddingFailedException
from app.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class OpenAIEmbeddingService(BaseEmbeddingService[OpenAIEmbeddings]):
    """
    OpenAI Embedding Service for generating embedding vectors.
    """

    def __init__(self, client: OpenAIEmbeddings) -> None:
        """
        Initialize the OpenAI embedding service with a client.

        Parameters:
            client (OpenAIEmbeddings): The OpenAI embedding client to use.
        """
        super().__init__(client)

    @staticmethod
    async def _with_retry(operation: Callable[[], Awaitable[T]], description: str) -> T:
        """
        Run an embedding call, retrying the two failures that are reliably transient.

        Only RateLimitError and APIConnectionError are retried. Anything else — an
        invalid key, an input over the token limit, a malformed request — will fail
        identically on every attempt, so retrying it just multiplies the latency before
        the caller learns what is wrong.

        Backoff is exponential from a small base and capped at a few attempts on
        purpose: the caller here is ally-be's ingest consumer, which already retries via
        SQS redelivery. The job of this loop is to ride out a brief 429 without bouncing
        work back to the queue, not to become a second retry system underneath one that
        already exists.
        """
        delay = EmbeddingConstants.RETRY_BASE_DELAY_SECONDS
        last_error: Exception | None = None

        for attempt in range(1, EmbeddingConstants.MAX_RETRIES + 1):
            try:
                return await operation()
            except (RateLimitError, APIConnectionError) as e:
                last_error = e
                if attempt == EmbeddingConstants.MAX_RETRIES:
                    break
                logger.warning(
                    f"{description} failed with {type(e).__name__} "
                    f"(attempt {attempt}/{EmbeddingConstants.MAX_RETRIES}); "
                    f"retrying in {delay}s"
                )
                await asyncio.sleep(delay)
                delay *= 2

        logger.exception(
            f"{description} failed after {EmbeddingConstants.MAX_RETRIES} attempts"
        )
        if isinstance(last_error, RateLimitError):
            raise EmbeddingFailedException(
                "OpenAI API rate limit exceeded. Please try again later."
            ) from last_error
        raise EmbeddingFailedException(
            "OpenAI API error. Please try again later."
        ) from last_error

    async def embed(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text using the OpenAI model.

        Parameters:
            text (str): The text to embed.

        Returns:
            List[float]: The resulting embedding vector.

        Raises:
            EmbeddingFailedException: If the OpenAI API rate limit is exceeded
            (triggering a RateLimitError) or
                if there is an API connection error (triggering an APIConnectionError).
        """
        return await self._with_retry(lambda: self.client.aembed_query(text), "embed")

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of texts using the OpenAI model.

        Splits into batches of EmbeddingConstants.BATCH_SIZE rather than sending
        everything in one request. The OpenAI embeddings endpoint caps a request at 2048
        inputs and 300k tokens (https://platform.openai.com/docs/guides/embeddings), and
        a chunked-corpus ingest can hand this several hundred passages at once — a
        300-page PDF is around 500 chunks, which as a single call would be a ~200k-token
        request that succeeds or fails as one indivisible unit. Batching means a
        transient failure costs one batch, not the whole document.

        Order is preserved across batches, which matters more than it looks: the caller
        zips these vectors back against its chunk list, so a reordering would silently
        attach the wrong embedding to a passage — retrievable, plausible, and wrong.
        Results are appended in batch order and never gathered concurrently, so the
        mapping cannot drift.

        Parameters:
            texts (List[str]): The list of texts to embed.

        Returns:
            List[List[float]]: The resulting embedding vectors, in the same order as
            `texts`.

        Raises:
            EmbeddingFailedException: If a batch still fails after its retries, or if
                the API
                returned a different number of vectors than were requested.
        """
        if not texts:
            return []

        batch_size = EmbeddingConstants.BATCH_SIZE
        vectors: List[List[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch_vectors = await self._with_retry(
                lambda b=batch: self.client.aembed_documents(b),
                f"embed_many batch {start}-{start + len(batch) - 1}",
            )

            # Never return a short or long result: the caller pairs these positionally,
            # so a count mismatch would misalign every subsequent vector rather than
            # failing visibly.
            if len(batch_vectors) != len(batch):
                logger.error(
                    f"Embedding count mismatch: asked for {len(batch)}, "
                    f"got {len(batch_vectors)}"
                )
                raise EmbeddingFailedException(
                    "OpenAI returned a different number of embeddings than requested."
                )

            vectors.extend(batch_vectors)

        return vectors
