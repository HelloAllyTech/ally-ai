from typing import List

from langchain_openai import OpenAIEmbeddings
from openai import APIConnectionError, RateLimitError

from app.core.embeddings.base import BaseEmbeddingService
from app.exceptions.custom_exceptions import EmbeddingFailedException
from app.utils.logger import get_logger

logger = get_logger(__name__)


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
        try:
            # TODO: Add retry mechanism
            return await self.client.aembed_query(text)

        except RateLimitError as e:
            logger.exception(type(e).__name__)
            raise EmbeddingFailedException(
                "OpenAI API rate limit exceeded. Please try again later."
            ) from e

        except APIConnectionError as e:
            logger.exception(type(e).__name__)
            raise EmbeddingFailedException(
                "OpenAI API error. Please try again later."
            ) from e

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of texts using the OpenAI model.

        Parameters:
            texts (List[str]): The list of texts to embed.

        Returns:
            List[List[float]]: The resulting embedding vectors.

        Raises:
            EmbeddingFailedException: If the OpenAI API rate limit is exceeded
            (triggering a RateLimitError) or
                if there is an API connection error (triggering an APIConnectionError).
        """
        try:
            # TODO: Add retry mechanism
            # TODO: Add divide and conquer to avoid rate limit.
            #  Not expecting exceed now, but OpenAI doesn't have a proper limit.
            #  Refer to
            #  https://community.openai.com/t/max-total-embeddings-tokens-per-request
            # /1254699?utm_source=chatgpt.com
            #  and https://platform.openai.com/docs/guides/embeddings/limitations

            return await self.client.aembed_documents(texts)

        except RateLimitError as e:
            logger.exception(type(e).__name__)
            raise EmbeddingFailedException(
                "OpenAI API rate limit exceeded. Please try again later."
            ) from e

        except APIConnectionError as e:
            logger.exception(type(e).__name__)
            raise EmbeddingFailedException(
                "OpenAI API error. Please try again later."
            ) from e
