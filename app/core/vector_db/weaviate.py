from typing import List

import weaviate
import weaviate.classes as wvc
from weaviate.client import WeaviateAsyncClient
from weaviate.collections.classes.internal import QueryReturn
from weaviate.exceptions import WeaviateConnectionError, AuthenticationFailedException

from app.core.config import settings
from app.core.constants import VectorDBCollectionNames
from app.core.embeddings.base import BaseEmbeddingService
from app.core.vector_db.base import VectorDB
from app.exceptions.custom_exceptions import (
    VectorDBSearchFailedException,
    EmbeddingFailedException,
    VectorDBFetchFailedException
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class WeaviateClient:
    @staticmethod
    def create_client():
        return weaviate.use_async_with_custom(
            http_host=settings.WEAVIATE_HTTP_HOST,
            http_port=settings.WEAVIATE_HTTP_PORT,
            http_secure=settings.WEAVIATE_HTTP_SECURE,
            grpc_host=settings.WEAVIATE_GRPC_HOST,
            grpc_port=settings.WEAVIATE_GRPC_PORT,
            grpc_secure=settings.WEAVIATE_GRPC_SECURE,
        )

    @staticmethod
    async def connect(client: WeaviateAsyncClient) -> None:
        return await client.connect()

    @staticmethod
    async def close(client: WeaviateAsyncClient) -> None:
        return await client.close()


class WeaviateDB(VectorDB):
    def __init__(self, client, embedding_service: BaseEmbeddingService) -> None:
        self.embedding_service = embedding_service
        super().__init__(client)

    async def similarity_search(self, vector: List[float], top_k: int = 1) -> QueryReturn:
        """
        Perform a similarity search in the vector database using the provided vector.

        This method retrieves the "CONVERSATIONS" collection from the Weaviate client and executes
        a near vector query to find the top_k most similar items. The search request includes
        metadata retrieval with a certainty score.

        Parameters:
            vector (List[float]): The embedding vector to search for similar items.
            top_k (int, optional): The maximum number of similar results to return. Defaults to 1.

        Returns:
            QueryReturn: The result of the similarity search, including the matching items and associated metadata.

        Raises:
            VectorDBSearchFailedException: If a connection or authentication error occurs while querying Weaviate.
        """
        collection = self.client.collections.get(VectorDBCollectionNames.CONVERSATIONS)

        try:
            return await collection.query.near_vector(
                near_vector=vector,
                limit=top_k,
                return_metadata=wvc.query.MetadataQuery(certainty=True)
            )

        except WeaviateConnectionError as e:
            logger.exception(str(e))
            raise VectorDBSearchFailedException("Weaviate connection error. Please try again later.") from e

        except AuthenticationFailedException as e:
            logger.exception(str(e))
            raise VectorDBSearchFailedException("Weaviate authentication failed. Please try again later.") from e

    async def fetch_relevant_conversations(self, query: str, top_k: int = 1) -> QueryReturn:
        """
        Fetches the most relevant conversations from Weaviate for the given query.

        This function generates an embedding for the provided query using the embedding service,
        and then performs a similarity search in Weaviate to retrieve the top matching conversations.

        Parameters:
            query (str): The query string to search for relevant conversations.
            top_k (int, optional): The maximum number of conversations to return. Defaults to 1.

        Returns:
            QueryReturn: The result of the similarity search, which includes the most relevant conversations
                         and associated metadata.

        Raises:
            VectorDBFetchFailedException: If the embedding generation fails or if the similarity search encounters
                an error.
        """
        try:
            vector = await self.embedding_service.embed(query)

            logger.info(f"Fetching relevant conversations for query")
            return await self.similarity_search(vector, top_k)

        except EmbeddingFailedException as e:
            raise VectorDBFetchFailedException("Embedding failed. Please try again later.") from e

        except VectorDBSearchFailedException as e:
            raise VectorDBFetchFailedException("Weaviate search failed. Please try again later.") from e
