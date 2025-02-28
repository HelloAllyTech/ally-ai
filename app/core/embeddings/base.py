from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingService[ClientT](ABC):
    """
    Abstract base class for embedding services using a client to generate embedding vectors.

    Parameters:
        client (ClientT): An instance of an embedding client.
    """

    def __init__(self, client: ClientT) -> None:
        """
        Initialize the embedding service with a client.

        Parameters:
            client (ClientT): An instance of an embedding client.
        """
        self.client = client

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.

        Parameters:
            text (str): The text to embed.

        Returns:
            List[float]: The resulting embedding vector.

        Raises:
            EmbeddingFailedException: If the embedding generation fails.
        """
        pass
