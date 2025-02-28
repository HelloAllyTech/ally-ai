from abc import ABC, abstractmethod
from typing import List

class VectorDB[T](ABC):
    """
    Abstract base class for vector-based databases.

    This class defines the structure for vector-based databases and requires
    subclasses to implement the similarity_search method.
    """

    def __init__(self, client: T) -> None:
        """
        Initialize the vector database interface.

        Parameters:
            client (T): The underlying client or connection to the database.
        """
        self.client = client

    @abstractmethod
    async def similarity_search[QueryResult](self, vector: List[float], top_k: int = 1) -> QueryResult:
        """
        Perform a similarity search given a vector.

        Parameters:
            vector (List[float]): The input vector to compare against the database.
            top_k (int, optional): The number of top results to return. Defaults to 1.

        Returns:
            List[Any]: A list of search results, with the length of the list
            not exceeding top_k.

        Raises:
            VectorDBSearchFailedException: If the search operation fails.
        """
        pass

    @abstractmethod
    async def fetch_relevant_conversations[QueryResult](self, query: str, top_k: int = 1) -> QueryResult:
        """
        Fetches the most relevant conversations from the database for the given query.

        Parameters:
            query (str): The query to search for.
            top_k (int): The number of top conversations to return.

        Returns:
            list: A list of the most relevant conversations.

        Raises:
            VectorDBFetchFailedException: If the fetch operation fails.
        """
        pass
