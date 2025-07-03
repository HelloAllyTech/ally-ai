from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


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

    @abstractmethod
    async def create_document(self, collection_name: str, document_data: Dict[str, Any], vector: List[float], document_id: str) -> str:
        """
        Create a new document in the vector database.

        Parameters:
            collection_name (str): Name of the collection to store the document in.
            document_data (Dict[str, Any]): Data to be stored in the document.
            vector (List[float]): The embedding vector for the document.
            document_id (str): UUID to use as the document ID.

        Returns:
            str: The ID of the created document.

        Raises:
            VectorDBInsertFailedException: If the document insertion fails.
        """
        pass

    @abstractmethod
    async def get_document_by_id(self, collection_name: str, document_id: str, include_vector: bool = True) -> Dict[
        str, Any]:
        """
        Get a document by its ID from the vector database.

        Parameters:
            collection_name (str): Name of the collection to retrieve the document from.
            document_id (str): ID of the document to retrieve.
            include_vector (bool): Whether to include the vector in the response.

        Returns:
            Dict[str, Any]: The document data.

        Raises:
            DocumentNotFoundException: If the document is not found.
        """
        pass

    @abstractmethod
    async def update_document(self, collection_name: str, document_id: str, document_data: Dict[str, Any],
                              vector: List[float]) -> None:
        """
        Update an existing document in the vector database.

        Parameters:
            collection_name (str): Name of the collection containing the document.
            document_id (str): ID of the document to update.
            document_data (Dict[str, Any]): Updated data for the document.
            vector (List[float]): Updated embedding vector.

        Raises:
            VectorDBUpdateFailedException: If the document update fails.
        """
        pass

    @abstractmethod
    async def delete_document(self, collection_name: str, document_id: str) -> None:
        """
        Delete a document from the vector database.

        Parameters:
            collection_name (str): Name of the collection containing the document.
            document_id (str): ID of the document to delete.

        Raises:
            DocumentNotFoundException: If the document is not found.
            VectorDBDeleteFailedException: If the document deletion fails.
        """
        pass

    @abstractmethod
    async def search_documents(
            self,
            collection_name: str,
            query: str,
            limit: int = 10,
            filters: Optional[Dict[str, Any]] = None,
            include_vector: bool = False
    ) -> Dict[str, Any]:
        """
        Search for documents based on semantic similarity to the query.

        Parameters:
            collection_name (str): Name of the collection to search in.
            query (str): The search query for semantic similarity.
            limit (int): Maximum number of results to return.
            filters (Optional[Dict[str, Any]]): Dictionary of filters to apply.
            include_vector (bool): Whether to include vectors in the response.

        Returns:
            Dict[str, Any]: Search results containing documents and pagination info.

        Raises:
            VectorDBSearchFailedException: If the search operation fails.
        """
        pass
