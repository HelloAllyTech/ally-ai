from typing import Any, Dict, List, Optional
from uuid import UUID

from app.core.constants import VectorDBCollectionNames
from app.core.embeddings.base import BaseEmbeddingService
from app.core.vector_db.base import VectorDB
from app.exceptions.custom_exceptions import (
    DocumentAlreadyExistsException,
    DocumentNotFoundException,
    EmbeddingFailedException,
    VectorDBDeleteFailedException,
    VectorDBInsertFailedException,
    VectorDBSearchFailedException,
    VectorDBUpdateFailedException,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ReferenceDocumentService:
    def __init__(
        self, vector_db: VectorDB, embedding_service: BaseEmbeddingService
    ) -> None:
        """
        Initialize the ReferenceDocumentService with a vector
        database and embedding service.

        Parameters:
            vector_db (VectorDB): An instance of the vector database.
            embedding_service (BaseEmbeddingService): Service for generating
            embeddings.
        """
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.collection_name = VectorDBCollectionNames.REFERENCE_DOCUMENTS

    async def create_document(
        self,
        heading: str,
        content: str,
        category: str,
        tags: List[str],
        tenant_id: str,
        document_id: UUID,
    ) -> str:
        """
        Create a new reference document in the vector database.

        Parameters:
            heading (str): Heading or title of the document.
            content (str): Content of the document.
            category (str): Category of the document.
            tags (List[str]): Tags associated with the document.
            tenant_id (str): Tenant ID associated with the document.
            document_id (UUID): UUID to use as the document ID.

        Returns:
            str: The UUID of the created document.

        Raises:
            DocumentAlreadyExistsException: If a document with the given ID already
            exists.
            EmbeddingFailedException: If embedding generation fails.
            VectorDBInsertFailedException: If document insertion fails.
        """
        try:
            # Check if document with the given ID already exists
            try:
                existing_document = await self.vector_db.get_document_by_id(
                    collection_name=self.collection_name, document_id=str(document_id)
                )
                if existing_document:
                    raise DocumentAlreadyExistsException(
                        f"Reference document with ID {document_id} already exists"
                    )
            except DocumentNotFoundException:
                # Document doesn't exist, we can proceed with creation
                pass

            # Generate embedding for the content
            vector = await self.embedding_service.embed(content)

            # Prepare document data
            document_data = {
                "heading": heading,
                "content": content,
                "category": category,
                "tags": tags,
                "tenant_id": tenant_id,
            }

            # Create the document in the vector database
            document_id = await self.vector_db.create_document(
                collection_name=self.collection_name,
                document_data=document_data,
                vector=vector,
                document_id=str(document_id),
            )

            logger.info(f"Created reference document with ID: {document_id}")
            return document_id

        except DocumentAlreadyExistsException as e:
            logger.error(f"Document already exists: {type(e).__name__}")
            raise

        except EmbeddingFailedException as e:
            logger.error(f"Failed to generate embedding: {type(e).__name__}")
            raise

        except VectorDBInsertFailedException as e:
            logger.error(f"Failed to create reference document: {type(e).__name__}")
            raise

    async def update_document(
        self,
        document_id: str,
        heading: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Update an existing reference document in the vector database.

        Parameters:
            document_id (str): ID of the document to update.
            heading (Optional[str]): Updated heading or title.
            content (Optional[str]): Updated content.
            category (Optional[str]): Updated category.
            tags (Optional[List[str]]): Updated tags.

        Raises:
            DocumentNotFoundException: If the document is not found.
            EmbeddingFailedException: If embedding generation fails.
            VectorDBUpdateFailedException: If document update fails.
        """
        try:
            # Get the existing document to verify it exists and to use its values
            # if needed
            existing_document = await self.get_document(
                document_id, include_vector=True
            )

            # Prepare update data with non-None values or existing values
            update_data = {
                "heading": (
                    heading if heading is not None else existing_document["heading"]
                ),
                "category": (
                    category if category is not None else existing_document["category"]
                ),
                "tags": tags if tags is not None else existing_document["tags"],
                "tenant_id": existing_document[
                    "tenant_id"
                ],  # Always preserve tenant_id
            }

            # Handle content update and embedding generation if needed
            # Start with the existing vector
            vector = existing_document.get("vector").get("default")

            if content is not None:
                # Only generate a new vector if content has actually changed
                if content != existing_document["content"]:
                    update_data["content"] = content
                    vector = await self.embedding_service.embed(content)
                else:
                    # Content is the same as existing, just update the field
                    # without changing vector
                    update_data["content"] = content
            else:
                # Keep the existing content
                update_data["content"] = existing_document["content"]

            # Update the document in the vector database
            await self.vector_db.update_document(
                collection_name=self.collection_name,
                document_id=document_id,
                document_data=update_data,
                vector=vector,
            )

            logger.info(f"Updated reference document with ID: {document_id}")

        except DocumentNotFoundException:
            raise

        except EmbeddingFailedException as e:
            logger.error(f"Failed to generate embedding: {type(e).__name__}")
            raise

        except VectorDBUpdateFailedException as e:
            logger.error(f"Failed to update reference document: {type(e).__name__}")
            raise

    async def delete_document(self, document_id: str) -> None:
        """
        Delete a reference document from the vector database.

        Parameters:
            document_id (str): ID of the document to delete.

        Raises:
            DocumentNotFoundException: If the document is not found.
            VectorDBDeleteFailedException: If document deletion fails.
        """
        try:
            # Check if document exists
            try:
                await self.get_document(document_id)
            except Exception:
                raise DocumentNotFoundException(
                    f"Reference document with ID {document_id} not found"
                )

            # Delete the document
            await self.vector_db.delete_document(
                collection_name=self.collection_name, document_id=document_id
            )

            logger.info(f"Deleted reference document with ID: {document_id}")

        except DocumentNotFoundException:
            raise

        except VectorDBDeleteFailedException as e:
            logger.error(f"Failed to delete reference document: {type(e).__name__}")
            raise

    async def get_document(
        self, document_id: str, include_vector: bool = False
    ) -> Dict[str, Any]:
        """
        Get a reference document by its ID.

        Parameters:
            document_id (str): ID of the document to retrieve.
            include_vector (bool): Whether to include the vector in the response.

        Returns:
            Dict[str, Any]: The document data.

        Raises:
            DocumentNotFoundException: If the document is not found.
        """
        try:
            # Get the document
            result = await self.vector_db.get_document_by_id(
                collection_name=self.collection_name,
                document_id=document_id,
                include_vector=include_vector,
            )

            if not result:
                raise DocumentNotFoundException(
                    f"Reference document with ID {document_id} not found"
                )

            # Format the document data
            document = {
                "id": document_id,
                "heading": result.get("heading"),
                "content": result.get("content"),
                "category": result.get("category"),
                "tags": result.get("tags"),
                "tenant_id": result.get("tenant_id"),
            }

            # Include vector if it was requested and is available
            if include_vector and "vector" in result:
                document["vector"] = result["vector"]

            return document

        except DocumentNotFoundException:
            raise

        except Exception as e:
            logger.error(f"Failed to get document: {type(e).__name__}")
            raise DocumentNotFoundException(
                f"Reference document with ID {document_id} not found"
            )

    async def search_documents(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """
        Search for reference documents based on semantic similarity to the query.

        Parameters:
            query (str): The search query for semantic similarity.
            document_ids (Optional[List[str]]): List of document IDs to restrict
            the search to.
            limit (int): Maximum number of results to return.
            filters (Optional[Dict[str, Any]]): Dictionary of filters to apply.
                Supported keys: 'category' (str), 'tags' (List[str]), 'tenant_id' (str)
            sort_by (Optional[str]): Field to sort by (e.g., "heading", "category").
            sort_order (str): Sort order, either "asc" or "desc".

        Returns:
            Dict[str, Any]: Search results containing documents and pagination info.

        Raises:
            EmbeddingFailedException: If embedding generation fails.
            VectorDBSearchFailedException: If the search operation fails.
            ValueError: If invalid filter keys or values are provided.
        """
        try:
            # Validate filters if provided
            if filters:
                valid_filter_keys = {"category", "tags", "tenant_id"}
                invalid_keys = set(filters.keys()) - valid_filter_keys
                if invalid_keys:
                    raise ValueError(f"Invalid filter keys: {', '.join(invalid_keys)}")

            # Add document_ids to filters if provided
            if document_ids:
                if not filters:
                    filters = {}
                filters["id"] = document_ids

            # Search documents using the vector database
            results = await self.vector_db.search_documents(
                collection_name=self.collection_name,
                query=query,
                limit=limit,
                filters=filters,
            )

            # Format the results
            documents = []
            for obj in results["documents"]:
                doc = {
                    "id": obj["id"],
                    "heading": obj.get("heading", ""),
                    "content": obj.get("content", ""),
                    "category": obj.get("category", ""),
                    "tags": obj.get("tags", []),
                    "tenant_id": obj.get("tenant_id", ""),
                    "score": obj.get("score"),  # Include the score from the vector DB
                }
                documents.append(doc)

            # Apply client-side sorting if requested
            if sort_by and sort_by in ["heading", "category", "content"]:
                reverse = sort_order.lower() == "desc"
                documents.sort(key=lambda x: x.get(sort_by, ""), reverse=reverse)

            return {
                "documents": documents,
                "total": results["total"],
                "limit": limit,
                "categories": results["categories"],
            }

        except ValueError as e:
            logger.error(f"Invalid filter: {type(e).__name__}")
            raise

        except EmbeddingFailedException as e:
            logger.exception(f"Failed to generate embedding: {type(e).__name__}")
            raise

        except Exception as e:
            logger.exception(
                f"Failed to search reference documents: {type(e).__name__}"
            )
            raise VectorDBSearchFailedException(
                f"Failed to search reference documents: {type(e).__name__}"
            )
