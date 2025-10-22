from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_reference_document_service
from app.core.reference_documents.reference_document_service import (
    ReferenceDocumentService,
)
from app.exceptions.custom_exceptions import (
    DocumentAlreadyExistsException,
    DocumentNotFoundException,
    EmbeddingFailedException,
    VectorDBDeleteFailedException,
    VectorDBInsertFailedException,
    VectorDBSearchFailedException,
    VectorDBUpdateFailedException,
)
from app.schemas.reference_document import (
    ReferenceDocumentCreate,
    ReferenceDocumentDeleteResponse,
    ReferenceDocumentResponse,
    ReferenceDocumentSearchRequest,
    ReferenceDocumentSearchResponse,
    ReferenceDocumentUpdate,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=ReferenceDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["reference_documents"],
)
async def create_reference_document(
    document: ReferenceDocumentCreate,
    reference_document_service: ReferenceDocumentService = Depends(
        get_reference_document_service
    ),
):
    """
    Create a new reference document.

    This endpoint creates a new reference document in the Weaviate database.
    The document content is used to generate an embedding vector.

    Returns the created document with its ID.
    """
    try:
        document_id = await reference_document_service.create_document(
            heading=document.heading,
            content=document.content,
            category=document.category,
            tags=document.tags,
            tenant_id=document.tenant_id,
            document_id=document.document_id,
        )

        # Get the created document
        document_data = await reference_document_service.get_document(document_id)

        return ReferenceDocumentResponse(**document_data)

    except DocumentAlreadyExistsException:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Reference document with ID {document.document_id} already exists",
        )

    except EmbeddingFailedException as e:
        logger.error(f"Embedding generation failed: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to generate embedding for document. Please try again later.",
        )

    except VectorDBInsertFailedException as e:
        logger.error(f"Document creation failed: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create reference document. Please try again later.",
        )

    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong. Please try again later.",
        )


@router.put(
    "/{document_id}",
    response_model=ReferenceDocumentResponse,
    tags=["reference_documents"],
)
async def update_reference_document(
    document_id: str,
    document: ReferenceDocumentUpdate,
    reference_document_service: ReferenceDocumentService = Depends(
        get_reference_document_service
    ),
):
    """
    Update an existing reference document.

    This endpoint updates an existing reference document in the Weaviate database.
    If the content is updated, a new embedding vector is generated.

    Returns the updated document.
    """
    try:
        await reference_document_service.update_document(
            document_id=document_id,
            heading=document.heading,
            content=document.content,
            category=document.category,
            tags=document.tags,
        )

        # Get the updated document
        document_data = await reference_document_service.get_document(document_id)

        return ReferenceDocumentResponse(**document_data)

    except DocumentNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference document with ID {document_id} not found",
        )

    except EmbeddingFailedException as e:
        logger.error(f"Embedding generation failed: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to generate embedding for document. Please try again later.",
        )

    except VectorDBUpdateFailedException as e:
        logger.error(f"Document update failed: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update reference document. Please try again later.",
        )

    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong. Please try again later.",
        )


@router.delete(
    "/{document_id}",
    response_model=ReferenceDocumentDeleteResponse,
    tags=["reference_documents"],
)
async def delete_reference_document(
    document_id: str,
    reference_document_service: ReferenceDocumentService = Depends(
        get_reference_document_service
    ),
):
    """
    Delete a reference document.

    This endpoint deletes a reference document from the Weaviate database.

    Returns a confirmation of deletion.
    """
    try:
        await reference_document_service.delete_document(document_id)

        return ReferenceDocumentDeleteResponse(success=True)

    except DocumentNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference document with ID {document_id} not found",
        )

    except VectorDBDeleteFailedException as e:
        logger.error(f"Document deletion failed: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete reference document. Please try again later.",
        )

    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong. Please try again later.",
        )


@router.get(
    "/{document_id}",
    response_model=ReferenceDocumentResponse,
    tags=["reference_documents"],
)
async def get_reference_document(
    document_id: str,
    reference_document_service: ReferenceDocumentService = Depends(
        get_reference_document_service
    ),
):
    """
    Get a reference document by its ID.

    This endpoint retrieves a reference document from the Weaviate database.

    Returns the document data.
    """
    try:
        document_data = await reference_document_service.get_document(document_id)
        return ReferenceDocumentResponse(**document_data)

    except DocumentNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference document with ID {document_id} not found",
        )

    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong. Please try again later.",
        )


@router.post(
    "/search",
    response_model=ReferenceDocumentSearchResponse,
    tags=["reference_documents"],
)
async def search_reference_documents(
    search_request: ReferenceDocumentSearchRequest,
    reference_document_service: ReferenceDocumentService = Depends(
        get_reference_document_service
    ),
):
    """
    Search for reference documents based on semantic similarity.

    This endpoint performs a vector search in the Weaviate database to find documents
    that are semantically similar to the provided query. The search can be filtered
    using the filters dictionary with supported keys: 'category', 'tags', and
    'tenant_id'.
    Results can be sorted and paginated.

    Returns a list of matching documents with their similarity scores.
    """
    try:
        search_results = await reference_document_service.search_documents(
            query=search_request.query,
            document_ids=search_request.document_ids,
            limit=search_request.limit,
            filters=search_request.filters,
            sort_by=search_request.sort_by,
            sort_order=search_request.sort_order,
        )

        return ReferenceDocumentSearchResponse(**search_results)

    except ValueError as e:
        logger.error(f"Invalid filter parameters: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=type(e).__name__
        )

    except EmbeddingFailedException as e:
        logger.error(f"Embedding generation failed: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to generate embedding for search query. "
            "Please try again later.",
        )

    except VectorDBSearchFailedException as e:
        logger.error(f"Document search failed: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search reference documents. Please try again later.",
        )

    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong. Please try again later.",
        )
