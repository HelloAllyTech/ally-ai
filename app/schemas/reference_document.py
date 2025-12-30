from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ReferenceDocumentCreate(BaseModel):
    """Schema for creating a new reference document."""

    heading: str = Field(..., description="Heading or title of the reference document")
    content: str = Field(..., description="Content of the reference document")
    category: str = Field(..., description="Category of the reference document")
    tags: List[str] = Field(
        default_factory=list, description="Tags associated with the reference document"
    )
    tenant_id: str = Field(
        ..., description="Tenant ID associated with the reference document"
    )
    document_id: UUID = Field(..., description="UUID to use as the document ID")


class ReferenceDocumentUpdate(BaseModel):
    """Schema for updating an existing reference document."""

    heading: Optional[str] = Field(
        None, description="Updated heading or title of the reference document"
    )
    content: Optional[str] = Field(
        None, description="Updated content of the reference document"
    )
    category: Optional[str] = Field(
        None, description="Updated category of the reference document"
    )
    tags: Optional[List[str]] = Field(
        None, description="Updated tags associated with the reference document"
    )


class ReferenceDocumentResponse(BaseModel):
    """Schema for reference document response."""

    id: str = Field(..., description="Unique identifier of the reference document")
    heading: str = Field(..., description="Heading or title of the reference document")
    content: str = Field(..., description="Content of the reference document")
    category: str = Field(..., description="Category of the reference document")
    tags: List[str] = Field(
        ..., description="Tags associated with the reference document"
    )
    tenant_id: str = Field(
        ..., description="Tenant ID associated with the reference document"
    )


class ReferenceDocumentDeleteResponse(BaseModel):
    """Schema for reference document deletion response."""

    success: bool = Field(..., description="Whether the deletion was successful")


class ReferenceDocumentSearchRequest(BaseModel):
    """Schema for reference document search request."""

    query: str = Field(..., description="Search query for semantic similarity")
    document_ids: Optional[List[str]] = Field(
        None, description="List of document IDs to restrict the search to"
    )
    limit: int = Field(10, description="Maximum number of results to return")
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Filters to apply to the search (supported keys: category, "
        "tags, tenant_id)",
    )
    sort_by: Optional[str] = Field(
        None, description="Field to sort by (e.g., 'heading', 'category')"
    )
    sort_order: str = Field("desc", description="Sort order, either 'asc' or 'desc'")


class ReferenceDocumentSearchItem(ReferenceDocumentResponse):
    """Schema for a single document in search results."""

    score: Optional[float] = Field(None, description="Similarity score of the match")


class ReferenceDocumentSearchResponse(BaseModel):
    """Schema for reference document search response."""

    documents: List[ReferenceDocumentSearchItem] = Field(
        ..., description="List of matching documents"
    )
    total: int = Field(
        ..., description="Total number of matching documents found (unlimited)"
    )
    limit: int = Field(..., description="Maximum number of results returned")
    categories: Dict[str, int] = Field(
        default_factory=dict, description="Count of documents by category"
    )
