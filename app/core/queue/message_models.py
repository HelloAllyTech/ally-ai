from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """
    Enum for message types.
    """
    SUMMARY_REQUEST = "summary_request"
    SUMMARY_RESPONSE = "summary_response"
    CONVERSATION_ANALYSIS_REQUEST = "conversation_analysis_request"
    CONVERSATION_ANALYSIS_RESPONSE = "conversation_analysis_response"
    REFERENCE_DOCUMENT_REQUEST = "reference_document_request"
    REFERENCE_DOCUMENT_RESPONSE = "reference_document_response"
    ERROR = "error"


class BaseQueueMessage(BaseModel):
    """
    Base model for all queue messages.
    """
    message_type: MessageType
    message_id: str
    timestamp: int  # Unix timestamp in milliseconds
    correlation_id: Optional[str] = None  # For tracking related messages


class SummaryRequestMessage(BaseQueueMessage):
    """
    Message for requesting a summary of a conversation.
    """
    message_type: MessageType = MessageType.SUMMARY_REQUEST
    conversation_id: str
    messages: List[Dict[str, Any]]
    requested_fields: Optional[List[str]] = None
    tenant_id: Optional[str] = None


class SummaryResponseMessage(BaseQueueMessage):
    """
    Message containing the summary response.
    """
    message_type: MessageType = MessageType.SUMMARY_RESPONSE
    conversation_id: str
    summary: Dict[str, Any]
    requested_fields: Optional[List[str]] = None
    tenant_id: Optional[str] = None
    error: Optional[str] = None


class ConversationAnalysisRequestMessage(BaseQueueMessage):
    """
    Message for requesting analysis of a conversation.
    """
    message_type: MessageType = MessageType.CONVERSATION_ANALYSIS_REQUEST
    conversation_id: str
    messages: List[Dict[str, Any]]
    analysis_type: str
    parameters: Optional[Dict[str, Any]] = None
    tenant_id: Optional[str] = None


class ConversationAnalysisResponseMessage(BaseQueueMessage):
    """
    Message containing the conversation analysis response.
    """
    message_type: MessageType = MessageType.CONVERSATION_ANALYSIS_RESPONSE
    conversation_id: str
    analysis_type: str
    results: Dict[str, Any]
    tenant_id: Optional[str] = None
    error: Optional[str] = None


class ReferenceDocumentRequestMessage(BaseQueueMessage):
    """
    Message for requesting reference document operations.
    """
    message_type: MessageType = MessageType.REFERENCE_DOCUMENT_REQUEST
    operation: str  # "create", "update", "delete", "search"
    document_id: Optional[str] = None
    document_data: Optional[Dict[str, Any]] = None
    search_query: Optional[str] = None
    tenant_id: Optional[str] = None


class ReferenceDocumentResponseMessage(BaseQueueMessage):
    """
    Message containing the reference document operation response.
    """
    message_type: MessageType = MessageType.REFERENCE_DOCUMENT_RESPONSE
    operation: str
    document_id: Optional[str] = None
    results: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    tenant_id: Optional[str] = None
    error: Optional[str] = None


class ErrorMessage(BaseQueueMessage):
    """
    Message for reporting errors.
    """
    message_type: MessageType = MessageType.ERROR
    error_code: str
    error_message: str
    source_message_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
