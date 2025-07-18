from fastapi import status


class BaseCustomException(Exception):
    """
    Base class for all custom exceptions.
    """

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR) -> None:
        super().__init__(message)  # Store message in the built-in Exception class
        self.status_code = status_code

    def __str__(self):
        """Return a formatted string representation of the error."""
        return f"{self.__class__.__name__}: {self.args[0]} (Status Code: {self.status_code})"


class ConversationAnalysisFailedException(BaseCustomException):
    """
    Raised when conversation analysis fails.
    """

    def __init__(self,
                 message="Conversation analysis failed",
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                 ) -> None:
        super().__init__(message, status_code)


class SummarizationFailedException(BaseCustomException):
    """
    Raised when summarization fails.
    """

    def __init__(self, message="Summarization failed", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR) -> None:
        super().__init__(message, status_code)


class VectorDBSearchFailedException(BaseCustomException):
    """
    Raised when vector DB similarity search fails.
    """

    def __init__(self,
                 message: str = "Vector DB similarity search failed",
                 status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
                 ) -> None:
        super().__init__(message, status_code)


class VectorDBFetchFailedException(BaseCustomException):
    """
    Raised when vector DB relevant conversation fetch fails.
    """

    def __init__(self,
                 message="Vector DB relevant conversation fetch failed",
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                 ) -> None:
        super().__init__(message, status_code)


class EmbeddingFailedException(BaseCustomException):
    """
    Raised when text embedding fails.
    """

    def __init__(self, message="Failed to embed text", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR) -> None:
        super().__init__(message, status_code)


class NudgeGenerationFailedException(BaseCustomException):
    """
    Raised when nudge generation fails.
    """

    def __init__(self, message="Nudge generation failed", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR) -> None:
        super().__init__(message, status_code)


class LLMInvocationFailedException(BaseCustomException):
    """
    Raised when LLM invocation fails.
    """

    def __init__(self, message="LLM invocation failed", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR) -> None:
        super().__init__(message, status_code)


class SummaryNoteFailedException(BaseCustomException):
    """
    Raised when summary note generation fails.
    """

    def __init__(self,
                 message="Failed to generate summary note",
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                 ) -> None:
        super().__init__(message, status_code)


class ContentEnhancementFailedException(BaseCustomException):
    """
    Raised when content enhancement fails.
    """

    def __init__(self,
                 message="Content enhancement failed",
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                 ) -> None:
        super().__init__(message, status_code)


class IdentifyUserFailedException(BaseCustomException):
    """
    Raised when user identification fails.
    """

    def __init__(self, 
                 message="User identification failed", 
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR) -> None:
        super().__init__(message, status_code)


class DocumentNotFoundException(BaseCustomException):
    """
    Raised when a document is not found.
    """

    def __init__(self, 
                 message="Document not found", 
                 status_code=status.HTTP_404_NOT_FOUND) -> None:
        super().__init__(message, status_code)


class DocumentAlreadyExistsException(BaseCustomException):
    """
    Raised when attempting to create a document with an ID that already exists.
    """

    def __init__(self, 
                 message="Document with this ID already exists", 
                 status_code=status.HTTP_409_CONFLICT) -> None:
        super().__init__(message, status_code)


class VectorDBInsertFailedException(BaseCustomException):
    """
    Raised when vector DB insertion fails.
    """

    def __init__(self,
                 message="Vector DB insertion failed",
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                 ) -> None:
        super().__init__(message, status_code)


class VectorDBUpdateFailedException(BaseCustomException):
    """
    Raised when vector DB update fails.
    """

    def __init__(self,
                 message="Vector DB update failed",
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                 ) -> None:
        super().__init__(message, status_code)


class VectorDBDeleteFailedException(BaseCustomException):
    """
    Raised when vector DB deletion fails.
    """

    def __init__(self,
                 message="Vector DB deletion failed",
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                 ) -> None:
        super().__init__(message, status_code)


class CoreAPIFailedException(BaseCustomException):
    """
    Raised when Core API calls fail.
    """

    def __init__(self,
                 message="Core API call failed",
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                 endpoint: str = None,
                 response_status: int = None,
                 response_body: str = None
                 ) -> None:
        self.endpoint = endpoint
        self.response_status = response_status
        self.response_body = response_body
        
        # Enhance message with additional context if available
        if endpoint:
            message = f"Backend API call failed for endpoint: {endpoint}"
        if response_status:
            message += f" (Status: {response_status})"
        if response_body:
            message += f" - Response: {response_body[:200]}..."  # Truncate long responses
            
        super().__init__(message, status_code)


class TranscriptionFailedException(BaseCustomException):
    """
    Raised when audio transcription fails.
    """

    def __init__(self,
                 message="Audio transcription failed",
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                 audio_source: str = None,
                 error_details: str = None
                 ) -> None:
        self.audio_source = audio_source
        self.error_details = error_details
        
        # Enhance message with additional context if available
        if audio_source:
            message = f"Audio transcription failed for source: {audio_source}"
        if error_details:
            message += f" - Details: {error_details}"
            
        super().__init__(message, status_code)
