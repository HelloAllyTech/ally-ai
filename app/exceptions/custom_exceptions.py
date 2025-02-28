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
