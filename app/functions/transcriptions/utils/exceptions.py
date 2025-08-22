"""
Custom exceptions for the Lambda transcription function.
"""


class BaseCustomException(Exception):
    """
    Base class for all custom exceptions.
    """

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)  # Store message in the built-in Exception class
        self.status_code = status_code

    def __str__(self):
        """Return a formatted string representation of the error."""
        return (
            f"{self.__class__.__name__}: {self.args[0]} "
            f"(Status Code: {self.status_code})"
        )


class TranscriptionFailedException(BaseCustomException):
    """
    Raised when audio transcription fails.
    """

    def __init__(
        self,
        message="Audio transcription failed",
        status_code=500,
        audio_source: str = None,
        error_details: str = None,
    ) -> None:
        self.audio_source = audio_source
        self.error_details = error_details

        # Enhance message with additional context if available
        if audio_source:
            message = f"Audio transcription failed for source: {audio_source}"
        if error_details:
            message += f" - Details: {error_details}"

        super().__init__(message, status_code)
