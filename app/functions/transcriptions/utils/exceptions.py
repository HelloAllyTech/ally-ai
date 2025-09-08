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
        # Don't store sensitive data in instance variables for HIPAA compliance
        # self.audio_source = audio_source  # Removed for HIPAA compliance
        # self.error_details = error_details  # Removed for HIPAA compliance

        # Don't include sensitive data in error messages for HIPAA compliance
        # if audio_source:
        #     message = f"Audio transcription failed for source: {audio_source}"
        # if error_details:
        #     message += f" - Details: {error_details}"

        super().__init__(message, status_code)
