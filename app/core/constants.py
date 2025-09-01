from enum import Enum
from typing import Final

from pydantic import BaseModel, Field


class EmbeddingConstants:
    MODEL: Final[str] = "text-embedding-3-small"


class TextGenerationConstants:
    DEFAULT_MODEL: Final[str] = "gpt-4o-mini-2024-07-18"


class VectorDBCollectionNames:
    CONVERSATIONS: Final[str] = "Conversation"
    REFERENCE_DOCUMENTS: Final[str] = "ReferenceDocument"


class AgeRange(str, Enum):
    EIGHTEEN_TO_TWENTY_FOUR = "18-24"
    TWENTY_FIVE_TO_THIRTY_FOUR = "25-34"
    THIRTY_FIVE_TO_FORTY_FOUR = "35-44"
    FORTY_FIVE_TO_FIFTY_FOUR = "45-54"
    FIFTY_FIVE_TO_SIXTY_FOUR = "55-64"
    SIXTY_FIVE_PLUS = "65+"


class ReferenceDocumentConstants(BaseModel):
    """Model for reference document."""

    SIMILARITY_THRESHOLD: Final[float] = 0.5


class Language(BaseModel):
    """Model for language and its percentage in the conversation."""

    language: str = Field(..., description="Name of the language")
    percentage: float = Field(
        ..., description="Percentage of the language used in conversation"
    )


class UserRole(str, Enum):
    CLIENT = "CLIENT"
    COUNSELOR = "COUNSELOR"


class ENV(str, Enum):
    DEV = "DEV"
    DEVELOPMENT = "DEVELOPMENT"
    PROD = "PROD"
    STG = "STG"


class SQSWorkerConstants:
    """Constants for SQS worker configuration."""

    MAX_MESSAGES: Final[int] = 10
    WAIT_TIME_SECONDS: Final[int] = 10
    VISIBILITY_TIMEOUT: Final[int] = 120
    POLLING_INTERVAL: Final[int] = 0

class APISettings:
    API_V1_STR: str = "/api/v1"
