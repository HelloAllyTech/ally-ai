from enum import Enum
from typing import Final

from pydantic import BaseModel, Field


class EmbeddingConstants:
    MODEL: Final[str] = "text-embedding-3-small"


class TextGenerationConstants:
    DEFAULT_MODEL: Final[str] = "gpt-4o-mini-2024-07-18"


class AgeRange(str, Enum):
    BELOW_FIVE = "0-5"
    SIX_TO_TWELVE = "6-12"
    TWELVE_TO_SEVENTEEN = "13-17"
    EIGHTEEN_TO_TWENTY_FOUR = "18-24"
    TWENTY_FIVE_TO_THIRTY_FOUR = "25-34"
    THIRTY_FIVE_TO_FORTY_FOUR = "35-44"
    FORTY_FIVE_TO_FIFTY_FOUR = "45-54"
    FIFTY_FIVE_TO_SIXTY_FOUR = "55-64"
    SIXTY_FIVE_PLUS = "65+"


class ReferenceDocumentConstants:
    """Model for reference document."""

    SIMILARITY_THRESHOLD: Final[float] = 0.5


class LanguageCode(str, Enum):
    """Enum for supported language codes."""

    ENGLISH = "en"
    HINDI = "hi"
    BENGALI = "bn"
    PUNJABI = "pa"
    GUJARATI = "gu"
    ORIYA = "or"
    TAMIL = "ta"
    TELUGU = "te"
    KANNADA = "kn"
    MALAYALAM = "ml"


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


class PipelineStage(str, Enum):
    """Stages of the transcribe-and-summarize pipeline.

    Tagged onto failures so an error can be attributed to a specific step
    (and forwarded to ally-core / Slack) instead of the generic
    "transcription failed". Ordered roughly by execution order.
    """

    REQUEST_PARSE = "request-parse"
    DOWNLOAD = "download"
    CONVERT = "convert"
    TRANSCRIBE = "transcribe"
    DIARIZE = "diarize"
    SUMMARIZE = "summarize"
    DELIVER = "deliver"


class SQSWorkerConstants:
    """Constants for SQS worker configuration."""

    MAX_MESSAGES: Final[int] = 10
    WAIT_TIME_SECONDS: Final[int] = 10
    # Must exceed the worst-case end-to-end processing time of a single message
    # (download + transcription + diarization + summary). Sarvam alone allows a
    # 600s job timeout, so a 120s visibility window let SQS redeliver the message
    # mid-flight and a second worker would process the same chat concurrently.
    # Keep this comfortably above the longest provider timeout.
    VISIBILITY_TIMEOUT: Final[int] = 900
    POLLING_INTERVAL: Final[int] = 0


class APISettings:
    API_V1_STR: str = "/api/v1"
    API_STR: str = "/api"
    X_API_KEY_HEADER: str = "x-api-key"
