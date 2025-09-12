from enum import Enum
from typing import Final

import weaviate.classes.config as wvc
from pydantic import BaseModel, Field


class EmbeddingConstants:
    MODEL: Final[str] = "text-embedding-3-small"


class TextGenerationConstants:
    DEFAULT_MODEL: Final[str] = "gpt-4o-mini-2024-07-18"


class VectorDBCollectionNames:
    CONVERSATIONS: Final[str] = "Conversation"
    REFERENCE_DOCUMENTS: Final[str] = "ReferenceDocument"
    MIGRATION_HISTORY: Final[str] = "MigrationHistory"


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


# Migration System Constants
class MigrationHistoryProperties:
    """Properties for the MigrationHistory collection"""

    VERSION = wvc.Property(
        name="version", data_type=wvc.DataType.TEXT, description="Migration version"
    )

    NAME = wvc.Property(
        name="name", data_type=wvc.DataType.TEXT, description="Migration name"
    )

    DESCRIPTION = wvc.Property(
        name="description",
        data_type=wvc.DataType.TEXT,
        description="Migration description",
    )

    STATUS = wvc.Property(
        name="status", data_type=wvc.DataType.TEXT, description="Migration status"
    )

    CREATED_AT = wvc.Property(
        name="created_at",
        data_type=wvc.DataType.DATE,
        description="Migration creation timestamp",
    )

    COMPLETED_AT = wvc.Property(
        name="completed_at",
        data_type=wvc.DataType.DATE,
        description="Migration completion timestamp",
    )

    @classmethod
    def get_all_properties(cls):
        """Get all properties for the MigrationHistory collection"""
        return [
            cls.VERSION,
            cls.NAME,
            cls.DESCRIPTION,
            cls.STATUS,
            cls.CREATED_AT,
            cls.COMPLETED_AT,
        ]


class ConversationProperties:
    """Properties for the Conversation collection"""

    CHAT_ID = wvc.Property(
        name="chat_id", data_type=wvc.DataType.INT, description="Chat ID"
    )

    MESSAGE = wvc.Property(
        name="message", data_type=wvc.DataType.TEXT, description="Message content"
    )

    ROLE = wvc.Property(
        name="role", data_type=wvc.DataType.TEXT, description="User role"
    )

    TIMESTAMP = wvc.Property(
        name="timestamp", data_type=wvc.DataType.DATE, description="Message timestamp"
    )

    @classmethod
    def get_all_properties(cls):
        """Get all properties for the Conversation collection"""
        return [cls.CHAT_ID, cls.MESSAGE, cls.ROLE, cls.TIMESTAMP]


class ReferenceDocumentProperties:
    """Properties for the ReferenceDocument collection"""

    HEADING = wvc.Property(
        name="heading", data_type=wvc.DataType.TEXT, description="Document heading"
    )

    CONTENT = wvc.Property(
        name="content", data_type=wvc.DataType.TEXT, description="Document content"
    )

    CREATED_AT = wvc.Property(
        name="created_at", data_type=wvc.DataType.DATE, description="Creation timestamp"
    )

    @classmethod
    def get_all_properties(cls):
        """Get all properties for the ReferenceDocument collection"""
        return [cls.HEADING, cls.CONTENT, cls.CREATED_AT]
