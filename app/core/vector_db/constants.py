"""
Weaviate Vector Database Constants
Contains collection names and property definitions for Weaviate collections
"""

import weaviate.classes.config as wvc


class VectorDBCollectionNames:
    """Collection names for Weaviate vector database"""

    MIGRATION_HISTORY = "MigrationHistory"
    CONVERSATIONS = "Conversation"
    REFERENCE_DOCUMENTS = "ReferenceDocument"


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
