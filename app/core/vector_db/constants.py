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
    ROADMAP_OPPORTUNITIES = "RoadmapOpportunity"


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

    CATEGORY = wvc.Property(
        name="category", data_type=wvc.DataType.TEXT, description="Document category"
    )

    TAGS = wvc.Property(
        name="tags", data_type=wvc.DataType.TEXT_ARRAY, description="Document tags"
    )

    TENANT_ID = wvc.Property(
        name="tenant_id",
        data_type=wvc.DataType.TEXT,
        description="Tenant ID associated with the document",
    )

    @classmethod
    def get_all_properties(cls):
        """Get all properties for the ReferenceDocument collection"""
        return [cls.HEADING, cls.CONTENT, cls.CATEGORY, cls.TAGS, cls.TENANT_ID]


class RoadmapOpportunityProperties:
    """
    Properties for the RoadmapOpportunity collection — semantic duplicate detection for the
    Ally Product Roadmap board.

    THE OPPORTUNITY TEXT IS DELIBERATELY NOT STORED HERE. ally-be's Postgres
    (roadmap_opportunities.description) is the system of record and this collection is a
    DERIVED index holding vectors plus the minimum metadata needed to filter and reconcile.
    Duplicating the description would mean a write on every edit, and any missed write would
    feed a stale description into the LLM's duplicate judgement — while buying nothing, since
    ally-be already has every description in hand when it runs that step.

    `stage` is excluded for the same reason, only more so: it changes constantly through the
    admin workflow and would be the staleness hotspot.

    The Weaviate object UUID IS the roadmap_opportunities.id, so there is no separate id
    property to keep in sync and every write is idempotent by construction.
    """

    PRODUCT_GOAL = wvc.Property(
        name="product_goal",
        data_type=wvc.DataType.TEXT,
        description="Product goal name, for optionally scoping a search to one goal",
    )

    TEXT_HASH = wvc.Property(
        name="text_hash",
        data_type=wvc.DataType.TEXT,
        description="SHA-256 of the embedded text; lets ally-be detect a stale vector",
    )

    EMBEDDING_MODEL = wvc.Property(
        name="embedding_model",
        data_type=wvc.DataType.TEXT,
        description="Model that produced the vector, so a model change is detectable",
    )

    EMBEDDED_AT = wvc.Property(
        name="embedded_at",
        data_type=wvc.DataType.DATE,
        description="When the vector was generated",
    )

    @classmethod
    def get_all_properties(cls):
        """Get all properties for the RoadmapOpportunity collection"""
        return [
            cls.PRODUCT_GOAL,
            cls.TEXT_HASH,
            cls.EMBEDDING_MODEL,
            cls.EMBEDDED_AT,
        ]
