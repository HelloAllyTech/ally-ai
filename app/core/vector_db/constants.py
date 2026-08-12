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
    KNOWLEDGE_CHUNKS = "KnowledgeChunk"


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
    Properties for the RoadmapOpportunity collection — semantic duplicate detection for
    the Ally Product Roadmap board.

    THE OPPORTUNITY TEXT IS DELIBERATELY NOT STORED HERE. ally-be's Postgres
    (roadmap_opportunities.description) is the system of record and this collection is a
    DERIVED index holding vectors plus the minimum metadata needed to filter and
    reconcile. Duplicating the description would mean a write on every edit, and any
    missed write would feed a stale description into the LLM's duplicate judgement —
    while buying nothing, since ally-be already has every description in hand when it
    runs that step.

    `stage` is excluded for the same reason, only more so: it changes constantly through
    the admin workflow and would be the staleness hotspot.

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


class KnowledgeChunkProperties:
    """
    Properties for the KnowledgeChunk collection — passage-level retrieval for the
    WhatsApp Q&A bot that answers mental healthcare workers' questions.

    Distinct from ReferenceDocument, which is left untouched: that collection stores one
    object per document with a SINGLE embedding of the whole body, which is why it can
    only ever cite a whole document. Here one document becomes many chunks, so an answer
    can point at the specific passage it came from.

    THE CHUNK TEXT *IS* STORED HERE, unlike RoadmapOpportunity, which deliberately omits
    its source text. The difference is where the consuming loop runs. RoadmapOpportunity
    feeds a duplicate judgement that ally-be performs, and ally-be already holds every
    description in hand — so storing the text there would buy nothing and risk
    staleness. Retrieval-augmented answering runs INSIDE this service in a single call,
    so without the text every question would need a round trip back to ally-be for each
    hit.

    Staleness is closed structurally rather than by trust: chunk text is IMMUTABLE for a
    given (document_id, chunk_version, chunk_index). Editing a document bumps
    chunk_version in ally-be, which writes new chunk rows under new UUIDs and deletes
    the old vectors, so a chunk object never changes its text in place and there is
    nothing to keep in sync.

    `document_title` and `source_url` are denormalised for the same reason: a citation
    has to be renderable from the hit alone, without a back-call. They are the only two
    fields that can go stale (renaming a document does not rewrite its chunks); that is
    accepted deliberately, since a citation naming a document's previous title still
    identifies the right document, whereas a per-question round trip to resolve titles
    would cost latency on every answer.

    NO tenant_id. This corpus is deliberately GLOBAL — the bot is open to anyone with
    the number, so there is no tenant to scope by. If a private per-tenant corpus is
    ever needed it must be a NEW collection, not a filter bolted onto this one:
    retrieval that forgets a filter leaks, and an un-set filter is the easiest thing in
    the world to forget.

    The Weaviate object UUID IS ally-be's kb_document_chunks.id, so every write is
    idempotent by construction and a citation's chunk_id resolves straight back to the
    row holding its offsets.
    """

    DOCUMENT_ID = wvc.Property(
        name="document_id",
        data_type=wvc.DataType.TEXT,
        description=(
            "ally-be kb_documents.id this chunk belongs to; filter and delete key"
        ),
    )

    DOCUMENT_TITLE = wvc.Property(
        name="document_title",
        data_type=wvc.DataType.TEXT,
        description=(
            "Denormalised document title, so a citation renders without a back-call"
        ),
    )

    CHUNK_INDEX = wvc.Property(
        name="chunk_index",
        data_type=wvc.DataType.INT,
        description="Zero-based position of this chunk within its document",
    )

    TEXT = wvc.Property(
        name="text",
        data_type=wvc.DataType.TEXT,
        description=(
            "The passage itself — read back as grounding context for the answer"
        ),
    )

    CHAR_START = wvc.Property(
        name="char_start",
        data_type=wvc.DataType.INT,
        description="Offset of this passage into kb_documents.raw_text",
    )

    CHAR_END = wvc.Property(
        name="char_end",
        data_type=wvc.DataType.INT,
        description="End offset into kb_documents.raw_text",
    )

    PAGE_FROM = wvc.Property(
        name="page_from",
        data_type=wvc.DataType.INT,
        description=(
            "First source page, for a citable page number. 0 when not paginated"
        ),
    )

    PAGE_TO = wvc.Property(
        name="page_to",
        data_type=wvc.DataType.INT,
        description="Last source page. 0 when not paginated",
    )

    SECTION_PATH = wvc.Property(
        name="section_path",
        data_type=wvc.DataType.TEXT,
        description=(
            "Heading trail, e.g. 'Chapter 3 > Risk assessment'; cited when no page "
            "exists"
        ),
    )

    SOURCE_URL = wvc.Property(
        name="source_url",
        data_type=wvc.DataType.TEXT,
        description=(
            "Original URL, when the document was fetched from one; empty otherwise"
        ),
    )

    LANGUAGE = wvc.Property(
        name="language",
        data_type=wvc.DataType.TEXT,
        description=(
            "BCP-47 tag of the passage, so a mixed-language corpus stays measurable"
        ),
    )

    TAGS = wvc.Property(
        name="tags",
        data_type=wvc.DataType.TEXT_ARRAY,
        description="Document tags, copied onto each chunk so retrieval can be scoped",
    )

    TOKEN_COUNT = wvc.Property(
        name="token_count",
        data_type=wvc.DataType.INT,
        description=(
            "Tokens in `text`, so the agent can budget its context without "
            "re-tokenising"
        ),
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
        """Get all properties for the KnowledgeChunk collection"""
        return [
            cls.DOCUMENT_ID,
            cls.DOCUMENT_TITLE,
            cls.CHUNK_INDEX,
            cls.TEXT,
            cls.CHAR_START,
            cls.CHAR_END,
            cls.PAGE_FROM,
            cls.PAGE_TO,
            cls.SECTION_PATH,
            cls.SOURCE_URL,
            cls.LANGUAGE,
            cls.TAGS,
            cls.TOKEN_COUNT,
            cls.TEXT_HASH,
            cls.EMBEDDING_MODEL,
            cls.EMBEDDED_AT,
        ]
