# dependencies.py
from functools import lru_cache

import httpx
from fastapi import Depends, Query
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from weaviate.client import WeaviateAsyncClient

from app.core.ally_core import AllyCoreClient, AllyCoreService
from app.core.conversations.conversation_service import ConversationService
from app.core.embeddings.base import BaseEmbeddingService
from app.core.embeddings.openai_embedding_client import OpenAIEmbeddingClient
from app.core.embeddings.openai_embedding_service import OpenAIEmbeddingService
from app.core.knowledge_agent.agent import KnowledgeAgentService
from app.core.knowledge_base.corpus import KbCorpus, collection_for
from app.core.knowledge_base.knowledge_chunk_service import KnowledgeChunkService
from app.core.reference_documents.reference_document_service import (
    ReferenceDocumentService,
)
from app.core.roadmap.roadmap_opportunity_service import RoadmapOpportunityService
from app.core.summaries.summary_service import SummaryService
from app.core.text_generations.base import BaseTextGenerationService
from app.core.text_generations.openai_text_generation_client import (
    OpenAITextGenerationClient,
)
from app.core.text_generations.openai_text_generation_service import (
    OpenAITextGenerationService,
)
from app.core.vector_db.base import VectorDB
from app.core.vector_db.weaviate import WeaviateDB
from app.core.vector_db.weaviate_client import WeaviateClient


# Dependency for the Weaviate async client
async def get_weaviate_client() -> WeaviateAsyncClient:
    """
    Creates, connects, and yields a Weaviate async client.
    Ensures the client is properly closed after use.
    """
    return WeaviateClient.get_client()


async def get_ally_core_client() -> httpx.AsyncClient:
    return AllyCoreClient.get_client()


@lru_cache(maxsize=1)
def _get_ally_core_service_cached() -> AllyCoreService:
    return AllyCoreService(AllyCoreClient.get_client())


async def get_ally_core_service(
    client=Depends(get_ally_core_client),
) -> AllyCoreService:
    return _get_ally_core_service_cached()


# Dependency for the OpenAI embedding client
def get_openai_embedding_client() -> OpenAIEmbeddings:
    """
    Returns the singleton instance of the OpenAI embedding client.
    """
    return OpenAIEmbeddingClient.get_client()


# Dependency for the OpenAI text generation client
def get_openai_text_generation_client() -> ChatOpenAI:
    """
    Returns the singleton instance of the OpenAI text generation client.
    """
    return OpenAITextGenerationClient.get_client()


@lru_cache(maxsize=1)
def _get_embedding_service_cached() -> BaseEmbeddingService:
    return OpenAIEmbeddingService(OpenAIEmbeddingClient.get_client())


# Dependency for the OpenAI embedding service
def get_embedding_service(
    client=Depends(get_openai_embedding_client),
) -> BaseEmbeddingService:
    """
    Returns an instance of the BaseEmbeddingService.
    Uses the singleton OpenAI embedding client.
    """
    return _get_embedding_service_cached()


@lru_cache(maxsize=1)
def _get_text_generation_service_cached() -> BaseTextGenerationService:
    return OpenAITextGenerationService(
        client=OpenAITextGenerationClient.get_client(),
        embedding_service=_get_embedding_service_cached(),
    )


# Dependency for the OpenAI text generation service
def get_text_generation_service(
    client=Depends(get_openai_text_generation_client),
    embedding_service=Depends(get_embedding_service),
) -> BaseTextGenerationService:
    """
    Returns an instance of the BaseTextGenerationService.
    Uses the singleton OpenAI text generation client.
    """
    return _get_text_generation_service_cached()


@lru_cache(maxsize=1)
def _get_vector_db_cached() -> VectorDB:
    return WeaviateDB(
        WeaviateClient.get_client(),
        _get_embedding_service_cached(),
    )


# Dependency for the vector database, which combines the Weaviate client and
# embedding service
async def get_vector_db(
    client=Depends(get_weaviate_client),
    embedding_service=Depends(get_embedding_service),
) -> VectorDB:
    """
    Returns an instance of VectorDB.
    """
    return _get_vector_db_cached()


@lru_cache(maxsize=1)
def _get_conversation_service_cached() -> ConversationService:
    return ConversationService(
        _get_text_generation_service_cached(),
        _get_vector_db_cached(),
    )


# Dependency for the conversation service, which uses both the text generation
# service and vector DB
async def get_conversation_service(
    text_generation_service=Depends(get_text_generation_service),
    vector_db=Depends(get_vector_db),
) -> ConversationService:
    """
    Returns an instance of ConversationService.
    """
    return _get_conversation_service_cached()


@lru_cache(maxsize=1)
def _get_summary_service_cached() -> SummaryService:
    return SummaryService(_get_text_generation_service_cached())


# Dependency for the summary service, which uses the text generation service
async def get_summary_service(
    text_generation_service=Depends(get_text_generation_service),
) -> SummaryService:
    """
    Returns an instance of SummaryService.
    """
    return _get_summary_service_cached()


@lru_cache(maxsize=1)
def _get_roadmap_opportunity_service_cached() -> RoadmapOpportunityService:
    return RoadmapOpportunityService(
        _get_vector_db_cached(),
        _get_embedding_service_cached(),
    )


# Dependency for the roadmap opportunity (semantic duplicate detection) service
async def get_roadmap_opportunity_service(
    vector_db=Depends(get_vector_db), embedding_service=Depends(get_embedding_service)
) -> RoadmapOpportunityService:
    """
    Returns an instance of RoadmapOpportunityService.
    """
    return _get_roadmap_opportunity_service_cached()


# One cached instance per corpus, keyed by it. Each is bound to its own collection at
# construction, so a request cannot end up reading the wrong corpus partway through.
@lru_cache(maxsize=len(KbCorpus))
def _get_knowledge_chunk_service_cached(corpus: KbCorpus) -> KnowledgeChunkService:
    return KnowledgeChunkService(
        _get_vector_db_cached(),
        _get_embedding_service_cached(),
        collection_for(corpus),
    )


# Dependency for a knowledge-chunk corpus service
async def get_knowledge_chunk_service(
    corpus: KbCorpus = Query(
        KbCorpus.WHATSAPP_QA,
        description=(
            "Which corpus to act on. Each resolves to its own Weaviate collection."
        ),
    ),
    vector_db=Depends(get_vector_db),
    embedding_service=Depends(get_embedding_service),
) -> KnowledgeChunkService:
    """
    Returns the KnowledgeChunkService bound to `corpus`'s collection.

    DEFAULTED, not required, and only for the deploy window: ally-ai ships before
    ally-be (the vector index has to accept the new corpus before anything writes to
    it), and a required field would 422 every call the currently-deployed ally-be makes
    in between — taking the live WhatsApp bot down for the gap.

    The default is safe here in a way it would not be with one shared collection. A
    caller who omits it reads the WhatsApp collection, which holds only WhatsApp
    material: the failure is an empty, obviously-wrong result, not another corpus's
    passages arriving as if they answered the question. Once both services are on this
    version the default can be dropped.
    """
    return _get_knowledge_chunk_service_cached(corpus)


@lru_cache(maxsize=1)
def _get_knowledge_agent_service_cached() -> KnowledgeAgentService:
    return KnowledgeAgentService(
        _get_knowledge_chunk_service_cached(KbCorpus.WHATSAPP_QA)
    )


# Dependency for the knowledge agent (WhatsApp Q&A retrieval + answering) service
async def get_knowledge_agent_service() -> KnowledgeAgentService:
    """
    Returns an instance of KnowledgeAgentService.

    Pinned to the WhatsApp corpus, deliberately — this agent IS the WhatsApp bot's
    retrieve-and-answer loop, with its own decline thresholds and reply composition.
    The character interview agent does not reuse it: it retrieves through ally-be and
    composes in its own tool-calling loop, where the interviewer prompt decides what to
    do with a passage. Taking a corpus here would imply generality that does not exist.
    """
    return _get_knowledge_agent_service_cached()


@lru_cache(maxsize=1)
def _get_reference_document_service_cached() -> ReferenceDocumentService:
    return ReferenceDocumentService(
        _get_vector_db_cached(),
        _get_embedding_service_cached(),
    )


# Dependency for the reference document service
async def get_reference_document_service(
    vector_db=Depends(get_vector_db), embedding_service=Depends(get_embedding_service)
) -> ReferenceDocumentService:
    """
    Returns an instance of ReferenceDocumentService.
    """
    return _get_reference_document_service_cached()
