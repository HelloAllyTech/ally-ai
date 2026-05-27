# dependencies.py
import httpx
from functools import lru_cache
from fastapi import Depends
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from weaviate.client import WeaviateAsyncClient

from app.core.ally_core import AllyCoreClient, AllyCoreService
from app.core.conversations.conversation_service import ConversationService
from app.core.embeddings.base import BaseEmbeddingService
from app.core.embeddings.openai_embedding_client import OpenAIEmbeddingClient
from app.core.embeddings.openai_embedding_service import OpenAIEmbeddingService
from app.core.reference_documents.reference_document_service import (
    ReferenceDocumentService,
)
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
