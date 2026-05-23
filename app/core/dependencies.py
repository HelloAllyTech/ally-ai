# dependencies.py
import httpx
from fastapi import Depends
from functools import lru_cache
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


@lru_cache()
def get_ally_core_service_cached(client: httpx.AsyncClient) -> AllyCoreService:
    return AllyCoreService(client)

async def get_ally_core_service(
    client=Depends(get_ally_core_client),
) -> AllyCoreService:
    return get_ally_core_service_cached(client)


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


@lru_cache()
def get_embedding_service_cached(client: OpenAIEmbeddings) -> BaseEmbeddingService:
    return OpenAIEmbeddingService(client)

# Dependency for the OpenAI embedding service
def get_embedding_service(
    client=Depends(get_openai_embedding_client),
) -> BaseEmbeddingService:
    """
    Returns an instance of the BaseEmbeddingService.
    Uses the singleton OpenAI embedding client.
    """
    return get_embedding_service_cached(client)


@lru_cache()
def get_text_generation_service_cached(
    client: ChatOpenAI, embedding_service: BaseEmbeddingService
) -> BaseTextGenerationService:
    return OpenAITextGenerationService(
        client=client, embedding_service=embedding_service
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
    return get_text_generation_service_cached(client, embedding_service)


@lru_cache()
def get_vector_db_cached(
    client: WeaviateAsyncClient, embedding_service: BaseEmbeddingService
) -> VectorDB:
    return WeaviateDB(client, embedding_service)

# Dependency for the vector database, which combines the Weaviate client and
# embedding service
async def get_vector_db(
    client=Depends(get_weaviate_client),
    embedding_service=Depends(get_embedding_service),
) -> VectorDB:
    """
    Returns an instance of VectorDB.
    """
    return get_vector_db_cached(client, embedding_service)


@lru_cache()
def get_conversation_service_cached(
    text_generation_service: BaseTextGenerationService, vector_db: VectorDB
) -> ConversationService:
    return ConversationService(text_generation_service, vector_db)

# Dependency for the conversation service, which uses both the text generation
# service and vector DB
async def get_conversation_service(
    text_generation_service=Depends(get_text_generation_service),
    vector_db=Depends(get_vector_db),
) -> ConversationService:
    """
    Returns an instance of ConversationService.
    """
    return get_conversation_service_cached(text_generation_service, vector_db)


@lru_cache()
def get_summary_service_cached(
    text_generation_service: BaseTextGenerationService,
) -> SummaryService:
    return SummaryService(text_generation_service)

# Dependency for the summary service, which uses the text generation service
async def get_summary_service(
    text_generation_service=Depends(get_text_generation_service),
) -> SummaryService:
    """
    Returns an instance of SummaryService.
    """
    return get_summary_service_cached(text_generation_service)


@lru_cache()
def get_reference_document_service_cached(
    vector_db: VectorDB, embedding_service: BaseEmbeddingService
) -> ReferenceDocumentService:
    return ReferenceDocumentService(vector_db, embedding_service)

# Dependency for the reference document service
async def get_reference_document_service(
    vector_db=Depends(get_vector_db), embedding_service=Depends(get_embedding_service)
) -> ReferenceDocumentService:
    """
    Returns an instance of ReferenceDocumentService.
    """
    return get_reference_document_service_cached(vector_db, embedding_service)
