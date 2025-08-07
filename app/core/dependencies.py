# dependencies.py

from fastapi import Depends
import boto3
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from weaviate.client import WeaviateAsyncClient

from app.core.constants import EmbeddingConstants, TextGenerationConstants
from app.core.embeddings.base import BaseEmbeddingService
from app.core.embeddings.openai_embedding_service import OpenAIEmbeddingService
from app.core.embeddings.openai_embedding_client import OpenAIEmbeddingClient
from app.core.reference_documents.reference_document_service import ReferenceDocumentService
from app.core.summaries.summary_service import SummaryService
from app.core.text_generations.base import BaseTextGenerationService
from app.core.text_generations.openai_text_generation_service import OpenAITextGenerationService
from app.core.text_generations.openai_text_generation_client import OpenAITextGenerationClient
from app.core.vector_db.base import VectorDB
from app.core.vector_db.weaviate import WeaviateDB
from app.core.vector_db.weaviate_client import WeaviateClient
from app.core.conversations.conversation_service import ConversationService

# Dependency for the Weaviate async client
async def get_weaviate_client() -> WeaviateAsyncClient:
    """
    Creates, connects, and yields a Weaviate async client.
    Ensures the client is properly closed after use.
    """
    return WeaviateClient.get_client()


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


# Dependency for the OpenAI embedding service
def get_embedding_service(
        client=Depends(get_openai_embedding_client)
) -> BaseEmbeddingService:
    """
    Returns an instance of the BaseEmbeddingService.
    Uses the singleton OpenAI embedding client.
    """
    return OpenAIEmbeddingService(client)


# Dependency for the OpenAI text generation service
def get_text_generation_service(
        client=Depends(get_openai_text_generation_client),
        embedding_service=Depends(get_embedding_service)
) -> BaseTextGenerationService:
    """
    Returns an instance of the BaseTextGenerationService.
    Uses the singleton OpenAI text generation client.
    """
    return OpenAITextGenerationService(client=client, embedding_service=embedding_service)


# Dependency for the vector database, which combines the Weaviate client and embedding service
async def get_vector_db(
        client=Depends(get_weaviate_client),
        embedding_service=Depends(get_embedding_service)
) -> VectorDB:
    """
    Returns an instance of VectorDB.
    """
    return WeaviateDB(client, embedding_service)


# Dependency for the conversation service, which uses both the text generation service and vector DB
async def get_conversation_service(
        text_generation_service=Depends(get_text_generation_service),
        vector_db=Depends(get_vector_db)
) -> ConversationService:
    """
    Returns an instance of ConversationService.
    """
    return ConversationService(text_generation_service, vector_db)


# Dependency for the summary service, which uses the text generation service
async def get_summary_service(
        text_generation_service=Depends(get_text_generation_service),
) -> SummaryService:
    """
    Returns an instance of SummaryService.
    """
    return SummaryService(text_generation_service)


# Dependency for the reference document service
async def get_reference_document_service(
        vector_db=Depends(get_vector_db),
        embedding_service=Depends(get_embedding_service)
) -> ReferenceDocumentService:
    """
    Returns an instance of ReferenceDocumentService.
    """
    return ReferenceDocumentService(vector_db, embedding_service)

