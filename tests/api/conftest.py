"""
Pytest configuration for API tests.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.schemas.conversation import IdentifyResponse
from app.schemas.summary import SummaryNoteAndTagsResponse, Tag
from uuid import uuid4

# Set test environment variables before importing anything
os.environ.update(
    {
        "ENV__ENV": "TEST",
        "LOG__LEVEL": "DEBUG",
        "WEAVIATE__HTTP_HOST": "localhost",
        "WEAVIATE__HTTP_PORT": "8080",
        "WEAVIATE__HTTP_SECURE": "false",
        "WEAVIATE__GRPC_HOST": "localhost",
        "WEAVIATE__GRPC_PORT": "50051",
        "WEAVIATE__GRPC_SECURE": "false",
        "WEAVIATE__CONCURRENT_REQUESTS": "10",
        "OPENAI__API_KEY": "test-key",
        "OPENAI__ORGANIZATION_ID": "test-org",
        "OPENAI__RATE_LIMIT": "100",
        "OPENAI__WINDOW_SECONDS": "60",
        "LANGSMITH__TRACING": "false",
        "LANGSMITH__ENDPOINT": "https://api.smith.langchain.com",
        "LANGSMITH__API_KEY": "test-key",
        "LANGSMITH__PROJECT": "test-project",
        "AWS__REGION": "us-east-1",
        "AWS__ACCESS_KEY_ID": "test-key",
        "AWS__SECRET_ACCESS_KEY": "test-secret",
        "AWS__ENDPOINT_URL": "http://localhost:4566",
        "QUEUE__TRANSCRIBE_AND_SUMMARIZE_RESULTS_BUCKET": "test-bucket",
        "QUEUE__TRANSCRIPTION_RESULTS_QUEUE_URL": "http://localhost:4566/test-queue",
        "QUEUE__TRANSCRIBE_AND_SUMMARIZE_RESPONSE_QUEUE_URL": (
            "http://localhost:4566/response-queue"
        ),
        "LLM__MAX_CONCURRENT_LLM_CALLS": "10",
        "SLACK_ALERTS__ENABLED": "false",
        "SLACK_ALERTS__API_TOKEN": "test-token",
        "SLACK_ALERTS__CHANNEL_ID": "test-channel",
        "SLACK_ALERTS__LOG_LEVEL": "WARNING",
    }
)


@pytest.fixture(autouse=True)
def mock_openai_clients():
    """Mock OpenAI clients for API tests."""
    with (
        patch(
            "app.core.text_generations.openai_text_generation_client."
            "OpenAITextGenerationClient.get_client"
        ) as mock_text_client,
        patch(
            "app.core.embeddings.openai_embedding_client."
            "OpenAIEmbeddingClient.get_client"
        ) as mock_embedding_client,
        patch(
            "app.core.text_generations.openai_text_generation_service."
            "OpenAITextGenerationService.identify_user"
        ) as mock_identify_user,
        patch(
            "app.core.text_generations.openai_text_generation_service."
            "OpenAITextGenerationService.generate_nudge"
        ) as mock_generate_nudge,
        patch(
            "app.core.conversations.conversation_service." "ConversationService.analyze"
        ) as mock_analyze,
        patch(
            "app.core.conversations.conversation_service."
            "ConversationService.identify"
        ) as mock_identify,
        patch(
            "app.core.summaries.summary_service."
            "SummaryService.generate_summary_and_tags"
        ) as mock_generate_summary,
        patch(
            "app.core.summaries.summary_service." "SummaryService.enhance_content"
        ) as mock_enhance_content,
        patch(
            "app.core.summaries.summary_service."
            "SummaryService.get_tag_positivity_ratings"
        ) as mock_get_tag_ratings,
        patch(
            "app.core.summaries.summary_service."
            "SummaryService.generate_simulation_summary"
        ) as mock_generate_simulation,
        patch(
            "app.core.reference_documents.reference_document_service."
            "ReferenceDocumentService.create_document"
        ) as mock_create_document,
        patch(
            "app.core.reference_documents.reference_document_service."
            "ReferenceDocumentService.update_document"
        ) as mock_update_document,
        patch(
            "app.core.reference_documents.reference_document_service."
            "ReferenceDocumentService.delete_document"
        ) as mock_delete_document,
        patch(
            "app.core.reference_documents.reference_document_service."
            "ReferenceDocumentService.get_document"
        ) as mock_get_document,
        patch(
            "app.core.reference_documents.reference_document_service."
            "ReferenceDocumentService.search_documents"
        ) as mock_search_documents,
    ):

        # Mock text generation client - return a proper mock object, not AsyncMock
        mock_text_client_instance = MagicMock()

        # Create a mock response that has the expected attributes
        mock_response = MagicMock()
        mock_response.nudge = "nudge1"
        mock_response.stage = "stage1"

        # Configure ainvoke to return different responses based on the structured output
        # model
        def mock_ainvoke(prompt, structured_output_model=None, **kwargs):
            return mock_response

        mock_text_client_instance.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        mock_text_client_instance.with_structured_output = MagicMock(
            return_value=mock_text_client_instance
        )
        mock_text_client.return_value = mock_text_client_instance

        # Mock embedding client - return a proper mock object, not AsyncMock
        mock_embedding_client_instance = MagicMock()
        mock_embedding_client_instance.aembed_many = AsyncMock()
        mock_embedding_client_instance.aembed_query = AsyncMock()
        mock_embedding_client.return_value = mock_embedding_client_instance

        # Mock the text generation service methods directly
        mock_identify_user.return_value = IdentifyResponse(
            speaker0="client", speaker1="counselor"
        )
        mock_generate_nudge.return_value = "nudge1"

        # Mock the conversation service methods directly - set default return values
        mock_analyze.return_value = ("stage1", "nudge1")
        mock_identify.return_value = IdentifyResponse(
            speaker0="client", speaker1="counselor"
        )

        # Mock the summary service methods directly - set default return values
        mock_generate_summary.return_value = SummaryNoteAndTagsResponse(
            session_summary="Test summary",
            tags=[Tag(tag="anxiety", positivity_rating=2)],
            call_quality=85,
        )
        mock_enhance_content.return_value = "Enhanced content"
        mock_get_tag_ratings.return_value = [Tag(tag="anxiety", positivity_rating=2)]
        mock_generate_simulation.return_value = {
            "improvements": ["Ask more open-ended questions"],
            "positives": ["Good rapport building"],
        }

        # Mock the reference document service methods directly - set default return
        # values
        from uuid import uuid4

        test_doc_id = str(uuid4())
        mock_create_document.return_value = test_doc_id
        mock_update_document.return_value = None
        mock_delete_document.return_value = None
        mock_get_document.return_value = {
            "id": test_doc_id,
            "heading": "Test Document",
            "content": "This is a test document for reference.",
            "category": "test",
            "tags": ["test", "example"],
            "tenant_id": "test-tenant",
        }
        mock_search_documents.return_value = {
            "documents": [
                {
                    "id": test_doc_id,
                    "heading": "Test Document",
                    "content": "This is a test document for reference.",
                    "category": "test",
                    "tags": ["test", "example"],
                    "tenant_id": "test-tenant",
                    "score": 0.95,
                }
            ],
            "total_count": 1,
        }

        yield {
            "text_client": mock_text_client_instance,
            "embedding_client": mock_embedding_client_instance,
        }


@pytest.fixture(autouse=True)
def mock_weaviate_client():
    """Mock Weaviate client for API tests."""
    with patch(
        "app.core.vector_db.weaviate_client.WeaviateClient.get_client"
    ) as mock_client:
        mock_client_instance = MagicMock()

        # Mock collection and its methods
        mock_collection = MagicMock()

        # Mock query results with proper data structure
        mock_query_result = MagicMock()
        mock_query_result.objects = [
            MagicMock(
                properties={
                    "stage": "stage1",
                    "nudge": "nudge1",
                    "conversation": "test conversation",
                }
            )
        ]
        mock_collection.query.near_vector = AsyncMock(return_value=mock_query_result)
        mock_collection.query.get = AsyncMock(return_value=mock_query_result)

        mock_collection.data.create = AsyncMock()
        mock_collection.data.update = AsyncMock()
        mock_collection.data.delete = AsyncMock()
        mock_collection.data.get = AsyncMock()

        # Mock the collection getter
        mock_client_instance.collections.get = MagicMock(return_value=mock_collection)

        mock_client.return_value = mock_client_instance
        yield mock_client_instance
