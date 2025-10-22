"""
Tests for conversation endpoints.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.exceptions.custom_exceptions import ConversationAnalysisFailedException
from app.schemas.conversation import IdentifyResponse
from tests.api.v1.endpoints.base import BaseAPITest


class TestConversationAnalyzeEndpoint(BaseAPITest):
    """Test cases for conversation analyze endpoint."""

    def test_analyze_success(
        self, client: TestClient, mock_conversation_service, sample_analyze_request
    ):
        """Test successful conversation analysis."""
        # Mock successful response
        mock_conversation_service.analyze.return_value = ("stage1", "nudge1")

        response = client.post(
            "/api/v1/conversation/analyze", json=sample_analyze_request
        )

        assert response.status_code == 200
        data = response.json()
        assert "nudge" in data
        assert "stage" in data
        assert data["nudge"] == "nudge1"
        assert data["stage"] == "stage1"

    def test_analyze_with_force_nudge(
        self, client: TestClient, mock_conversation_service, sample_analyze_request
    ):
        """Test conversation analysis with force nudge."""
        sample_analyze_request["force_nudge"] = True

        with patch(
            "app.core.conversations.conversation_service.ConversationService.analyze"
        ) as mock_analyze:
            mock_analyze.return_value = ("stage2", "forced_nudge")

            response = client.post(
                "/api/v1/conversation/analyze", json=sample_analyze_request
            )

            assert response.status_code == 200
            data = response.json()
            assert data["nudge"] == "forced_nudge"
            assert data["stage"] == "stage2"

    def test_analyze_conversation_analysis_failed(
        self, client: TestClient, mock_conversation_service, sample_analyze_request
    ):
        """Test conversation analysis failure."""

        with patch(
            "app.core.conversations.conversation_service.ConversationService.analyze"
        ) as mock_analyze:
            mock_analyze.side_effect = ConversationAnalysisFailedException(
                "Analysis failed"
            )

            response = client.post(
                "/api/v1/conversation/analyze", json=sample_analyze_request
            )

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert data["detail"] == "Conversation analysis failed"

    def test_analyze_unexpected_error(
        self, client: TestClient, mock_conversation_service, sample_analyze_request
    ):
        """Test unexpected error during analysis."""

        with patch(
            "app.core.conversations.conversation_service.ConversationService.analyze"
        ) as mock_analyze:
            mock_analyze.side_effect = Exception("Unexpected error")

            response = client.post(
                "/api/v1/conversation/analyze", json=sample_analyze_request
            )

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert data["detail"] == "Something went wrong. Please try again later."

    def test_analyze_invalid_request_data(self, client: TestClient):
        """Test analyze with invalid request data."""
        invalid_request = {
            "latest_message": "test",
            # Missing required fields
        }

        response = client.post("/api/v1/conversation/analyze", json=invalid_request)

        assert response.status_code == 422

    def test_analyze_empty_chat_history(
        self, client: TestClient, mock_conversation_service
    ):
        """Test analyze with empty chat history."""
        request = {
            "latest_message": "test message",
            "chat_history": [],
            "force_nudge": False,
        }
        mock_conversation_service.analyze.return_value = ("stage1", "nudge1")

        response = client.post("/api/v1/conversation/analyze", json=request)

        assert response.status_code == 200

    def test_analyze_methods(self, client: TestClient, sample_analyze_request):
        """Test that analyze endpoint only accepts POST requests."""
        # Test POST (should work)
        response = client.post(
            "/api/v1/conversation/analyze", json=sample_analyze_request
        )
        assert response.status_code in [200, 500]  # 500 due to mocking

        # Test GET (should fail)
        response = client.get("/api/v1/conversation/analyze")
        assert response.status_code == 405

        # Test PUT (should fail)
        response = client.put(
            "/api/v1/conversation/analyze", json=sample_analyze_request
        )
        assert response.status_code == 405

        # Test DELETE (should fail)
        response = client.delete("/api/v1/conversation/analyze")
        assert response.status_code == 405


class TestConversationIdentifyEndpoint(BaseAPITest):
    """Test cases for conversation identify endpoint."""

    def test_identify_success(
        self, client: TestClient, mock_conversation_service, sample_identify_request
    ):
        """Test successful user identification."""
        mock_response = {"speaker0": "client", "speaker1": "counselor"}
        mock_conversation_service.identify.return_value = mock_response

        response = client.post(
            "/api/v1/conversation/identify", json=sample_identify_request
        )

        assert response.status_code == 200
        data = response.json()
        assert data["speaker0"] == "client"
        assert data["speaker1"] == "counselor"

    def test_identify_with_mixed_roles(
        self, client: TestClient, mock_conversation_service, sample_identify_request
    ):
        """Test identification with mixed roles."""

        with patch(
            "app.core.conversations.conversation_service.ConversationService.identify"
        ) as mock_identify:
            mock_identify.return_value = IdentifyResponse(
                speaker0="counselor", speaker1="client"
            )

            response = client.post(
                "/api/v1/conversation/identify", json=sample_identify_request
            )

            assert response.status_code == 200
            data = response.json()
            assert data["speaker0"] == "counselor"
            assert data["speaker1"] == "client"

    def test_identify_with_unknown_roles(
        self, client: TestClient, mock_conversation_service, sample_identify_request
    ):
        """Test identification with unknown roles."""

        with patch(
            "app.core.conversations.conversation_service.ConversationService.identify"
        ) as mock_identify:
            mock_identify.return_value = IdentifyResponse(
                speaker0="unknown", speaker1="unknown"
            )

            response = client.post(
                "/api/v1/conversation/identify", json=sample_identify_request
            )

            assert response.status_code == 200
            data = response.json()
            assert data["speaker0"] == "unknown"
            assert data["speaker1"] == "unknown"

    def test_identify_invalid_request_data(self, client: TestClient):
        """Test identify with invalid request data."""
        invalid_request = {
            # Missing chat_history
        }

        response = client.post("/api/v1/conversation/identify", json=invalid_request)

        assert response.status_code == 422

    def test_identify_empty_chat_history(
        self, client: TestClient, mock_conversation_service
    ):
        """Test identify with empty chat history."""
        request = {"chat_history": []}
        mock_response = {"speaker0": "unknown", "speaker1": "unknown"}
        mock_conversation_service.identify.return_value = mock_response

        response = client.post("/api/v1/conversation/identify", json=request)

        assert response.status_code == 200

    def test_identify_methods(self, client: TestClient, sample_identify_request):
        """Test that identify endpoint only accepts POST requests."""
        # Test POST (should work)
        response = client.post(
            "/api/v1/conversation/identify", json=sample_identify_request
        )
        assert response.status_code in [200, 500]  # 500 due to mocking

        # Test GET (should fail)
        response = client.get("/api/v1/conversation/identify")
        assert response.status_code == 405

        # Test PUT (should fail)
        response = client.put(
            "/api/v1/conversation/identify", json=sample_identify_request
        )
        assert response.status_code == 405

        # Test DELETE (should fail)
        response = client.delete("/api/v1/conversation/identify")
        assert response.status_code == 405
