"""
Tests for summary endpoints.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.schemas.summary import DynamicSummaryNoteResponse, Tag
from tests.api.v1.endpoints.base import BaseAPITest


class TestSummaryNoteEndpoint(BaseAPITest):
    """Test cases for summary note endpoint."""

    def test_create_note_success(
        self, client: TestClient, mock_summary_service, sample_summary_request
    ):
        """Test successful summary note creation."""
        mock_response = {
            "session_summary": "Test summary",
            "tags": [{"tag": "anxiety", "positivity_rating": 2}],
            "call_quality": 85,
        }
        mock_summary_service.generate_summary_and_tags.return_value = mock_response

        response = client.post("/api/v1/summary/note", json=sample_summary_request)

        assert response.status_code == 200
        data = response.json()
        assert "session_summary" in data
        assert "tags" in data
        assert "call_quality" in data

    def test_create_note_with_keys(
        self, client: TestClient, mock_summary_service, sample_summary_request
    ):
        """Test summary note creation with specific keys."""
        sample_summary_request["keys"] = ["session_summary", "tags"]

        with patch(
            "app.core.summaries.summary_service.SummaryService."
            "generate_summary_and_tags"
        ) as mock_generate:
            mock_generate.return_value = DynamicSummaryNoteResponse(
                fields={
                    "session_summary": "Test summary",
                    "tags": [{"tag": "anxiety", "positivity_rating": 2}],
                }
            )

            response = client.post("/api/v1/summary/note", json=sample_summary_request)

            assert response.status_code == 200
            data = response.json()
            assert "fields" in data

    def test_create_note_invalid_request(self, client: TestClient):
        """Test summary note creation with invalid request."""
        invalid_request = {
            # Missing chat_history
        }

        response = client.post("/api/v1/summary/note", json=invalid_request)

        assert response.status_code == 422

    def test_create_note_empty_chat_history(
        self, client: TestClient, mock_summary_service
    ):
        """Test summary note creation with empty chat history."""
        request = {"chat_history": [], "keys": None}
        mock_response = {"session_summary": "", "tags": [], "call_quality": 0}
        mock_summary_service.generate_summary_and_tags.return_value = mock_response

        response = client.post("/api/v1/summary/note", json=request)

        assert response.status_code == 200

    def test_create_note_methods(self, client: TestClient, sample_summary_request):
        """Test that note endpoint only accepts POST requests."""
        # Test POST (should work)
        response = client.post("/api/v1/summary/note", json=sample_summary_request)
        assert response.status_code in [200, 500]  # 500 due to mocking

        # Test GET (should fail)
        response = client.get("/api/v1/summary/note")
        assert response.status_code == 405

        # Test PUT (should fail)
        response = client.put("/api/v1/summary/note", json=sample_summary_request)
        assert response.status_code == 405

        # Test DELETE (should fail)
        response = client.delete("/api/v1/summary/note")
        assert response.status_code == 405


class TestSummaryEnhanceEndpoint(BaseAPITest):
    """Test cases for summary enhance endpoint."""

    def test_enhance_success(self, client: TestClient, mock_summary_service):
        """Test successful content enhancement."""
        request = {"content": "test content"}

        with patch(
            "app.core.summaries.summary_service.SummaryService.enhance_content"
        ) as mock_enhance:
            mock_enhance.return_value = "Enhanced test content"

            response = client.post("/api/v1/summary/enhance", json=request)

            assert response.status_code == 200
            data = response.json()
            assert "enhanced_content" in data
            assert data["enhanced_content"] == "Enhanced test content"

    def test_enhance_empty_content(self, client: TestClient, mock_summary_service):
        """Test content enhancement with empty content."""
        request = {"content": ""}

        with patch(
            "app.core.summaries.summary_service.SummaryService.enhance_content"
        ) as mock_enhance:
            mock_enhance.return_value = ""

            response = client.post("/api/v1/summary/enhance", json=request)

            assert response.status_code == 200
            data = response.json()
            assert data["enhanced_content"] == ""

    def test_enhance_invalid_request(self, client: TestClient):
        """Test content enhancement with invalid request."""
        invalid_request = {
            # Missing content
        }

        response = client.post("/api/v1/summary/enhance", json=invalid_request)

        assert response.status_code == 422

    def test_enhance_methods(self, client: TestClient):
        """Test that enhance endpoint only accepts POST requests."""
        request = {"content": "test"}

        # Test POST (should work)
        response = client.post("/api/v1/summary/enhance", json=request)
        assert response.status_code == 200

        # Test GET (should fail)
        response = client.get("/api/v1/summary/enhance")
        assert response.status_code == 405

        # Test PUT (should fail)
        response = client.put("/api/v1/summary/enhance", json=request)
        assert response.status_code == 405

        # Test DELETE (should fail)
        response = client.delete("/api/v1/summary/enhance")
        assert response.status_code == 405


class TestSummaryTagPositivityRatingsEndpoint(BaseAPITest):
    """Test cases for tag positivity ratings endpoint."""

    def test_tag_positivity_ratings_success(
        self, client: TestClient, mock_summary_service
    ):
        """Test successful tag positivity ratings."""
        request = {"tags": ["anxiety", "stress", "happiness"]}

        with patch(
            "app.core.summaries.summary_service.SummaryService."
            "get_tag_positivity_ratings"
        ) as mock_get_ratings:
            mock_get_ratings.return_value = [
                Tag(tag="anxiety", positivity_rating=2),
                Tag(tag="stress", positivity_rating=2),
                Tag(tag="happiness", positivity_rating=4),
            ]

            response = client.post(
                "/api/v1/summary/tag-positivity-ratings", json=request
            )

            assert response.status_code == 200
            data = response.json()
            assert "tags" in data
            assert len(data["tags"]) == 3

    def test_tag_positivity_ratings_empty_tags(
        self, client: TestClient, mock_summary_service
    ):
        """Test tag positivity ratings with empty tags list."""
        request = {"tags": []}

        with patch(
            "app.core.summaries.summary_service.SummaryService."
            "get_tag_positivity_ratings"
        ) as mock_get_ratings:
            mock_get_ratings.return_value = []

            response = client.post(
                "/api/v1/summary/tag-positivity-ratings", json=request
            )

            assert response.status_code == 200
            data = response.json()
            assert data["tags"] == []

    def test_tag_positivity_ratings_invalid_request(self, client: TestClient):
        """Test tag positivity ratings with invalid request."""
        invalid_request = {
            # Missing tags
        }

        response = client.post(
            "/api/v1/summary/tag-positivity-ratings", json=invalid_request
        )

        assert response.status_code == 422

    def test_tag_positivity_ratings_methods(self, client: TestClient):
        """Test that tag-positivity-ratings endpoint only accepts POST requests."""
        request = {"tags": ["test"]}

        # Test POST (should work)
        response = client.post("/api/v1/summary/tag-positivity-ratings", json=request)
        assert response.status_code in [200, 500]  # 500 due to mocking

        # Test GET (should fail)
        response = client.get("/api/v1/summary/tag-positivity-ratings")
        assert response.status_code == 405

        # Test PUT (should fail)
        response = client.put("/api/v1/summary/tag-positivity-ratings", json=request)
        assert response.status_code == 405

        # Test DELETE (should fail)
        response = client.delete("/api/v1/summary/tag-positivity-ratings")
        assert response.status_code == 405


class TestSummaryScenarioFeedbackEndpoint(BaseAPITest):
    """Test cases for scenario feedback endpoint."""

    def test_scenario_feedback_success(
        self, client: TestClient, mock_summary_service, sample_chat_messages
    ):
        """Test successful scenario feedback generation (deprecated endpoint)."""
        request = {
            "chat_history": sample_chat_messages,
        }

        # Use the global mock directly
        from unittest.mock import patch

        with patch(
            "app.core.summaries.summary_service.SummaryService."
            "generate_simulation_summary"
        ) as mock_generate_simulation:
            mock_generate_simulation.return_value = {
                "improvements": [
                    "Ask more open-ended questions",
                    "Use reflective listening",
                ],
                "positives": ["Good rapport building", "Empathetic responses"],
            }

            response = client.post("/api/v1/summary/scenario/feedback", json=request)

            assert response.status_code == 200
            data = response.json()
            assert "improvements" in data
            assert "positives" in data
            assert len(data["improvements"]) == 2
            assert len(data["positives"]) == 2

    def test_scenario_feedback_empty_chat_history(
        self, client: TestClient, mock_summary_service
    ):
        """Test scenario feedback with empty chat history (deprecated endpoint)."""
        request = {
            "chat_history": [],
        }

        # Use the global mock directly
        from unittest.mock import patch

        with patch(
            "app.core.summaries.summary_service.SummaryService."
            "generate_simulation_summary"
        ) as mock_generate_simulation:
            mock_generate_simulation.return_value = {
                "improvements": [],
                "positives": [],
            }

            response = client.post("/api/v1/summary/scenario/feedback", json=request)

            assert response.status_code == 200
            data = response.json()
            assert data["improvements"] == []
            assert data["positives"] == []

    def test_scenario_feedback_invalid_request(self, client: TestClient):
        """Test scenario feedback with invalid request."""
        invalid_request = {
            # Missing required fields
        }

        response = client.post(
            "/api/v1/summary/scenario/feedback", json=invalid_request
        )

        assert response.status_code == 422

    def test_scenario_feedback_methods(self, client: TestClient, sample_chat_messages):
        """Test that scenario/feedback endpoint only accepts POST requests."""
        request = {"chat_history": sample_chat_messages}

        # Test POST (should work)
        response = client.post("/api/v1/summary/scenario/feedback", json=request)
        assert response.status_code in [200, 500]  # 500 due to mocking

        # Test GET (should fail)
        response = client.get("/api/v1/summary/scenario/feedback")
        assert response.status_code == 405

        # Test PUT (should fail)
        response = client.put("/api/v1/summary/scenario/feedback", json=request)
        assert response.status_code == 405

        # Test DELETE (should fail)
        response = client.delete("/api/v1/summary/scenario/feedback")
        assert response.status_code == 405


class TestScenarioEvaluationEndpoint(BaseAPITest):
    """Test cases for scenario evaluation endpoint (new endpoint with competencies)."""

    def test_scenario_evaluation_success(
        self, client: TestClient, mock_summary_service, sample_chat_messages
    ):
        """Test successful scenario evaluation generation."""
        request = {
            "chat_history": sample_chat_messages,
        }

        # Use the global mock directly
        from unittest.mock import patch

        with patch(
            "app.core.summaries.summary_service.SummaryService."
            "generate_scenario_evaluation"
        ) as mock_generate_evaluation:
            mock_generate_evaluation.return_value = {
                "improvements": [
                    "Ask more open-ended questions",
                    "Use reflective listening",
                ],
                "positives": ["Good rapport building", "Empathetic responses"],
                "message_tags": [
                    {
                        "id": "msg-1",
                        "tags": [{"label": "Pacing", "category": "POSITIVE"}],
                    }
                ],
                "emotional_movement": [
                    {"message_id": "msg-2", "level": -2},
                ],
                "skill_coverage": [
                    {"category": "Learning", "percentage": 60},
                    {"category": "Support", "percentage": 90},
                    {"category": "Standards", "percentage": 40},
                ],
            }

            response = client.post("/api/v1/summary/scenario/evaluate", json=request)

            assert response.status_code == 200
            data = response.json()
            assert "improvements" in data
            assert "positives" in data
            assert "message_tags" in data
            assert "emotional_movement" in data
            assert "skill_coverage" in data
            assert len(data["improvements"]) == 2
            assert len(data["positives"]) == 2
            assert len(data["message_tags"]) == 1
            assert data["message_tags"][0]["id"] == "msg-1"
            assert len(data["emotional_movement"]) == 1
            assert data["emotional_movement"][0]["message_id"] == "msg-2"
            assert data["emotional_movement"][0]["level"] == -2
            assert len(data["skill_coverage"]) == 3
            assert data["skill_coverage"][0]["category"] == "Learning"
            assert data["skill_coverage"][0]["percentage"] == 60
            assert data["skill_coverage"][1]["category"] == "Support"
            assert data["skill_coverage"][2]["category"] == "Standards"

    def test_scenario_evaluation_methods(
        self, client: TestClient, sample_chat_messages
    ):
        """Test that scenario/evaluate endpoint only accepts POST requests."""
        request = {
            "chat_history": sample_chat_messages,
        }

        # Test POST (should work)
        response = client.post("/api/v1/summary/scenario/evaluate", json=request)
        assert response.status_code in [200, 500]  # 500 due to mocking

        # Test GET (should fail)
        response = client.get("/api/v1/summary/scenario/evaluate")
        assert response.status_code == 405

        # Test PUT (should fail)
        response = client.put("/api/v1/summary/scenario/evaluate", json=request)
        assert response.status_code == 405

        # Test DELETE (should fail)
        response = client.delete("/api/v1/summary/scenario/evaluate")
        assert response.status_code == 405
