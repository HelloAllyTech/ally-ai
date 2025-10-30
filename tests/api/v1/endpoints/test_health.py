"""
Tests for health endpoint.
"""

from fastapi.testclient import TestClient

from tests.api.v1.endpoints.base import BaseAPITest


class TestHealthEndpoint(BaseAPITest):
    """Test cases for health endpoint."""

    def test_health_check_success(self, client: TestClient):
        """Test successful health check."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_check_response_format(self, client: TestClient):
        """Test health check response format."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_check_methods(self, client: TestClient):
        """Test that health endpoint only accepts GET requests."""
        # Test GET (should work)
        response = client.get("/api/health")
        assert response.status_code == 200

        # Test POST (should fail)
        response = client.post("/api/health")
        assert response.status_code == 405

        # Test PUT (should fail)
        response = client.put("/api/health")
        assert response.status_code == 405

        # Test DELETE (should fail)
        response = client.delete("/api/health")
        assert response.status_code == 405

    def test_health_check_content_type(self, client: TestClient):
        """Test health check content type."""
        response = client.get("/api/health")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_health_check_multiple_requests(self, client: TestClient):
        """Test multiple health check requests."""
        for _ in range(5):
            response = client.get("/api/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
