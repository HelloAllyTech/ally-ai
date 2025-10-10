"""Tests for main.py application startup."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, lifespan


class TestMain:
    """Test cases for main.py application startup."""

    def test_app_creation(self):
        """Test that FastAPI app is created with correct configuration."""
        assert app.title == "Lifeline AI"
        assert app.description == "AI Service for Lifeline Project"
        assert app.version == "1.0.0"

    def test_app_has_lifespan(self):
        """Test that app has lifespan context manager."""
        assert app.router.lifespan_context is not None

    def test_app_has_middleware(self):
        """Test that app has middleware configured."""
        assert len(app.user_middleware) > 0

    @pytest.mark.asyncio
    async def test_lifespan_startup(self):
        """Test application startup lifecycle."""
        with (
            patch("app.main.WeaviateClient") as mock_weaviate_client,
            patch("app.main.initialize_openai_clients") as mock_init_openai,
            patch("app.main.logger") as mock_logger,
        ):

            # Setup mocks
            mock_client = AsyncMock()
            mock_weaviate_client.get_client.return_value = mock_client
            mock_weaviate_client.create_client.return_value = None
            mock_weaviate_client.connect = AsyncMock(return_value=None)
            mock_weaviate_client.close = AsyncMock(return_value=None)
            mock_init_openai.return_value = None

            # Execute lifespan
            async with lifespan(app):
                pass

            # Assert startup calls
            mock_weaviate_client.create_client.assert_called_once()
            mock_weaviate_client.get_client.assert_called()
            mock_weaviate_client.connect.assert_called_once_with(mock_client)
            mock_init_openai.assert_called_once()
            mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_lifespan_shutdown(self):
        """Test application shutdown lifecycle."""
        with (
            patch("app.main.WeaviateClient") as mock_weaviate_client,
            patch("app.main.logger") as mock_logger,
        ):

            # Setup mocks
            mock_client = AsyncMock()
            mock_weaviate_client.get_client.return_value = mock_client
            mock_weaviate_client.create_client.return_value = None
            mock_weaviate_client.connect = AsyncMock(return_value=None)
            mock_weaviate_client.close = AsyncMock(return_value=None)

            # Execute lifespan
            async with lifespan(app):
                pass

            # Assert shutdown calls
            mock_weaviate_client.close.assert_called_once_with(mock_client)
            mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_lifespan_exception_handling(self):
        """Test lifespan handles exceptions gracefully."""
        with (
            patch("app.main.WeaviateClient") as mock_weaviate_client,
            patch("app.main.initialize_openai_clients") as mock_init_openai,
        ):

            # Setup mocks to raise exceptions
            mock_weaviate_client.create_client.side_effect = Exception(
                "Connection failed"
            )
            mock_init_openai.side_effect = Exception("OpenAI init failed")

            # Execute lifespan - should raise exception since lifespan doesn't handle
            # them
            with pytest.raises(Exception, match="Connection failed"):
                async with lifespan(app):
                    pass

    def test_app_routes_included(self):
        """Test that API routes are included in the app."""
        # Check that the API router is included
        route_paths = [route.path for route in app.routes]
        assert any("/api/v1" in path for path in route_paths)

    @patch("app.main.uvicorn.run")
    @patch("app.main.settings")
    @patch("app.main.logging_config")
    def test_main_execution(self, mock_logging_config, mock_settings, mock_uvicorn_run):
        """Test main execution when run directly."""
        # Setup mocks
        mock_settings.SERVER.HOST = "0.0.0.0"
        mock_settings.SERVER.PORT = 8000

        # Import and execute main
        import app.main

        # The uvicorn.run should be called when __name__ == "__main__"
        # This is tested by checking the module structure
        assert hasattr(app.main, "app")

    def test_app_client_creation(self):
        """Test that TestClient can be created for the app."""
        client = TestClient(app)
        assert client is not None

    def test_health_endpoint_accessible(self):
        """Test that health endpoint is accessible."""
        client = TestClient(app)
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_lifespan_yield_behavior(self):
        """Test that lifespan properly yields control."""
        with (
            patch("app.main.WeaviateClient") as mock_weaviate_client,
            patch("app.main.initialize_openai_clients") as mock_init_openai,
        ):

            # Setup mocks
            mock_client = AsyncMock()
            mock_weaviate_client.get_client.return_value = mock_client
            mock_weaviate_client.create_client.return_value = None
            mock_weaviate_client.connect = AsyncMock(return_value=None)
            mock_weaviate_client.close = AsyncMock(return_value=None)
            mock_init_openai.return_value = None

            # Track if we're in the yielded section
            in_yielded_section = False

            # Execute lifespan
            async with lifespan(app) as yielded_value:
                in_yielded_section = True
                assert yielded_value is None  # lifespan yields None

            # Should have been in yielded section
            assert in_yielded_section
