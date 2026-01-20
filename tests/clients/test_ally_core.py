import json
import pytest
import httpx
from unittest.mock import AsyncMock

from app.clients.ally_core import AllyCoreClient


class TestAllyCoreClient:

    @pytest.fixture
    def mock_client(self):
        client = AsyncMock(spec=httpx.AsyncClient)

        # Mock response returned by client.post()
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None

        client.post.return_value = mock_response
        return client

    @pytest.mark.asyncio
    async def test_process_transcript_required_only(self, mock_client, monkeypatch):

        ally_client = AllyCoreClient(mock_client)

        await ally_client.process_transcript(chat_id=123)

        # Assert client.post was called once
        mock_client.post.assert_awaited_once()

        # Extract call arguments
        _, kwargs = mock_client.post.call_args

        assert kwargs["json"] == {
            "chat_id": 123
        }

        assert kwargs["headers"]["x-api-key"] == "test-api-key"
        assert kwargs["headers"]["content-type"] == "application/json"
        assert kwargs["headers"]["accept"] == "application/json"
