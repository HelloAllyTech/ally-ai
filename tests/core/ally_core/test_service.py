from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.ally_core import AllyCoreService

# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def mock_settings():
    """
    Mock application settings.
    """
    with patch("app.core.ally_core.service.settings") as settings:
        settings.ALLY_CORE.ENDPOINT = "https://ally-core.test"
        settings.ALLY_CORE.API_KEY = "test-api-key"
        yield settings


@pytest.fixture
def mock_client():
    """
    Mock httpx.AsyncClient.
    """
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def service(mock_client, mock_settings):
    """
    AllyCoreService instance with mocked dependencies.
    """
    return AllyCoreService(mock_client)


# -----------------------------
# Tests
# -----------------------------


@pytest.mark.asyncio
async def test_process_transcript_required_only(service, mock_client):
    mock_response = AsyncMock()
    mock_response.raise_for_status.return_value = None
    mock_client.post.return_value = mock_response

    await service.process_transcript(chat_id=123)

    mock_client.post.assert_awaited_once()
    args, kwargs = mock_client.post.call_args

    assert args[0] == "https://ally-core.test/api/v1/chats/process-transcript"

    assert kwargs["headers"] == {
        "x-api-key": "test-api-key",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    assert kwargs["json"] == {
        "chat_id": 123,
    }


@pytest.mark.asyncio
async def test_process_transcript_with_optional_fields(service, mock_client):
    mock_response = AsyncMock()
    mock_response.raise_for_status.return_value = None
    mock_client.post.return_value = mock_response

    await service.process_transcript(
        chat_id=1,
        download_presigned_url="https://download.url",
        delete_presigned_url="https://delete.url",
        error="some error",
    )

    _, kwargs = mock_client.post.call_args

    assert kwargs["json"] == {
        "chat_id": 1,
        "downloadPresignedUrl": "https://download.url",
        "deletePresignedUrl": "https://delete.url",
        "error": "some error",
    }


@pytest.mark.asyncio
async def test_process_transcript_http_error(service, mock_client):
    response = httpx.Response(
        status_code=400,
        content=b"Bad request",
        request=httpx.Request(
            "POST",
            "https://ally-core.test/api/v1/chats/process-transcript",
        ),
    )

    mock_client.post.side_effect = httpx.HTTPStatusError(
        "HTTP error",
        request=response.request,
        response=response,
    )

    with pytest.raises(httpx.HTTPStatusError):
        await service.process_transcript(chat_id=99)


@pytest.mark.asyncio
async def test_process_transcript_request_error(service, mock_client):
    mock_client.post.side_effect = httpx.RequestError(
        "Connection failed",
        request=httpx.Request(
            "POST",
            "https://ally-core.test/api/v1/chats/process-transcript",
        ),
    )

    with pytest.raises(httpx.RequestError):
        await service.process_transcript(chat_id=99)
