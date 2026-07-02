from unittest.mock import AsyncMock, Mock, patch

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
    mock_response = Mock()
    mock_response.raise_for_status = Mock(return_value=None)
    mock_client.post.return_value = mock_response

    await service.process_transcript(chat_id=123, transcription=None, summary=None)

    mock_client.post.assert_awaited_once()
    args, kwargs = mock_client.post.call_args

    assert args[0] == "https://ally-core.test/api/v1/chats/process-transcript"

    assert kwargs["headers"] == {
        "x-api-key": "test-api-key",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    assert kwargs["json"] == {
        "chatId": 123,
    }


@pytest.mark.asyncio
async def test_process_transcript_with_optional_fields(service, mock_client):
    mock_response = Mock()
    mock_response.raise_for_status = Mock(return_value=None)
    mock_client.post.return_value = mock_response

    await service.process_transcript(
        chat_id=1,
        transcription=[
            {"speaker": "client", "text": "hello"},
            {"speaker": "coach", "text": "hi"},
        ],
        summary={"topics": ["rapport"], "score": 0.9},
        error="some error",
    )

    _, kwargs = mock_client.post.call_args

    assert kwargs["json"] == {
        "chatId": 1,
        "transcription": [
            {"speaker": "client", "text": "hello"},
            {"speaker": "coach", "text": "hi"},
        ],
        "summary": {"topics": ["rapport"], "score": 0.9},
        "error": "some error",
    }


@pytest.mark.asyncio
async def test_process_transcript_includes_attempt_analytics_fields(
    service, mock_client
):
    """The Phase-B STT/model/phase signals are forwarded with matching keys."""
    mock_response = Mock()
    mock_response.raise_for_status = Mock(return_value=None)
    mock_client.post.return_value = mock_response

    await service.process_transcript(
        chat_id=7,
        transcription=[{"speaker": "client", "text": "hi"}],
        summary={"session_summary": "ok"},
        stt_provider_succeeded="openai",
        stt_attempts=[
            {"provider": "deepgram", "ok": False, "error": "empty"},
            {"provider": "openai", "ok": True},
        ],
        summary_model="gpt-4o-mini-2024-07-18",
        phase_reached="delivered",
    )

    _, kwargs = mock_client.post.call_args
    body = kwargs["json"]
    assert body["sttProviderSucceeded"] == "openai"
    assert body["sttAttempts"] == [
        {"provider": "deepgram", "ok": False, "error": "empty"},
        {"provider": "openai", "ok": True},
    ]
    assert body["summaryModel"] == "gpt-4o-mini-2024-07-18"
    assert body["phaseReached"] == "delivered"


@pytest.mark.asyncio
async def test_process_transcript_omits_attempt_fields_when_absent(
    service, mock_client
):
    """Absent signals must not appear in the payload (backward-compatible)."""
    mock_response = Mock()
    mock_response.raise_for_status = Mock(return_value=None)
    mock_client.post.return_value = mock_response

    await service.process_transcript(chat_id=8, transcription=None, summary=None)

    _, kwargs = mock_client.post.call_args
    body = kwargs["json"]
    assert "sttProviderSucceeded" not in body
    assert "sttAttempts" not in body
    assert "summaryModel" not in body
    assert "phaseReached" not in body


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
        await service.process_transcript(chat_id=99, transcription=None, summary=None)


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
        await service.process_transcript(chat_id=99, transcription=None, summary=None)
