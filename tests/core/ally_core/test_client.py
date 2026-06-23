from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.ally_core import AllyCoreClient


@pytest.fixture(autouse=True)
def reset_global_client():
    """
    Ensure the global client is reset before each test.
    """
    from app.core.ally_core import client

    client._ally_core_client = None
    yield
    client._ally_core_client = None


@pytest.mark.asyncio
async def test_get_client_raises_if_not_initialized():
    with pytest.raises(Exception, match="Ally core client not initialised."):
        AllyCoreClient.get_client()


@pytest.mark.asyncio
async def test_create_client_creates_httpx_client():
    with patch("app.core.ally_core.client.settings") as mock_settings:
        mock_settings.ALLY_CORE.ENDPOINT = "https://ally-core.test"
        mock_settings.ALLY_CORE.MAX_CONNECTIONS = 100
        mock_settings.ALLY_CORE.MAX_KEEPALIVE_CONNECTIONS = 20

        await AllyCoreClient.create_client()
        client = AllyCoreClient.get_client()

        assert isinstance(client, httpx.AsyncClient)
        assert client.base_url == httpx.URL("https://ally-core.test")


@pytest.mark.asyncio
async def test_create_client_is_idempotent():
    """
    Calling create_client multiple times should not recreate the client.
    """
    with patch("app.core.ally_core.client.settings") as mock_settings:
        mock_settings.ALLY_CORE.ENDPOINT = "https://ally-core.test"
        mock_settings.ALLY_CORE.MAX_CONNECTIONS = 100
        mock_settings.ALLY_CORE.MAX_KEEPALIVE_CONNECTIONS = 20

        await AllyCoreClient.create_client()
        client1 = AllyCoreClient.get_client()

        await AllyCoreClient.create_client()
        client2 = AllyCoreClient.get_client()

        assert client1 is client2


@pytest.mark.asyncio
async def test_client_timeout_and_limits():
    with patch("app.core.ally_core.client.settings") as mock_settings:
        mock_settings.ALLY_CORE.ENDPOINT = "https://ally-core.test"
        mock_settings.ALLY_CORE.MAX_CONNECTIONS = 100
        mock_settings.ALLY_CORE.MAX_KEEPALIVE_CONNECTIONS = 20

        await AllyCoreClient.create_client()
        client = AllyCoreClient.get_client()

        timeout = client.timeout

        assert timeout.connect == 3.0
        # Read/write raised from 5s so the heavy process-transcript callback
        # isn't prematurely timed out (and mistaken for a failure) under load.
        assert timeout.read == 60.0
        assert timeout.write == 60.0
        assert timeout.pool == 10.0


@pytest.mark.asyncio
async def test_close_calls_aclose():
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    await AllyCoreClient.close(mock_client)

    mock_client.aclose.assert_awaited_once()
