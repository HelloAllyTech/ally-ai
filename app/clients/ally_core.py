# app/clients/ally_core.py

import httpx

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AllyCoreClient:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client
        self._base_url = settings.ALLY_CORE.ENDPOINT
        self._api_key = settings.ALLY_CORE.API_KEY

    async def process_transcript(
        self,
        chat_id: int,
        download_presigned_url: str | None = None,
        delete_presigned_url: str | None = None,
        error: str | None = None,
    ) -> None:
        url = f"{self._base_url}/api/v1/chats/process-transcript"

        headers = {
            "x-api-key": self._api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Required fields
        payload: dict[str, str | int] = {
            "chat_id": chat_id,
        }

        # Optional fields (only added if present)
        if download_presigned_url is not None:
            payload["downloadPresignedUrl"] = download_presigned_url

        if delete_presigned_url is not None:
            payload["deletePresignedUrl"] = delete_presigned_url

        if error is not None:
            payload["error"] = error

        try:
            response = await self._client.post(
                url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            logger.error(
                "AllyCore request failed",
                status=e.response.status_code,
                body=e.response.text,
                chat_id=chat_id,
            )
            raise

        except httpx.RequestError as e:
            logger.error(
                "AllyCore network error",
                error=str(e),
                chat_id=chat_id,
            )
            raise
