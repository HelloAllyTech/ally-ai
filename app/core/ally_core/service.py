import httpx
from typing import Any, Dict, List

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AllyCoreService:

    def __init__(self, client: httpx.AsyncClient):
        self._client = client
        self._base_url = settings.ALLY_CORE.ENDPOINT
        self._api_key = settings.ALLY_CORE.API_KEY

    async def get_prompts_by_codes(self, codes: List[str]) -> Dict[str, str]:
        """Fetch current-version prompt text for the given prompt codes from
        ally-be's prompt management (GET /api/v1/prompts/by-codes).

        Returns a {promptCode: text} map; codes with no/empty content are simply
        absent. Used by the drift judge to source its rubric from the registry.
        """
        if not codes:
            return {}

        url = f"{self._base_url}/api/v1/prompts/by-codes"
        headers = {"x-api-key": self._api_key, "Accept": "application/json"}

        try:
            response = await self._client.get(
                url,
                headers=headers,
                params={"codes": ",".join(codes)},
            )
            response.raise_for_status()
            return response.json() or {}
        except httpx.HTTPStatusError as e:
            logger.error(
                f"AllyCore get_prompts_by_codes failed, "
                f"status={e.response.status_code}, body={e.response.text}, codes={codes}"
            )
            raise
        except httpx.RequestError as e:
            logger.error(
                f"AllyCore get_prompts_by_codes network error error={str(e)}, codes={codes}"
            )
            raise

    async def process_transcript(
        self,
        chat_id: int,
        transcription: List[Dict[str, Any]] | None = None,
        summary: Dict[str, Any] | None = None,
        error: str | None = None,
        stage: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        url = f"{self._base_url}/api/v1/chats/process-transcript"

        headers = {
            "x-api-key": self._api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        # Echo the trace id as a header too, so it is visible in access logs
        # even before the body is parsed.
        if correlation_id:
            headers["x-correlation-id"] = correlation_id

        # Required fields
        payload: dict[str, Any] = {
            "chatId": chat_id,
        }

        # Optional fields (only added if present)
        if transcription is not None:
            payload["transcription"] = transcription

        if summary is not None:
            payload["summary"] = summary

        if error is not None:
            payload["error"] = error

        if stage is not None:
            payload["stage"] = stage

        if correlation_id is not None:
            payload["correlationId"] = correlation_id

        is_error = error is not None
        try:
            response = await self._client.post(
                url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            logger.info(
                "AllyCore process-transcript ok status=%s chat_id=%s "
                "is_error=%s stage=%s correlation_id=%s",
                response.status_code,
                chat_id,
                is_error,
                stage,
                correlation_id,
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                f"AllyCore request failed, status={e.response.status_code}, "
                f"body={e.response.text}, chat_id={chat_id}, is_error={is_error}, "
                f"stage={stage}, correlation_id={correlation_id}"
            )
            raise

        except httpx.RequestError as e:
            logger.error(
                f"AllyCore network error error={str(e)}, chat_id={chat_id}, "
                f"is_error={is_error}, stage={stage}, correlation_id={correlation_id}",
            )
            raise
