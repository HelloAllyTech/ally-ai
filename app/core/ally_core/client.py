import httpx

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ally_core_client = None


class AllyCoreClient:
    @staticmethod
    def get_client() -> httpx.AsyncClient:
        global _ally_core_client

        if not _ally_core_client:
            raise Exception("Ally core client not initialised.")

        return _ally_core_client

    @staticmethod
    async def create_client() -> None:
        global _ally_core_client

        if not _ally_core_client:
            logger.info("Creating a new ally core httpx client...")
            _ally_core_client = httpx.AsyncClient(
                base_url=settings.ALLY_CORE.ENDPOINT,
                # The process-transcript callback delivers the full transcript +
                # summary and the receiver persists/encrypts it before replying;
                # under concurrent load that legitimately takes well over the old
                # 5s read budget, and a premature read-timeout used to be
                # misread as a failure. Give the read/write/pool a realistic
                # window; keep connect short to fail fast on a down host.
                timeout=httpx.Timeout(
                    connect=3.0,
                    read=60.0,
                    write=60.0,
                    pool=10.0,
                ),
                limits=httpx.Limits(
                    max_connections=settings.ALLY_CORE.MAX_CONNECTIONS,
                    max_keepalive_connections=settings.ALLY_CORE.MAX_KEEPALIVE_CONNECTIONS,
                ),
            )

    @staticmethod
    async def close(client: httpx.AsyncClient) -> None:
        logger.info("Closing ally core httpx client...")
        return await client.aclose()
