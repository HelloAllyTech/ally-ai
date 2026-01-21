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
            raise Exception("All core client not initialised.")

        return _ally_core_client

    @staticmethod
    async def create_client() -> None:
        global _ally_core_client

        if not _ally_core_client:
            logger.info("Creating a new ally core httpx client...")
            _ally_core_client = httpx.AsyncClient(
                base_url=settings.ALLY_CORE.ENDPOINT,
                timeout=httpx.Timeout(
                    connect=3.0,
                    read=5.0,
                    write=5.0,
                    pool=5.0,
                ),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                ),
            )

    @staticmethod
    async def close(client: httpx.AsyncClient) -> None:
        logger.info("Closing ally core httpx client...")
        return await client.aclose()
