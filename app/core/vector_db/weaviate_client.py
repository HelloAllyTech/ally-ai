import weaviate
from weaviate import WeaviateAsyncClient

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global Weaviate client
_weaviate_client = None


class WeaviateClient:
    @staticmethod
    def get_client() -> WeaviateAsyncClient:
        global _weaviate_client

        if not _weaviate_client:
            logger.error("Weaviate client has not been created. Please create a client first.")
            raise Exception("Weaviate client has not been created. Please create a client first.")

        return _weaviate_client

    @staticmethod
    def create_client() -> None:
        global _weaviate_client

        if not _weaviate_client:
            logger.info("Creating a new Weaviate client...")
            _weaviate_client = weaviate.use_async_with_custom(
                http_host=settings.WEAVIATE_HTTP_HOST,
                http_port=settings.WEAVIATE_HTTP_PORT,
                http_secure=settings.WEAVIATE_HTTP_SECURE,
                grpc_host=settings.WEAVIATE_GRPC_HOST,
                grpc_port=settings.WEAVIATE_GRPC_PORT,
                grpc_secure=settings.WEAVIATE_GRPC_SECURE,
            )
        else:
            logger.warning("Weaviate client already exists. Reusing the existing client.")

    @staticmethod
    async def connect(client: WeaviateAsyncClient) -> None:
        logger.info("Connecting to Weaviate client...")
        await client.connect()

        if await client.is_live():
            logger.info("Weaviate client is live.")
        else:
            logger.error("Weaviate client is not live.")
            # raise WeaviateConnectionError("Weaviate client is not live.")

    @staticmethod
    async def close(client: WeaviateAsyncClient) -> None:
        logger.info("Closing Weaviate client...")
        return await client.close()
