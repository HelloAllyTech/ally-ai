from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global OpenAI embedding client
_openai_embedding_client = None


class OpenAIEmbeddingClient:
    @staticmethod
    def get_client() -> OpenAIEmbeddings:
        """
        Get the singleton instance of the OpenAI embedding client.

        Returns:
            OpenAIEmbeddings: The OpenAI embedding client.
        """
        global _openai_embedding_client

        if not _openai_embedding_client:
            logger.error(
                "OpenAI embedding client has not been created. Please create a "
                "client first."
            )
            raise Exception(
                "OpenAI embedding client has not been created. Please create a "
                "client first."
            )

        return _openai_embedding_client

    @staticmethod
    def create_client(model: str) -> None:
        """
        Create a singleton instance of the OpenAI embedding client.

        Parameters:
            model (str): The name of the model to use.
        """
        global _openai_embedding_client

        if not _openai_embedding_client:
            logger.info(f"Creating a new OpenAI embedding client with model {model}...")
            _openai_embedding_client = OpenAIEmbeddings(
                model=model,
                api_key=settings.OPENAI.API_KEY,
                organization=settings.OPENAI.ORGANIZATION_ID,
            )
        else:
            logger.warning(
                "OpenAI embedding client already exists. Reusing the existing client."
            )
