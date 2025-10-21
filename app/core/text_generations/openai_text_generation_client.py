from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global OpenAI chat client
_openai_chat_client = None


class OpenAITextGenerationClient:
    @staticmethod
    def get_client() -> ChatOpenAI:
        """
        Get the singleton instance of the OpenAI chat client.

        Returns:
            ChatOpenAI: The OpenAI chat client.
        """
        global _openai_chat_client  # noqa: F824

        if not _openai_chat_client:
            logger.error(
                "OpenAI chat client has not been created. Please create a client first."
            )
            raise Exception(
                "OpenAI chat client has not been created. Please create a client first."
            )

        return _openai_chat_client

    @staticmethod
    def create_client(model_name: str) -> None:
        """
        Create a singleton instance of the OpenAI chat client.

        Parameters:
            model_name (str): The name of the model to use.
        """
        global _openai_chat_client  # noqa: F824

        if not _openai_chat_client:
            logger.info(f"Creating a new OpenAI chat client with model {model_name}...")
            _openai_chat_client = ChatOpenAI(
                model=model_name,
                api_key=settings.OPENAI.API_KEY,
                organization=settings.OPENAI.ORGANIZATION_ID,
            )
        else:
            logger.warning(
                "OpenAI chat client already exists. Reusing the existing client."
            )
