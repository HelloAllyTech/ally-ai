from app.core.constants import EmbeddingConstants, TextGenerationConstants
from app.core.embeddings.openai_embedding_client import OpenAIEmbeddingClient
from app.core.text_generations.openai_text_generation_client import OpenAITextGenerationClient


# Initialize OpenAI clients
def initialize_openai_clients():
    """
    Initialize the OpenAI clients as singletons.
    This should be called during application startup.
    """
    # Initialize the embedding client
    OpenAIEmbeddingClient.create_client(EmbeddingConstants.MODEL)

    # Initialize the text generation client with the default model
    OpenAITextGenerationClient.create_client(TextGenerationConstants.DEFAULT_MODEL)
