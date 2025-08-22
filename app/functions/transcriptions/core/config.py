from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Lambda function settings loaded from environment variables.

    This class defines the configuration settings for the transcription Lambda function,
    including AWS credentials, SQS queue URLs, and transcription provider settings.
    """

    # OpenAI Creds
    OPENAI_API_KEY: str = Field(...)
    OPENAI_ORGANIZATION_ID: str = Field(...)

    # Deepgram Creds
    DEEPGRAM_API_KEY: str = Field(...)

    # SQS Queue URLs
    TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL: str = Field(...)
    TRANSCRIPTION_RESULTS_QUEUE_URL: str = Field(...)

    # Transcription Provider
    TRANSCRIPTION_PROVIDER: str = Field(default="openai")


settings = Settings()
