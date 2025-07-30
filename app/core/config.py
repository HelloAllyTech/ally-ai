import os
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env files.

    This class defines the configuration settings for the application, including log settings,
    server configuration, API version, Weaviate database settings, and OpenAI credentials.
    """
    model_config = SettingsConfigDict(env_file=[".env", "./.env", "../.env"])

    # ENV
    ENV: str = Field(...)

    # Log settings
    LOG_LEVEL: str = "INFO"
    SLACK_ALERTS_ENABLED: bool = False
    SLACK_ALERTS_API_TOKEN: str = Field(...)
    SLACK_ALERTS_CHANNEL_ID: str = Field(...)
    SLACK_ALERTS_LOG_LEVEL: str = "WARNING"

    # Server configuration
    SERVER_HOST: str = "localhost"
    SERVER_PORT: int = 8000

    # API Version
    API_V1_STR: str = "/api/v1"

    # Weaviate
    WEAVIATE_HTTP_HOST: str = Field(...)
    WEAVIATE_HTTP_PORT: int = Field(...)
    WEAVIATE_HTTP_SECURE: bool = Field(...)
    WEAVIATE_GRPC_HOST: str = Field(...)
    WEAVIATE_GRPC_PORT: int = Field(...)
    WEAVIATE_GRPC_SECURE: bool = Field(...)
    WEAVIATE_CONCURRENT_REQUESTS: int = Field(...)

    # OpenAI Creds
    OPENAI_API_KEY: str = Field(...)
    OPENAI_ORGANIZATION_ID: str = Field(...)

    # Deepgram Creds
    DEEPGRAM_API_KEY: str = Field(...)

    # LangSmith Creds
    LANGSMITH_TRACING: str = Field(...)
    LANGSMITH_ENDPOINT: str = Field(...)
    LANGSMITH_API_KEY: str = Field(...)
    LANGSMITH_PROJECT: str = Field(...)

    # SQS Configs
    AWS_ACCESS_KEY_ID: str = Field(...)
    AWS_SECRET_ACCESS_KEY: str = Field(...)
    AWS_REGION: str = Field(...)
    AWS_ENDPOINT_URL: str = Field(...)

    # S3 Configuration
    S3_TRANSCRIBE_AND_SUMMARIZE_RESULTS_BUCKET: str = Field(...)

    # SQS Queue URLs
    TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL: str = Field(...)
    TRANSCRIBE_AND_SUMMARIZE_RESULTS_QUEUE_URL: str = Field(...)

     # Reference document settings
    REFERENCE_DOCUMENTS_DISTANCE_THRESHOLD: float = Field(default=0.65)
    
    @field_validator('SERVER_PORT', mode='before')
    @classmethod
    def parse_str_to_int(cls, v):
        """
        Parses a string value to an integer.
        """
        return int(v)

    @field_validator('WEAVIATE_HTTP_SECURE', 'WEAVIATE_GRPC_SECURE', 'SLACK_ALERTS_ENABLED', mode='before')
    @classmethod
    def parse_str_to_bool(cls, v):
        """
        Parses a string value to a boolean.
        """
        return isinstance(v, str) and v.lower() == 'true'

    def model_post_init(self, __context=None) -> None:
        """
        After initialization, add certain values to os.environ.
        You can also propagate other settings if needed.
        """
        os.environ['LANGSMITH_TRACING'] = self.LANGSMITH_TRACING
        os.environ['LANGSMITH_ENDPOINT'] = self.LANGSMITH_ENDPOINT
        os.environ['LANGSMITH_API_KEY'] = self.LANGSMITH_API_KEY
        os.environ['LANGSMITH_PROJECT'] = self.LANGSMITH_PROJECT


settings = Settings()
