from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env files.

    This class defines the configuration settings for the application, including log settings,
    server configuration, API version, Weaviate database settings, and OpenAI credentials.
    """
    model_config = SettingsConfigDict(env_file=[".env", "./.env", "../.env"])

    # Log settings
    LOG_LEVEL: str = "INFO"

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

    # OpenAI Creds
    OPENAI_API_KEY: str = Field(...)
    OPENAI_ORGANIZATION_ID: str = Field(...)

    @field_validator('SERVER_PORT', mode='before')
    @classmethod
    def parse_str_to_int(cls, v):
        """
        Parses a string value to an integer.
        """
        return int(v)

    @field_validator('WEAVIATE_HTTP_SECURE', 'WEAVIATE_GRPC_SECURE', mode='before')
    @classmethod
    def parse_str_to_bool(cls, v):
        """
        Parses a string value to a boolean.
        """
        return isinstance(v, str) and v.lower() == 'true'


settings = Settings()
