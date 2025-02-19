from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Tuple

from dotenv import find_dotenv, dotenv_values
import os

# Load environment variables from .env
dotenv_path = find_dotenv()
env_vars = dotenv_values(dotenv_path)

os.environ.update(env_vars)

class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"

    # Server configuration
    SERVER_HOST: str = "localhost"
    SERVER_PORT: int = 8000

    # Accept CORS_URLS as either a tuple or a comma-separated string.
    CORS_URLS: Tuple[str, ...] = Field(default=())

    # API Version
    API_V1_STR: str = "/api/v1"

    @field_validator('CORS_URLS', mode='before')
    @classmethod
    def parse_cors_urls(cls, v):
        # If v is already a tuple or list, return it as a tuple.
        if isinstance(v, (tuple, list)):
            return tuple(url.strip().rstrip('/') for url in v if url.strip())
        # Otherwise, assume it's a comma-separated string.
        if isinstance(v, str):
            return tuple(url.strip().rstrip('/') for url in v.split(",") if url.strip())
        return v

    @field_validator('SERVER_PORT', mode='before')
    @classmethod
    def parse_str_to_int(cls, v):
        return int(v)


settings = Settings()