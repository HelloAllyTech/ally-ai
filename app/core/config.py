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

    # API Version
    API_V1_STR: str = "/api/v1"

    @field_validator('SERVER_PORT', mode='before')
    @classmethod
    def parse_str_to_int(cls, v):
        return int(v)


settings = Settings()
