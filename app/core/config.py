import os
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# --------------------
# Sub-configs
# --------------------

class EnvSettings(BaseModel):
    ENV: str = Field(...)


class LogSettings(BaseModel):
    LEVEL: str = Field("INFO")


class SlackSettings(BaseModel):
    ENABLED: bool = Field(False)
    API_TOKEN: str = Field(...)
    CHANNEL_ID: str = Field(...)
    LOG_LEVEL: str = Field("WARNING")


class ServerSettings(BaseModel):
    HOST: str = Field("localhost")
    PORT: int = Field(8000)



class WeaviateSettings(BaseModel):
    HTTP_HOST: str = Field(...)
    HTTP_PORT: int = Field(...)
    HTTP_SECURE: bool = Field(...)
    GRPC_HOST: str = Field(...)
    GRPC_PORT: int = Field(...)
    GRPC_SECURE: bool = Field(...)
    CONCURRENT_REQUESTS: int = Field(...)


class OpenAISettings(BaseModel):
    API_KEY: str = Field(...)
    ORGANIZATION_ID: str = Field(...)


class LangSmithSettings(BaseModel):
    TRACING: str = Field(...)
    ENDPOINT: str = Field(...)
    API_KEY: str = Field(...)
    PROJECT: str = Field(...)



class AWSSettings(BaseModel):
    ACCESS_KEY_ID: Optional[str] = None
    SECRET_ACCESS_KEY: Optional[str] = None
    REGION: Optional[str] = None
    ENDPOINT_URL: Optional[str] = None


class QueueSettings(BaseModel):
    TRANSCRIBE_AND_SUMMARIZE_RESULTS_BUCKET: str = Field(...)
    TRANSCRIPTION_RESULTS_QUEUE_URL: str = Field(...)
    TRANSCRIBE_AND_SUMMARIZE_RESPONSE_QUEUE_URL: str = Field(...)


class ReferenceDocSettings(BaseModel):
    DISTANCE_THRESHOLD: float = Field(default=0.65)


# --------------------
# Root App Settings
# --------------------

class AppSettings(BaseSettings):
    """Root settings container that composes all sub-configs."""

    model_config = SettingsConfigDict(
        env_file=[".env", "./.env", "../.env"],
        extra="forbid",  # fail on extra vars
        env_nested_delimiter="__",
    )


    ENV: EnvSettings
    LOG: LogSettings
    SLACK_ALERTS: SlackSettings
    SERVER: ServerSettings
    WEAVIATE: WeaviateSettings
    OPENAI: OpenAISettings
    LANGSMITH: LangSmithSettings
    AWS: AWSSettings
    QUEUE: QueueSettings
    REFERENCE_DOCS: ReferenceDocSettings

    def model_post_init(self, __context=None) -> None:
        """
        After initialization, automatically propagate LangSmith values to os.environ.
        """
        os.environ["LANGSMITH_TRACING"] = self.LANGSMITH.TRACING
        os.environ["LANGSMITH_ENDPOINT"] = self.LANGSMITH.ENDPOINT
        os.environ["LANGSMITH_API_KEY"] = self.LANGSMITH.API_KEY
        os.environ["LANGSMITH_PROJECT"] = self.LANGSMITH.PROJECT



# --------------------
# Global Settings Singleton
# --------------------

settings = AppSettings()