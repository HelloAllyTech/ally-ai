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
    RATE_LIMIT: int = Field(...)
    WINDOW_SECONDS: int = Field(...)

class DeepgramSettings(BaseModel):
    API_KEY: str = Field(...)

class SarvamSettings(BaseModel):
    API_KEY: str = Field(...)

class TranscriptionSettings(BaseModel):
    PROVIDER: str = Field(...)

class LangSmithSettings(BaseModel):
    TRACING: str = Field(...)
    ENDPOINT: str = Field(...)
    API_KEY: str = Field(...)
    PROJECT: str = Field(...)


class AWSSettings(BaseModel):
    REGION: str = Field(...)
    ACCESS_KEY_ID: Optional[str] = None
    SECRET_ACCESS_KEY: Optional[str] = None
    ENDPOINT_URL: Optional[str] = None


class QueueSettings(BaseModel):
    TRANSCRIPTION_RESULTS_QUEUE_URL: str = Field(...)
    TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL: str = Field(...)


class ReferenceDocSettings(BaseModel):
    DISTANCE_THRESHOLD: float = Field(default=0.65)


class ApiSettings(BaseModel):

    X_API_KEY: str = Field(...)


class LLMSettings(BaseModel):
    MAX_CONCURRENT_LLM_CALLS: int = Field(...)


class AllyCoreSettings(BaseModel):
    ENDPOINT: str = Field("localhost")
    API_KEY: str = Field(...)


class HipaaAuditSettings(BaseModel):
    ENABLED: bool = Field(False)
    LOG_GROUP_NAME: str = Field(...)
    LOG_STREAM_NAME: str = Field(...)
    ENABLE_CONSOLE_LOGS: bool = Field(False)


# --------------------
# Root App Settings
# --------------------


class AppSettings(BaseSettings):
    """Root settings container that composes all sub-configs."""

    model_config = SettingsConfigDict(
        env_file=[".env", "./.env", "../.env"],
        extra="ignore",  # ignore extra vars
        env_nested_delimiter="__",
    )

    ENV: EnvSettings
    LOG: LogSettings
    SLACK_ALERTS: SlackSettings
    SERVER: ServerSettings
    WEAVIATE: WeaviateSettings
    OPENAI: OpenAISettings
    API: ApiSettings
    LANGSMITH: LangSmithSettings
    AWS: AWSSettings
    QUEUE: QueueSettings
    REFERENCE_DOCUMENTS_DISTANCE_THRESHOLD: float = 0.65
    LLM: LLMSettings
    HIPAA_AUDIT: HipaaAuditSettings
    ALLY_CORE: AllyCoreSettings
    DEEPGRAM: DeepgramSettings
    SARVAM: SarvamSettings
    TRANSCRIPTION: TranscriptionSettings

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
