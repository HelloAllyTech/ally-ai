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

class GeminiSettings(BaseModel):
    # Optional so the service still boots in envs that haven't set GEMINI__API_KEY
    # yet; the drift judge raises a clear error if invoked without a key.
    API_KEY: Optional[str] = None


class DriftJudgeSettings(BaseModel):
    """Conversation drift judge (see drift-metrics-spec.md). Gemini for now."""

    MODEL: str = Field("gemini-2.5-pro")
    # Bump when the judge rubric changes; reported back to the caller (ally-be)
    # and stored on each judgment row so a re-judge with a new rubric coexists
    # with prior runs. This service is a stateless judge — it owns no database.
    PROMPT_VERSION: str = Field("v1")


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


class LLMUsageSettings(BaseModel):
    """Token-usage emission for the Core token-consumption dashboard.

    No-ops unless QUEUE_URL is set (should point at the queue ally-be's
    LearnMessageAndEventConsumer listens on — Core routes by message_type)."""

    ENABLED: bool = Field(default=True)
    QUEUE_URL: str = Field(default="")
    COUNT_EMBEDDING_TOKENS: bool = Field(default=False)


class ReferenceDocSettings(BaseModel):
    DISTANCE_THRESHOLD: float = Field(default=0.65)


class ApiSettings(BaseModel):

    X_API_KEY: str = Field(...)


class LLMSettings(BaseModel):
    MAX_CONCURRENT_LLM_CALLS: int = Field(...)


class AllyCoreSettings(BaseModel):
    ENDPOINT: str = Field("localhost")
    API_KEY: str = Field(...)
    MAX_CONNECTIONS: int = Field(100)
    MAX_KEEPALIVE_CONNECTIONS: int = Field(20)


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
    GEMINI: GeminiSettings = Field(default_factory=GeminiSettings)
    DRIFT_JUDGE: DriftJudgeSettings = Field(default_factory=DriftJudgeSettings)
    LLM_USAGE: LLMUsageSettings = Field(default_factory=LLMUsageSettings)

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
