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


class AnthropicSettings(BaseModel):
    # Optional so the service still boots in envs that haven't set ANTHROPIC__API_KEY
    # yet, matching GeminiSettings. The knowledge agent raises a clear error if a Claude
    # model is selected without a key, rather than silently answering on a different
    # model than the admin chose.
    API_KEY: Optional[str] = None


class KnowledgeAgentSettings(BaseModel):
    """WhatsApp Q&A bot's retrieval-augmented answering agent.

    Thresholds are defaults only — ally-be overrides them per request from the
    `whatsapp_bot` global settings row, so retrieval can be tuned without a deploy.

    MIN_SIMILARITY and DECLINE_SIMILARITY are deliberately different numbers.
    The first is a permissive retrieval floor; the second is the actual decision
    about whether the corpus covers the question. A relevant passage matched
    against a short paraphrased question scores roughly 0.40-0.60 with
    text-embedding-3-small, so a single hard floor at the decision value would
    decline constantly on legitimate rephrasings. Retrieving down to 0.35 and
    gating separately lets the model adjudicate the middle band, which is what
    it is actually good at.
    """

    # Default provider/model for the answer call. Claude is the default because this
    # prompt's hardest requirement is refusing to answer from outside the retrieved
    # passages, which is an instruction-following problem. Overridable per prompt from
    # ally-be's prompt management.
    DEFAULT_PROVIDER: str = Field("anthropic")
    DEFAULT_MODEL: str = Field("claude-sonnet-4-6")
    # The translate step is a mechanical transform, so it runs on a cheaper model by
    # default.
    TRANSLATE_MODEL: str = Field("claude-haiku-4-5-20251001")
    # The crisis classifier also runs on the cheap model, but for a different reason: it
    # runs CONCURRENTLY with the answer call on every question, so its latency is hidden
    # but its cost is not. A small model is adequate — the prompt asks for one binary
    # judgement with an explicit instruction to prefer the false positive.
    CRISIS_MODEL: str = Field("claude-haiku-4-5-20251001")
    TOP_K: int = Field(8)
    MIN_SIMILARITY: float = Field(0.35)
    DECLINE_SIMILARITY: float = Field(0.42)
    # Passages within this similarity of the best hit are kept; the rest are dropped
    # even if they cleared the floor. Stops one strong match from being diluted by seven
    # weak ones.
    SIMILARITY_BAND: float = Field(0.08)
    MAX_PASSAGES: int = Field(5)
    MAX_CONTEXT_TOKENS: int = Field(3000)
    # Bump when the answer prompt changes in a way that could move answers; echoed back
    # to ally-be and stored per message so a behaviour change can be traced to its
    # instructions.
    PROMPT_VERSION: str = Field("v1")


class DriftJudgeSettings(BaseModel):
    """Conversation drift judge (see drift-metrics-spec.md). Gemini for now."""

    MODEL: str = Field("gemini-2.5-pro")
    # Bump when the judge rubric changes; reported back to the caller (ally-be)
    # and stored on each judgment row so a re-judge with a new rubric coexists
    # with prior runs. This service is a stateless judge — it owns no database.
    #
    # v2 (Weak Performing Metrics): adds the clienthood labels
    # (role_inversion, offered_solution, solutions_offered, resistance_briefed)
    # and the progression labels (introduced_new_information,
    # stuck_is_appropriate). v1 rows are NOT migrated — they simply carry nulls
    # in the new columns, and the dashboard pins to the newest version that has
    # rows, so the two never average together.
    PROMPT_VERSION: str = Field("v2")


class LanguageJudgeSettings(BaseModel):
    """Language-quality judge (see language-eval-judge-schema.md). Sibling of
    the drift judge; separate call, separate rubric version. Comparisons are
    only valid within one (MODEL, PROMPT_VERSION) pair.

    v2 widened dialect_lexicon from regional-variety-only to lexical
    correctness (adding nonexistent_word and wrong_sense). The old scope had no
    home for "that is not a Tamil word" or "that word means something else
    here" — which is what partner organisations were reporting — so the
    dimension read as near-zero while being their blocking issue. v1 rows keep
    the narrow meaning and are never averaged with v2.
    """

    MODEL: str = Field("gemini-2.5-pro")
    # Bump when the judge rubric or typology changes; reported back to the
    # caller (ally-be) and stored on each annotation row so a re-judge with a
    # new rubric coexists with prior runs. Stateless — owns no database.
    PROMPT_VERSION: str = Field("v2")


class FeedbackGroundednessJudgeSettings(BaseModel):
    """Feedback-groundedness judge — is post-session feedback true of the
    transcript? Sibling of the drift and language judges: separate call,
    separate rubric version, comparisons valid only within one
    (MODEL, PROMPT_VERSION) pair."""

    MODEL: str = Field("gemini-2.5-pro")
    PROMPT_VERSION: str = Field("v1")


class AnalyticsAgentSettings(BaseModel):
    """Analytics Agent (admin Analytics -> Analytics Agent tab).

    Two calls per question: a planner that sees the schema catalogue and writes
    SQL, and a narrator that sees the result rows and writes the answer. The
    planner is the harder job — a wrong column choice produces a confident wrong
    number — so it may run on a stronger model than the narrator.
    """

    PLANNER_MODEL: str = Field("gemini-2.5-pro")
    ANSWER_MODEL: str = Field("gemini-2.5-flash")
    # Bump when either prompt changes in a way that could move answers; echoed
    # back to ally-be so a surprising answer can be traced to its instructions.
    PROMPT_VERSION: str = Field("v1")


class DeepgramSettings(BaseModel):
    API_KEY: str = Field(...)


class SarvamSettings(BaseModel):
    API_KEY: str = Field(...)


class TranscriptionSettings(BaseModel):
    # Primary provider (backwards compatible). Used as the sole provider when
    # PROVIDERS is not set.
    PROVIDER: str = Field(...)
    # Ordered fallback chain. The worker tries these in order, failing over on
    # error or an empty transcript. Comma-separated so it maps cleanly from an
    # env var (TRANSCRIPTION__PROVIDERS). Providers with a missing or
    # placeholder API key are skipped at startup, so this default safely
    # degrades: with only Deepgram + OpenAI keys set it resolves to
    # deepgram -> openai, and Sarvam slots back in automatically once a real
    # SARVAM key is configured. Set to a single name to disable fallback.
    PROVIDERS: Optional[str] = Field(default="deepgram,sarvam,openai")
    # Optional hard cap (seconds) on a single provider attempt inside the
    # fallback chain. The SUM of attempts must stay under the SQS visibility
    # timeout (900s). Leave unset to rely on each provider's own timeout.
    PER_PROVIDER_TIMEOUT_SECONDS: Optional[int] = Field(default=None)
    # When True, the PRIMARY provider is chosen per session (rotate the chain by
    # chat_id) so traffic spreads evenly across providers and per-provider
    # failure rates become comparable. The fallback chain is preserved, so
    # reliability is unchanged — only which provider leads. Default off; flip on
    # to run the STT-provider comparison, off to revert to the fixed order.
    RATION_PROVIDERS: bool = Field(default=False)


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
    ANTHROPIC: AnthropicSettings = Field(default_factory=AnthropicSettings)
    KNOWLEDGE_AGENT: KnowledgeAgentSettings = Field(
        default_factory=KnowledgeAgentSettings
    )
    DRIFT_JUDGE: DriftJudgeSettings = Field(default_factory=DriftJudgeSettings)
    LANGUAGE_JUDGE: LanguageJudgeSettings = Field(default_factory=LanguageJudgeSettings)
    ANALYTICS_AGENT: AnalyticsAgentSettings = Field(
        default_factory=AnalyticsAgentSettings
    )
    LLM_USAGE: LLMUsageSettings = Field(default_factory=LLMUsageSettings)
    FEEDBACK_GROUNDEDNESS_JUDGE: FeedbackGroundednessJudgeSettings = Field(
        default_factory=FeedbackGroundednessJudgeSettings
    )

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
