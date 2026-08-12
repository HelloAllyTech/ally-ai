from enum import Enum
from typing import Final

from pydantic import BaseModel, Field


class EmbeddingConstants:
    MODEL: Final[str] = "text-embedding-3-small"

    # Texts per OpenAI embeddings request. The API caps a request at 2048 inputs and
    # 300k tokens, so the binding limit here is tokens, not inputs: a knowledge-base
    # chunk targets ~400 tokens, making 64 inputs roughly 26k tokens — an order of
    # magnitude under the ceiling, which leaves headroom for a batch of unusually long
    # chunks without a size-based rejection.
    #
    # Batching at all is load-bearing, not tidiness: a 300-page PDF is ~500 chunks, and
    # embedding those in one call would be a single ~200k-token request that fails as a
    # unit. Whole-batch failure is the worst outcome for the caller, which tracks
    # progress per chunk.
    BATCH_SIZE: Final[int] = 64

    # Retries per batch on a rate limit or a connection error, which are the two
    # failures that are reliably transient. Deliberately small: the caller (ally-be's
    # ingest consumer) has its own retry via SQS redelivery, so the job here is to ride
    # out a brief 429, not to become a second retry system layered under one that
    # already exists.
    MAX_RETRIES: Final[int] = 3

    # Seconds before the first retry; doubles each attempt (0.5s, 1s, 2s). The total
    # added delay stays around 3.5s so a batch cannot sit long enough to threaten the
    # ingest consumer's SQS visibility timeout.
    RETRY_BASE_DELAY_SECONDS: Final[float] = 0.5


class TextGenerationConstants:
    DEFAULT_MODEL: Final[str] = "gpt-4o-mini-2024-07-18"


class AgeRange(str, Enum):
    BELOW_FIVE = "0-5"
    SIX_TO_TWELVE = "6-12"
    TWELVE_TO_SEVENTEEN = "13-17"
    EIGHTEEN_TO_TWENTY_FOUR = "18-24"
    TWENTY_FIVE_TO_THIRTY_FOUR = "25-34"
    THIRTY_FIVE_TO_FORTY_FOUR = "35-44"
    FORTY_FIVE_TO_FIFTY_FOUR = "45-54"
    FIFTY_FIVE_TO_SIXTY_FOUR = "55-64"
    SIXTY_FIVE_PLUS = "65+"


class ReferenceDocumentConstants:
    """Model for reference document."""

    SIMILARITY_THRESHOLD: Final[float] = 0.5


class LanguageCode(str, Enum):
    """Enum for supported language codes."""

    ENGLISH = "en"
    HINDI = "hi"
    BENGALI = "bn"
    PUNJABI = "pa"
    GUJARATI = "gu"
    ORIYA = "or"
    TAMIL = "ta"
    TELUGU = "te"
    KANNADA = "kn"
    MALAYALAM = "ml"


class Language(BaseModel):
    """Model for language and its percentage in the conversation."""

    language: str = Field(..., description="Name of the language")
    percentage: float = Field(
        ..., description="Percentage of the language used in conversation"
    )


class UserRole(str, Enum):
    CLIENT = "CLIENT"
    COUNSELOR = "COUNSELOR"


class ENV(str, Enum):
    DEV = "DEV"
    DEVELOPMENT = "DEVELOPMENT"
    PROD = "PROD"
    STG = "STG"


class PipelineStage(str, Enum):
    """Stages of the transcribe-and-summarize pipeline.

    Tagged onto failures so an error can be attributed to a specific step
    (and forwarded to ally-core / Slack) instead of the generic
    "transcription failed". Ordered roughly by execution order.
    """

    REQUEST_PARSE = "request-parse"
    DOWNLOAD = "download"
    CONVERT = "convert"
    TRANSCRIBE = "transcribe"
    DIARIZE = "diarize"
    SUMMARIZE = "summarize"
    DELIVER = "deliver"


class SQSWorkerConstants:
    """Constants for SQS worker configuration."""

    # INVARIANT: never receive more messages than we can process concurrently.
    # SQS starts the visibility clock on ALL received messages at receive time.
    # If we fetch more than MAX_CONCURRENT_MESSAGES, the surplus sit invisible
    # waiting for a processing slot while their visibility ticks down; a slow
    # transcription on the active slots then lets the waiting messages breach
    # VISIBILITY_TIMEOUT, so SQS redelivers them (duplicate processing) and,
    # after the queue's maxReceiveCount, dead-letters them → chat FAILED with no
    # transcript. Keeping MAX_MESSAGES == MAX_CONCURRENT_MESSAGES means every
    # received message starts processing immediately, so its visibility window
    # only ever covers its OWN processing time. Scale throughput horizontally
    # (more worker replicas), not by fetching a bigger batch one worker can't
    # chew. Aligned with LLM.MAX_CONCURRENT_LLM_CALLS.
    MAX_CONCURRENT_MESSAGES: Final[int] = 5
    MAX_MESSAGES: Final[int] = 5
    WAIT_TIME_SECONDS: Final[int] = 10
    # Must exceed the worst-case end-to-end processing time of a single message
    # (download + transcription + diarization + summary). Sarvam alone allows a
    # 600s job timeout, so a 120s visibility window let SQS redeliver the message
    # mid-flight and a second worker would process the same chat concurrently.
    # Keep this comfortably above the longest provider timeout.
    VISIBILITY_TIMEOUT: Final[int] = 900
    POLLING_INTERVAL: Final[int] = 0
    # Hard ceiling on processing a SINGLE message. The poll loop awaits the whole
    # batch before fetching more, so without this a single hung call (an STT/LLM
    # request that never returns) blocks the worker from polling at all and every
    # queued chat times out behind it (the reaper's "no transcript" batch). Kept
    # BELOW VISIBILITY_TIMEOUT (900s) so a timed-out message is abandoned before
    # SQS would redeliver it, and below the ally-be summary reaper (1200s) so a
    # legitimately slow-but-finishing job still beats the timeout.
    MESSAGE_PROCESSING_TIMEOUT_SECONDS: Final[int] = 840


class APISettings:
    API_V1_STR: str = "/api/v1"
    API_STR: str = "/api"
    X_API_KEY_HEADER: str = "x-api-key"
