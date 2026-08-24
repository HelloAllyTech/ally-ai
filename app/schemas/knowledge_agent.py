"""Request/response shapes for the WhatsApp bot's knowledge agent.

Stateless by construction: conversation history and every retrieval threshold arrive on
the request. ally-be owns the rows, the phone numbers and the settings; this service
owns the retrieval and the wording.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.core.knowledge_agent.schemas import AnswerIntent, DeclineReason


class AgentTurn(BaseModel):
    """One previous message, so a follow-up like 'what about for children?' resolves."""

    role: str = Field(
        ..., description="'user' for the worker, anything else = assistant"
    )
    content: str = ""


class KnowledgeAnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: List[AgentTurn] = Field(
        default_factory=list,
        description="Recent turns, oldest first. Only the last few are used.",
    )
    prompts: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "ally-be prompt overrides, keyed by prompt code. Supplies the template "
            "text plus the admin-selected provider/model/temperature."
        ),
    )

    # --- retrieval knobs, all overridable per request from the whatsapp_bot settings
    # row ---
    top_k: Optional[int] = Field(None, ge=1, le=50)
    min_similarity: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Retrieval FLOOR. Permissive on purpose; see decline_similarity.",
    )
    decline_similarity: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "The actual decision: below this the agent declines WITHOUT calling the "
            "LLM. Separate from min_similarity because a relevant passage against a "
            "short paraphrased question scores ~0.40-0.60 with this embedding model, "
            "so a single hard floor at the decision value would decline constantly on "
            "legitimate rephrasings."
        ),
    )
    max_passages: Optional[int] = Field(None, ge=1, le=20)
    max_context_tokens: Optional[int] = Field(None, ge=200, le=100_000)
    similarity_band: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Keep hits within this similarity of the best one; drop the rest.",
    )
    max_answer_chars: int = Field(
        1400,
        ge=100,
        le=4000,
        description=(
            "Answer budget. ally-be appends source lines and caps the whole message at "
            "1600 characters, the portable ceiling across WhatsApp providers."
        ),
    )
    translate_query: bool = Field(
        True,
        description=(
            "Detect the question's language and embed an English restatement of it. On "
            "by default: cross-lingual embedding alignment is weak, so without this a "
            "Hindi or Tamil question retrieves badly and the answer is confidently "
            "built on the wrong passages."
        ),
    )
    document_ids: Optional[List[UUID]] = Field(
        None,
        description="Restrict retrieval to these documents; None searches everything",
    )


class Citation(BaseModel):
    """A resolved citation — everything ally-be needs to render a source line."""

    passage_number: int = Field(
        ..., description="Which numbered passage in the prompt this was"
    )
    chunk_id: UUID = Field(
        ...,
        description="Resolves against kb_document_chunks for the exact passage text",
    )
    document_id: str
    document_title: str
    page_from: int
    page_to: int
    section_path: str
    source_url: str
    similarity: float


class RetrievalMeta(BaseModel):
    """What retrieval actually did, so an answer can be explained after the fact."""

    top_k: int
    min_similarity: float
    decline_similarity: float
    hit_count: int
    top_similarity: float
    passages_used: int
    query_language: str = "en"
    translated_query: Optional[str] = Field(
        None,
        description=(
            "The English text actually embedded, when the question was translated. "
            "None means it was searched as written. Surfaced so an admin looking at a "
            "bad answer can see whether translation changed the question's meaning."
        ),
    )
    unsupported: bool = Field(
        False,
        description=(
            "True when the model answered but cited nothing. Kept rather than "
            "discarded (cross-passage synthesis legitimately cites nothing), but "
            "counted so the dashboard can show whether grounding is holding."
        ),
    )
    translation_degraded: bool = Field(
        False,
        description=(
            "True when query translation was attempted and failed, so retrieval ran "
            "on the original-language text instead. Surfaced independently of "
            "decline_reason because a degraded search can still clear the decline "
            "threshold and reach the model — this stays True even then, so an "
            "admin auditing a weak or wrong answer isn't misled into treating it as "
            "a normal-quality retrieval."
        ),
    )


class KnowledgeAnswerResponse(BaseModel):
    intent: AnswerIntent
    answer: str = ""
    language: str = "en"
    confidence: float = 0.0
    citations: List[Citation] = Field(default_factory=list)
    decline_reason: DeclineReason = DeclineReason.NONE
    retrieval: RetrievalMeta
    # The provider and model that ACTUALLY ran, which is not necessarily the one
    # configured: dispatch falls back when a key is missing. Empty for a pre-generation
    # decline, since no model ran at all. ally-be stores these on
    # wa_messages.retrieval_meta so a behaviour change after an admin swaps models stays
    # traceable.
    provider: str = ""
    model: str = ""
    prompt_version: str = ""


class CrisisCheckRequest(BaseModel):
    message: str = Field(..., min_length=1)
    prompts: Optional[Dict[str, Any]] = Field(
        None, description="ally-be prompt overrides, keyed by prompt code."
    )


class CrisisCheckResponse(BaseModel):
    """The crisis classifier's verdict.

    `failed` is reported rather than swallowed. A caller must be able to tell "the
    classifier looked and said no" from "the classifier could not run", because the
    second is a degraded safety net that only the keyword rules are holding up — and
    that is something an operator should be able to see rather than infer from a
    suspiciously quiet dashboard.
    """

    is_crisis: bool = False
    signal: str = ""
    confidence: float = 0.0
    failed: bool = False
    provider: str = ""
    model: str = ""
