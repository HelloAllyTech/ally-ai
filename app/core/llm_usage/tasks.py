from enum import Enum
from typing import Any


class LLMTask(str, Enum):
    """Task labels for token-usage analytics. Must match ally-be's LlmTask enum."""

    NUDGE = "nudge"
    SUMMARY = "summary"
    DYNAMIC_SUMMARY = "dynamic_summary"
    SCENARIO_EVALUATION = "scenario_evaluation"
    COUNSELOR_ANALYSIS = "counselor_analysis"
    USER_IDENTIFICATION = "user_identification"
    CONTENT_ENHANCE = "content_enhance"
    TAG_POSITIVITY = "tag_positivity"
    DIARIZATION = "diarization"
    EMBEDDING = "embedding"
    DRIFT_JUDGE = "drift_judge"
    LANGUAGE_JUDGE = "language_judge"
    FEEDBACK_GROUNDEDNESS_JUDGE = "feedback_groundedness_judge"
    # Analytics Agent: one planning call (question -> SQL) and one narration
    # call (rows -> answer) per question, priced separately because the
    # planner carries the whole schema catalogue and the narrator the rows.
    ANALYTICS_AGENT_PLAN = "analytics_agent_plan"
    ANALYTICS_AGENT_ANSWER = "analytics_agent_answer"
    # WhatsApp Q&A bot. Two labels, not one: answering a question can involve both calls
    # and their cost shapes differ by an order of magnitude — the answer call carries
    # the retrieved passages on the admin-selected model, the translate call is a short
    # mechanical transform on a cheap one. Merged, cost-per-question would be
    # unattributable.
    #
    # No embedding label yet. The corpus and query embeddings are real cost, but
    # emitting them needs an 'embedding' branch in emitter._has_quantity plus matching
    # handling in ally-be's consumer (see the note in KnowledgeChunkService). Adding an
    # enum member here before ally-be has the matching one would be exactly the drift
    # this enum's docstring warns against.
    WHATSAPP_RAG_ANSWER = "whatsapp_rag_answer"
    WHATSAPP_QUERY_TRANSLATE = "whatsapp_query_translate"
    WHATSAPP_CRISIS_CLASSIFY = "whatsapp_crisis_classify"


def resolve_model_name(model: Any) -> str:
    """Best-effort model id from a LangChain chat model (ChatOpenAI etc.)."""
    return (
        getattr(model, "model_name", None) or getattr(model, "model", None) or "unknown"
    )
