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
    # Analytics Agent: one planning call (question -> SQL) and one narration
    # call (rows -> answer) per question, priced separately because the
    # planner carries the whole schema catalogue and the narrator the rows.
    ANALYTICS_AGENT_PLAN = "analytics_agent_plan"
    ANALYTICS_AGENT_ANSWER = "analytics_agent_answer"


def resolve_model_name(model: Any) -> str:
    """Best-effort model id from a LangChain chat model (ChatOpenAI etc.)."""
    return (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or "unknown"
    )
