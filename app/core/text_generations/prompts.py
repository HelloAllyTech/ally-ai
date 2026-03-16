from langchain_core.prompts import PromptTemplate
from app.prompts.resolver import load_template


NUDGE_PROMPT = PromptTemplate(
    template=load_template("nudge/nudge"),
    input_variables=["conversation", "chat_history", "suggestion"],
)

SUMMARY_PROMPT = PromptTemplate(
    template=load_template("summary/summary"),
    input_variables=["chat_history"],
)

CONTENT_ENHANCE_PROMPT = PromptTemplate(
    template=load_template("notes/content_enhance"),
    input_variables=["content"],
)

IDENTIFY_USER_PROMPT = PromptTemplate(
    template=load_template("user/identify_user"),
    input_variables=["conversations"],
)

DYNAMIC_SUMMARY_PROMPT = PromptTemplate(
    template=load_template("summary/dynamic_summary"),
    input_variables=["chat_history", "key_descriptions"],
)

TAG_POSITIVITY_RATING_PROMPT = PromptTemplate(
    template=load_template("tags/positivity_rating"),
    input_variables=["tags"],
)

DIARIZATION_PROMPT = PromptTemplate(
    template=load_template("audio/diarization"),
    input_variables=["transcription"],
)

COUNSELOR_ANALYSIS_PROMPT = PromptTemplate(
    template=load_template("analysis/counselor_analysis"),
    input_variables=["message"],
)

SCENARIO_EVALUATION_PROMPT = PromptTemplate(
    input_variables=["chat_history"],
    template=load_template("scenario/scenario_evaluation")
    .replace("{MESSAGE_TAG_PROMPT_TEXT}", load_template("shared/message_tags"))
    .replace("{SKILL_COVERAGE_DESCRIPTIONS}", load_template("shared/skill_coverage")),
)

SCENARIO_EVALUATION_WITH_MEMORY_PROMPT = PromptTemplate(
    input_variables=["chat_history", "previous_summary", "custom_prompt_section"],
    template=load_template("scenario/scenario_evaluation_with_memory")
    .replace("{MESSAGE_TAG_PROMPT_TEXT}", load_template("shared/message_tags"))
    .replace("{SKILL_COVERAGE_DESCRIPTIONS}", load_template("shared/skill_coverage")),
)

SIMULATION_ANALYSIS_PROMPT = PromptTemplate(
    template=load_template("simulation/simulation_analysis"),
    input_variables=["chat_history"],
)

SIMULATION_ANALYSIS_WITH_MEMORY_PROMPT = PromptTemplate(
    template=load_template("simulation/simulation_analysis_with_memory"),
    input_variables=["chat_history", "previous_summary", "custom_prompt_section"],
)
