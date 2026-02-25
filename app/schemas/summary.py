from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from app.core.constants import AgeRange, Language
from app.schemas.common import ChatMessage


class TagCategory(str, Enum):
    """Enum for message tag category."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class MessageTagLabel(str, Enum):
    """
    Static set of allowed message tag labels.

    Must match MessageTagLabelEnum in structured_output_models.
    """

    # Helpful Skill Tags (POSITIVE)
    ATTUNEMENT = "Attunement"
    GENUINE_WARMTH = "Genuine warmth"
    USE_OF_SILENCE = "Use of Silence"
    STEADY_PACING = "Steady pacing"
    ACTIVE_LISTENING = "Active listening"
    REFLECTION_OF_FEELINGS = "Reflection of Feelings"
    PARAPHRASING = "Paraphrasing"
    INSIGHT_GENERATION = "Insight Generation"
    SUMMARIZING = "Summarizing"
    REFLECTION_OF_EMOTIONS = "Reflection of Emotions"
    ACKNOWLEDGING = "Acknowledging"
    NORMALISATION = "Normalisation"
    VALIDATION = "Validation"
    NON_JUDGMENTAL_RESPONSE = "Non-judgmental response"
    OPEN_ENDED_QUESTION = "Open ended question"
    CLARIFYING_RESPONSE = "Clarifying Response"
    DEEPER_EXPLORATION = "Deeper Exploration"
    IDENTIFYING_STRENGTHS = "Identifying Strengths"
    AMBIVALENCE_REFLECTION = "Ambivalence reflection"
    EMOTIONAL_CONTAINMENT = "Emotional containment"
    GROUNDING_SUPPORT = "Grounding support"
    AFFIRMATION = "Affirmation"
    ADAPTIVE_COPING_EXPLORATION = "Adaptive coping exploration"
    CURIOUS_APPROACH = "Curious Approach"
    COLLABORATIVE_GOAL_SETTING = "Collaborative goal setting"
    REALISTIC_HOPE_BUILDING = "Realistic Hope Building"

    # Unhelpful Skill Tags (NEGATIVE)
    MISSED_OPPORTUNITY_TO_DEEPEN = "Missed opportunity to deepen"
    NEED_FOR_SLOWER_PACE = "Need for slower pace"
    REDUCED_PACING_NEEDED = "Reduced pacing needed"
    EXPAND_EMOTIONAL_VALIDATION = "Expand emotional validation"
    AVOID_COMPARISON_OR_REASSURANCE = "Avoid comparison or reassurance"
    AVOID_CLOSE_ENDED_QUESTIONS = "Avoid close ended questions"
    PACE_QUESTIONS = "Pace questions"
    ENHANCE_NON_DIRECTIVE_APPROACH = "Enhance Non-directive approach"
    DELAY_PROBLEM_SOLVING = "Delay problem-solving"
    INCREASE_USE_OF_SILENCE = "Increase use of silence"
    ALIGN_WITH_CLIENT_READINESS = "Align with client readiness"
    FACILITATE_COPING_EXPLORATION = "Facilitate Coping exploration"
    COLLABORATIVE_DIRECTION_NEEDED = "Collaborative direction needed"
    ALIGN_GOALS_WITH_READINESS = "Align goals with readiness"
    ANCHOR_HOPE_IN_REALITY = "Anchor hope in reality"
    AVOID_GENERAL_REASSURANCE = "Avoid general reassurance"
    STRENGTHEN_VALUES_LINK = "Strengthen values link"
    REINFORCE_AUTONOMY = "Reinforce autonomy"


class MessageTag(BaseModel):
    """A single tag with a label and category."""

    label: MessageTagLabel = Field(..., description="The tag label")
    category: TagCategory = Field(
        ..., description="Whether the tag is POSITIVE or NEGATIVE"
    )


class MessageTagItem(BaseModel):
    """Tags associated with a specific message in the transcript."""

    id: str = Field(..., description="The message ID from the transcript")
    tags: List[MessageTag] = Field(
        default_factory=list,
        description="List of tags for this message",
    )


class EmotionalMovementItem(BaseModel):
    """Emotional level for a client message."""

    message_id: str = Field(..., description="The client message ID")
    level: int = Field(
        ...,
        ge=-5,
        le=5,
        description=(
            "Emotional level from -5 (very negative/distressed) to +5 (very "
            "positive/happy)"
        ),
    )


class SkillCoverageItem(BaseModel):
    """Coverage percentage for a skill category."""

    category: str = Field(
        ...,
        description=(
            "The skill category name (e.g. Listening Engagement, Emotional "
            "Attunement, Supportive engagement)"
        ),
    )
    percentage: int = Field(
        ...,
        description="Coverage percentage for this category (0-100)",
    )

    @field_validator("percentage")
    @classmethod
    def clamp_percentage(cls, v: int) -> int:
        """Clamp to [0, 100]."""
        return max(0, min(100, v))


class SummaryNoteAndTagsRequest(BaseModel):
    """
    A Pydantic model representing the request for summarizing chat messages.
    """

    chat_history: List[ChatMessage] = Field(..., description="List of chat messages")
    keys: Optional[List[str]] = Field(
        None,
        description="Optional list of keys to include in the response. If provided, "
        "only these fields will be included and a dynamic response will be generated.",
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "chat_history": [
                    {"role": "counselor", "content": "How are you feeling?"},
                    {"role": "client", "content": "Not feeling well"},
                ],
                "keys": ["key_concerns", "dominant_feelings", "tags", "custom_field"],
            }
        }


class Tag(BaseModel):
    """
    A Pydantic model capturing the summary of the counseling session.
    """

    tag: str = Field(..., description="A tag to summarize the chat messages")
    positivity_rating: int = Field(..., description="Positivity rating of the tag")

    class ConfigDict:
        json_schema_extra = {"example": {"tag": "Stress", "positivity_rating": 2}}


class SummaryNoteAndTagsResponse(BaseModel):
    """
    A Pydantic model representing the response of the summary
    """

    call_id: Optional[str] = Field(None, description="Call ID of the session.")
    call_duration: Optional[int] = Field(
        None, description="Duration of the call in seconds."
    )
    call_date: Optional[str] = Field(
        None, description="Date of the counseling session."
    )
    call_time: Optional[str] = Field(None, description="Time of the call.")
    client_id: Optional[str] = Field(None, description="Client ID if available.")
    counsellor: Optional[str] = Field(None, description="Counsellor of the session.")
    call_type: Optional[str] = Field(None, description="Type of the call.")
    age: Optional[AgeRange] = Field(None, description="Age range of the client.")
    gender: Optional[str] = Field(None, description="Gender of the client.")
    profession: Optional[str] = Field(None, description="Profession of the client.")
    relationship_status: Optional[str] = Field(
        None, description="Relationship status of the client."
    )
    languages: Optional[List[Language]] = Field(
        None, description="Languages used in the conversation with their percentages."
    )
    location: Optional[str] = Field(None, description="Client's location.")
    code_of_concern: Optional[str] = Field(
        None, description="Code of concern for the session."
    )

    session_summary: Optional[str] = Field(
        None, description="Summary of the session as bullet points."
    )

    counseling_process_flow: Optional[str] = Field(
        None, description="Flow of the counseling process."
    )

    key_concerns: Optional[str] = Field(
        None, description="Key concerns shared by the client."
    )

    subjective_observations: Optional[str] = Field(
        None, description="Subjective observations of the client."
    )

    objective_observations: Optional[str] = Field(
        None, description="Objective observations of the client."
    )

    assessment: Optional[str] = Field(None, description="Assessment of the client.")

    dominant_feelings: Optional[str] = Field(
        None, description="Dominant feelings expressed by the client."
    )

    issues_worked_on: Optional[str] = Field(
        None, description="Issues worked on during the session."
    )

    key_therapeutic_techniques: Optional[str] = Field(
        None, description="Key therapeutic techniques used."
    )

    referrals_provided: Optional[str] = Field(
        None, description="Referrals provided to the client."
    )

    homework: Optional[str] = Field(
        None, description="Homework or tasks assigned to the client."
    )

    plan_for_next_call: Optional[str] = Field(
        None, description="Plan for the next call."
    )

    tags: List[Tag] = Field(
        ..., description="List of tags to summarize the chat messages"
    )

    listening_share: Optional[str] = Field(
        None, description="Listening share of the counselor."
    )
    reflective_questions_asked: int = Field(
        0, description="Count of reflective questions asked by the counselor."
    )
    open_ended_questions_asked: int = Field(
        0, description="Count of open-ended questions asked by the counselor."
    )
    back_channel_cues: int = Field(
        0, description="Count of back channel cues used by the counselor."
    )
    emotional_lift: Optional[str] = Field(
        None, description="Emotional lift of the client."
    )
    affirmations: int = Field(
        0, description="Count of affirmations used by the counselor."
    )
    reflective_listening: int = Field(
        0,
        description="Reflective listening score as percentage (0-100). Calculated "
        "as the ratio of counselor words that rephrase client content.",
    )
    avg_client_utterance_duration: Optional[float] = Field(
        None, description="Average duration of client utterances in seconds."
    )
    silence_by_counselor: int = Field(
        0,
        description="Count of silence moments by counselor. Only counts silence "
        "periods of 3+ seconds that occur after a client message and before a "
        "counselor message, or between two client messages.",
    )
    client_positivity_lift: Optional[float] = Field(
        None,
        description="Percentage change in client positivity over the conversation. "
        "Calculated by dividing client messages into 10 segments and measuring "
        "sentiment change between segments.",
    )
    counselor_interruptions: int = Field(
        0,
        description="Count of interruptions by the counselor. An interruption occurs "
        "when a counselor's message start_time falls within a client's message "
        "start_time to end_time range.",
    )

    call_quality: int = Field(
        ..., description="Quality of the call from a client perspective"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "call_id": None,
                "call_duration": None,
                "call_date": None,
                "call_time": None,
                "client_id": None,
                "counsellor": None,
                "call_type": None,
                "age": None,
                "gender": None,
                "profession": None,
                "relationship_status": None,
                "languages": [{"language": "English", "percentage": 0.85}],
                "location": None,
                "code_of_concern": "Stress Management",
                "session_summary": "The client expressed feelings of exhaustion and "
                "overwhelm due to the pressures of work and home responsibilities. "
                "The session focused on identifying goals for managing emotions and "
                "setting boundaries. The counselor guided the client through a "
                "breathing exercise and discussed practical steps for sharing "
                "household responsibilities with her husband and engaging in "
                "self-care activities. The client expressed hope for implementing "
                "the strategies discussed.",
                "counseling_process_flow": None,
                "key_concerns": "- Burnout\n- Work-life balance\n- Communication "
                "difficulties",
                "subjective_observations": "- Exhaustion\n- Overwhelm\n- "
                "Frustration\n- Detachment from work",
                "objective_observations": None,
                "assessment": "Client demonstrates insight into her challenges and "
                "is motivated to implement self-care strategies and communicate "
                "needs effectively.",
                "dominant_feelings": "- Fearful > Anxious > Overwhelmed\n- Fearful > "
                "Scared > Frightened\n- Fearful > Anxious > Worried\n- Happy > "
                "Optimistic > Hopeful",
                "issues_worked_on": "- Balancing work and home responsibilities\n- "
                "Setting boundaries at work\n- Communicating household needs",
                "key_therapeutic_techniques": "- Paced breathing exercise\n- Goal "
                "setting\n- Boundary setting",
                "referrals_provided": None,
                "homework": "- Read a book for 20 minutes before bed",
                "plan_for_next_call": "- Check on progress with setting boundaries\n- "
                "Discuss self-care routines",
                "tags": [
                    {"tag": "Exhaustion", "positivity_rating": 4},
                    {"tag": "Stress Management", "positivity_rating": 3},
                    {"tag": "Self-Care", "positivity_rating": 4},
                ],
                "listening_share": None,
                "reflective_questions_asked": 0,
                "open_ended_questions_asked": 0,
                "back_channel_cues": 0,
                "emotional_lift": "Calmer after breathing exercise",
                "affirmations": 0,
                "reflective_listening": 75,
                "avg_client_utterance_duration": 10.5,
                "silence_by_counselor": 15,
                "client_positivity_lift": 12.5,
                "counselor_interruptions": 2,
                "call_quality": 85,
            }
        }


class DynamicSummaryNoteResponse(BaseModel):
    """
    A Pydantic model representing the response for dynamic summary notes.
    """

    fields: dict[str, Optional[Union[str, int, float, List[dict]]]] = Field(
        default_factory=dict,
        description="A dictionary of dynamic fields in the summary",
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "fields": {
                    "work_life_balance_score": 3,
                    "sleep_quality_rating": 2,
                    "stress_level": "High",
                    "boundary_setting_attempts": "No emails after 8 PM",
                    "self_care_practices": "Morning yoga, Regular meals",
                    "languages": [
                        {"language": "English", "percentage": 85.5},
                        {"language": "Hindi", "percentage": 14.5},
                    ],
                }
            }
        }


class ContentEnhanceRequest(BaseModel):
    """
    A Pydantic model representing the request for enhancing the content.
    """

    content: str = Field(..., description="Content to be enhanced")

    class ConfigDict:
        json_schema_extra = {
            "example": {"content": "Exam stress - pressure from parents."}
        }


class ContentEnhanceResponse(BaseModel):
    """
    A Pydantic model representing the response of the content enhancement.
    """

    enhanced_content: str = Field(..., description="Enhanced content")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "enhanced_content": "The student is experiencing stress due to "
                "parental pressure."
            }
        }


class ContentEnhance(BaseModel):
    """
    A Pydantic model representing the response of the content enhancement.
    """

    enhanced_content: str = Field(
        ..., description="Enhanced content as bulletin points using '-'"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "enhanced_content": "The student is experiencing stress due to "
                "parental pressure."
            }
        }


class TagPositivityRatingRequest(BaseModel):
    """
    A Pydantic model representing the request for getting positivity ratings for tags.
    """

    tags: list[str] = Field(
        ..., description="List of tags to get positivity ratings for"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {"tags": ["Stress", "Anxiety", "Work-life balance"]}
        }


class TagPositivityRatingResponse(BaseModel):
    """
    A Pydantic model representing the response with positivity ratings for tags.
    """

    tags: list[Tag] = Field(
        ..., description="List of tags with their positivity ratings"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "tags": [
                    {"tag": "Stress", "positivity_rating": 2},
                    {"tag": "Anxiety", "positivity_rating": 1},
                    {"tag": "Work-life balance", "positivity_rating": 3},
                ]
            }
        }


class SimulationAnalysisRequest(BaseModel):
    """
    Request model for the deprecated /scenario/feedback endpoint.
    Use ScenarioEvaluationRequest for the new /scenario/evaluate endpoint.
    """

    chat_history: List[ChatMessage] = Field(
        ..., description="List of chat messages/exchanges in the simulation"
    )
    need_memory: bool = Field(
        default=False,
        description="Whether to generate memory summary alongside analysis",
    )
    previous_memory: Optional[str] = Field(
        default=None,
        description="Previous cumulative memory to build upon (when need_memory=True)",
    )
    memory_prompt: Optional[str] = Field(
        default=None,
        description="Custom instructions for memory generation (when need_memory=True)",
    )


class SimulationAnalysisResponse(BaseModel):
    """
    Response model for the deprecated /scenario/feedback endpoint.
    Use ScenarioEvaluationResponse for the new /scenario/evaluate endpoint.
    """

    improvements: List[str] = Field(
        ..., description="Areas that need improvement with specific, actionable points"
    )
    positives: List[str] = Field(
        ..., description="Things that went well and positive aspects demonstrated"
    )
    session_glimpse: Optional[str] = Field(
        default=None,
        description=(
            "Brief overview of the current session (only when need_memory=True)"
        ),
    )
    cumulative_memory: Optional[str] = Field(
        default=None,
        description=(
            "Comprehensive cumulative narrative across sessions (only when "
            "need_memory=True)"
        ),
    )


class ScenarioEvaluationRequest(BaseModel):
    """
    Request model for the /scenario/evaluate endpoint.
    """

    chat_history: List[ChatMessage] = Field(
        ..., description="List of chat messages with IDs for evaluation"
    )
    need_memory: bool = Field(
        default=False,
        description="Whether to generate memory summary alongside analysis",
    )
    previous_memory: Optional[str] = Field(
        default=None,
        description="Previous cumulative memory to build upon (when need_memory=True)",
    )
    memory_prompt: Optional[str] = Field(
        default=None,
        description="Custom instructions for memory generation (when need_memory=True)",
    )


class ScenarioEvaluationResponse(BaseModel):
    """
    Response model for the /scenario/evaluate endpoint.
    """

    improvements: List[str] = Field(
        ..., description="Areas that need improvement with specific, actionable points"
    )
    positives: List[str] = Field(
        ..., description="Things that went well and positive aspects demonstrated"
    )
    message_tags: List[MessageTagItem] = Field(
        default_factory=list,
        description="Per-message tags for counselor messages in the transcript",
    )
    emotional_movement: List[EmotionalMovementItem] = Field(
        default_factory=list,
        description="Emotional trajectory of client messages throughout the session",
    )
    skill_coverage: List[SkillCoverageItem] = Field(
        default_factory=list,
        description=(
            "Skill coverage percentages across categories (Listening Engagement, "
            "Emotional Attunement, Supportive engagement)"
        ),
    )
    session_glimpse: Optional[str] = Field(
        default=None,
        description=(
            "Brief overview of the current session (only when need_memory=True)"
        ),
    )
    cumulative_memory: Optional[str] = Field(
        default=None,
        description=(
            "Comprehensive cumulative narrative across sessions (only when "
            "need_memory=True)"
        ),
    )
