from typing import List, Optional, Union

from pydantic import BaseModel, Field

from app.core.constants import AgeRange, Language
from app.schemas.common import ChatMessage


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

    session_summary: Optional[str] = Field(None, description="Summary of the session.")

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
