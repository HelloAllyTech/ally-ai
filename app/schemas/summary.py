from typing import Optional, List, Union
from pydantic import BaseModel, Field


from app.schemas.common import ChatMessage


class SummaryNoteAndTagsRequest(BaseModel):
    """
    A Pydantic model representing the request for summarizing chat messages.
    """
    chat_history: List[ChatMessage] = Field(..., description="List of chat messages")
    keys: Optional[List[str]] = Field(None, description="Optional list of keys to include in the response. If provided, only these fields will be included and a dynamic response will be generated.")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "chat_history": [
                    {"role": "counselor", "content": "How are you feeling?"},
                    {"role": "client", "content": "Not feeling well"},
                ],
                "keys": ["key_concerns", "dominant_feelings", "tags", "custom_field"]
            }
        }


class Tag(BaseModel):
    """
    A Pydantic model capturing the summary of the counseling session.
    """
    tag: str = Field(..., description="A tag to summarize the chat messages")
    positivity_rating: int = Field(..., description="Positivity rating of the tag")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "tag": "Stress",
                "positivity_rating": 2
            }
        }


class SummaryNoteAndTagsResponse(BaseModel):
    """
    A Pydantic model representing the response of the summary
    """
    # Session Details
    date_of_session: Optional[str] = Field(None, description="Date of the counseling session.")
    new_call_follow_up: Optional[str] = Field(None, description="Indicates if this is a new or follow-up call.")
    session_number: Optional[str] = Field(None, description="Session number if it's a follow-up.")
    counselor_name: Optional[str] = Field(None, description="Name of the counselor handling the session.")

    # Demographic Details
    client_id: Optional[str] = Field(None, description="Client ID if available.")
    gender: Optional[str] = Field(None, description="Gender of the client.")
    age: Optional[int] = Field(None, description="Age of the client.")
    location: Optional[str] = Field(None, description="Client's location.")
    working_status: Optional[str] = Field(None, description="Client's working status.")
    any_formal_diagnosis: Optional[str] = Field(None, description="Any formal diagnosis if available.")
    code_of_concern: Optional[str] = Field(None, description="Code of concern for the session.")
    call_id: Optional[str] = Field(None, description="Call ID of the session.")
    call_duration: Optional[int] = Field(None, description="Duration of the call in seconds.")
    call_time: Optional[str] = Field(None, description="Time of the call.")
    counsellor: Optional[str] = Field(None, description="Counsellor of the session.")
    call_type: Optional[str] = Field(None, description="Type of the call.")
    profession: Optional[str] = Field(None, description="Profession of the client.")
    relationship_status: Optional[str] = Field(None, description="Relationship status of the client.")

    # Session Documentation
    key_concerns: Optional[str] = Field(None, description="Key concerns shared by the client.")
    dominant_feelings: Optional[List[str]] = Field(None, description="Dominant feelings expressed by the client.")
    counseling_process_flow: Optional[str] = Field(None, description="Flow of the counseling process.")
    therapeutic_interventions: Optional[str] = Field(None, description="Therapeutic interventions used.")
    issues_worked_on: Optional[str] = Field(None, description="Issues worked on during the session.")
    homework: Optional[str] = Field(None, description="Homework or tasks assigned to the client.")
    session_summary: Optional[str] = Field(None, description="Summary of the session.")
    subjective_observations: Optional[str] = Field(None, description="Subjective observations of the client.")
    objective_observations: Optional[str] = Field(None, description="Objective observations of the client.")
    assessment: Optional[str] = Field(None, description="Assessment of the client.")
    key_therapeutic_techniques: Optional[str] = Field(None, description="Key therapeutic techniques used.")
    referrals_provided: Optional[str] = Field(None, description="Referrals provided to the client.")
    plan_for_next_call: Optional[str] = Field(None, description="Plan for the next call.")
    metrics: Optional[str] = Field(None, description="Metrics for the session.")
    

    # Follow-up Plan
    follow_up_status: Optional[str] = Field(None, description="Status of the follow-up session.")
    follow_up_date: Optional[str] = Field(None, description="Date and time for the follow-up session.")
    follow_up_goals: Optional[str] = Field(None, description="Goals set for the next session.")

    # Counselor Impressions
    client_attitude: Optional[str] = Field(None, description="Client's attitude towards therapy.")
    emotional_state_start: Optional[str] = Field(None, description="Client's emotional state at the beginning.")
    emotional_state_change: Optional[str] = Field(None, description="Changes in client's emotional state.")
    problem_analysis: Optional[str] = Field(None, description="Counselor's analysis of the problem.")
    additional_insights: Optional[str] = Field(None, description="Additional insights from the counselor.")
    counselor_feelings: Optional[str] = Field(None, description="How the counselor felt during the session.")

    # Tags and Quality
    tags: List[Tag] = Field(..., description="List of tags to summarize the chat messages")
    call_quality: int = Field(..., description="Quality of the call from a client perspective")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "date_of_session": "2025-02-11",
                "new_call_follow_up": "Follow-up",
                "session_number": "2",
                "counselor_name": "Shruti",
                "client_id": "CL123",
                "gender": "Female",
                "age": 28,
                "location": "Mumbai",
                "working_status": "Working",
                "any_formal_diagnosis": None,
                "code_of_concern": "Work-life Concerns",
                "call_id": "CALL123",
                "call_duration": 1800,
                "call_time": "10:00 AM",
                "counsellor": "Shruti",
                "call_type": "Follow-up",
                "profession": "Software Engineer",
                "relationship_status": "Single",
                "key_concerns": [
                    "Feeling overwhelmed by responsibilities",
                    "Difficulty managing time"
                ],
                "dominant_feelings": [
                    "Anxious",
                    "Overwhelmed"
                ],
                "counseling_process_flow": [
                    "Initial assessment",
                    "Discussion of concerns",
                    "Strategy development"
                ],
                "therapeutic_interventions": [
                    "Mindfulness",
                    "Cognitive reframing"
                ],
                "issues_worked_on": [
                    "Time management",
                    "Stress reduction"
                ],
                "homework": [
                    "Practice meditation for 5 minutes daily",
                    "Keep a time log"
                ],
                "session_summary": "Client discussed work-life balance challenges and developed coping strategies.",
                "subjective_observations": [
                    "Client appears tired",
                    "Shows interest in learning new techniques"
                ],
                "objective_observations": [
                    "Maintained good eye contact",
                    "Engaged in discussion"
                ],
                "assessment": "Client is experiencing work-related stress but is motivated to improve.",
                "key_therapeutic_techniques": [
                    "Deep breathing",
                    "Thought challenging"
                ],
                "referrals_provided": [
                    "Stress management workshop",
                    "Time management course"
                ],
                "plan_for_next_call": [
                    "Review progress",
                    "Adjust strategies if needed"
                ],
                "metrics": [
                    "Stress level reduced by 20%",
                    "Sleep quality improved"
                ],
                "follow_up_status": "Scheduled",
                "follow_up_date": "2025-02-15 10:00",
                "follow_up_goals": [
                    "Monitor stress levels",
                    "Adjust coping strategies"
                ],
                "client_attitude": "Cooperative",
                "emotional_state_start": "Anxious",
                "emotional_state_change": "Calmer after session",
                "problem_analysis": "Client struggles with self-management",
                "additional_insights": "Needs consistent follow-up",
                "counselor_feelings": "Encouraged",
                "tags": [
                    {
                        "tag": "Stress",
                        "positivity_rating": 2
                    },
                    {
                        "tag": "Work-life balance",
                        "positivity_rating": 3
                    }
                ],
                "call_quality": 90
            }
        }


class DynamicSummaryNoteResponse(BaseModel):
    """
    A Pydantic model representing the response for dynamic summary notes.
    """
    fields: dict[str,  Union[str, int]] = Field(default_factory=dict, description="A dictionary of dynamic fields in the summary")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "fields": {
                    "work_life_balance_score": 3,
                    "sleep_quality_rating": 2,
                    "stress_level": "High",
                    "boundary_setting_attempts": "No emails after 8 PM",
                    "self_care_practices": "Morning yoga, Regular meals"
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
            "example": {
                "content": "Exam stress - pressure from parents."
            }
        }


class ContentEnhanceResponse(BaseModel):
    """
    A Pydantic model representing the response of the content enhancement.
    """
    enhanced_content: str = Field(..., description="Enhanced content")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "enhanced_content": "The student is experiencing stress due to parental pressure."
            }
        }


class ContentEnhance(BaseModel):
    """
    A Pydantic model representing the response of the content enhancement.
    """
    enhanced_content: str = Field(..., description="Enhanced content as bulletin points using '-'")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "enhanced_content": "The student is experiencing stress due to parental pressure."
            }
        }


class TagPositivityRatingRequest(BaseModel):
    """
    A Pydantic model representing the request for getting positivity ratings for tags.
    """
    tags: list[str] = Field(..., description="List of tags to get positivity ratings for")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "tags": ["Stress", "Anxiety", "Work-life balance"]
            }
        }


class TagPositivityRatingResponse(BaseModel):
    """
    A Pydantic model representing the response with positivity ratings for tags.
    """
    tags: list[Tag] = Field(..., description="List of tags with their positivity ratings")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "tags": [
                    {
                        "tag": "Stress",
                        "positivity_rating": 2
                    },
                    {
                        "tag": "Anxiety",
                        "positivity_rating": 1
                    },
                    {
                        "tag": "Work-life balance",
                        "positivity_rating": 3
                    }
                ]
            }
        }
