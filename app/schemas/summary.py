from typing import Optional, List, Union
from pydantic import BaseModel, Field

from app.schemas.common import ChatMessage
from app.core.constants import AgeRange, Language


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
    call_id: Optional[str] = Field(None, description="Call ID of the session.")
    call_duration: Optional[int] = Field(None, description="Duration of the call in seconds.")
    call_date: Optional[str] = Field(None, description="Date of the counseling session.")
    call_time: Optional[str] = Field(None, description="Time of the call.")
    client_id: Optional[str] = Field(None, description="Client ID if available.")
    counsellor: Optional[str] = Field(None, description="Counsellor of the session.")
    call_type: Optional[str] = Field(None, description="Type of the call.")
    age: Optional[AgeRange] = Field(None, description="Age range of the client.")
    gender: Optional[str] = Field(None, description="Gender of the client.")
    profession: Optional[str] = Field(None, description="Profession of the client.")
    relationship_status: Optional[str] = Field(None, description="Relationship status of the client.")
    languages: Optional[List[Language]] = Field(None, description="Languages used in the conversation with their percentages.")
    location: Optional[str] = Field(None, description="Client's location.")
    code_of_concern: Optional[str] = Field(None, description="Code of concern for the session.")

    session_summary: Optional[str] = Field(None, description="Summary of the session.")

    counseling_process_flow: Optional[str] = Field(None, description="Flow of the counseling process.")
   
    key_concerns: Optional[str] = Field(None, description="Key concerns shared by the client.")

    subjective_observations: Optional[str] = Field(None, description="Subjective observations of the client.")
    
    objective_observations: Optional[str] = Field(None, description="Objective observations of the client.")
    
    assessment: Optional[str] = Field(None, description="Assessment of the client.")

    dominant_feelings: Optional[List[str]] = Field(None, description="Dominant feelings expressed by the client.")

    issues_worked_on: Optional[str] = Field(None, description="Issues worked on during the session.")

    key_therapeutic_techniques: Optional[str] = Field(None, description="Key therapeutic techniques used.")

    referrals_provided: Optional[str] = Field(None, description="Referrals provided to the client.")

    homework: Optional[str] = Field(None, description="Homework or tasks assigned to the client.")

    plan_for_next_call: Optional[str] = Field(None, description="Plan for the next call.")

    tags: List[Tag] = Field(..., description="List of tags to summarize the chat messages")

    listening_share: Optional[str] = Field(None, description="Listening share of the counselor.")
    reflective_questions_asked: Optional[int] = Field(None, description="Count of reflective questions asked by the counselor.")
    emotional_lift: Optional[str] = Field(None, description="Emotional lift of the client.")

    call_quality: int = Field(..., description="Quality of the call from a client perspective")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "call_id": "CALL123",
                "call_duration": 1800,
                "call_date": "2025-02-11",
                "call_time": "10:00 AM",
                "client_id": "CL123",
                "counsellor": "Shruti",
                "call_type": "Follow-up",
                "age": "25-34",
                "gender": "Female",
                "profession": "Software Engineer",
                "relationship_status": "Single",
                "languages": [
                    {
                        "language": "English",
                        "percentage": 0.85
                    }
                ],
                "location": "Mumbai",
                "code_of_concern": "Work-life Concerns",
                "session_summary": "Client discussed work-life balance challenges and developed coping strategies.",
                "counseling_process_flow": [
                                    "Initial assessment",
                                    "Discussion of concerns",
                                    "Strategy development"
                                ],
                "key_concerns": [
                    "Feeling overwhelmed by responsibilities",
                    "Difficulty managing time"
                ],
                "subjective_observations": [
                    "Client appears tired",
                    "Shows interest in learning new techniques"
                ],
                "objective_observations": [
                    "Maintained good eye contact",
                    "Engaged in discussion"
                ],
                "assessment": "Client is experiencing work-related stress but is motivated to improve.",
                "dominant_feelings": [
                    "Anxious",
                    "Overwhelmed"
                ],
                "issues_worked_on": [
                    "Time management",
                    "Stress reduction"
                ],
                "key_therapeutic_techniques": [
                    "Deep breathing",
                    "Thought challenging"
                ],
                "referrals_provided": [
                    "Stress management workshop",
                    "Time management course"
                ],
                "homework": [
                    "Practice meditation for 5 minutes daily",
                    "Keep a time log"
                ],
                "plan_for_next_call": [
                    "Review progress",
                    "Adjust strategies if needed"
                ],
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
                "listening_share": "I listened to the client's concerns and provided support.",
                "reflective_questions_asked": [
                    "What are your thoughts on the session?",
                    "What did you think of the strategies we discussed?"
                ],
                "emotional_lift": "Client felt more relaxed after the session",
                
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
