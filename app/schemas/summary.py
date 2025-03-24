from typing import Optional, List
from pydantic import BaseModel, Field

from app.schemas.common import ChatMessage


class SummaryNoteAndTagsRequest(BaseModel):
    """
    A Pydantic model representing the request for summarizing chat messages.
    """
    chat_history: List[ChatMessage] = Field(..., description="List of chat messages")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "chat_history": [
                    {"role": "counselor", "content": "How are you feeling?"},
                    {"role": "client", "content": "Not feeling well"},
                ]
            }
        }


class SessionDetails(BaseModel):
    """
    A Pydantic model representing session details.
    """
    date_of_session: Optional[str] = Field(None, description="Date of the counseling session.")
    new_call_follow_up: Optional[str] = Field(None, description="Indicates if this is a new or follow-up call.")
    session_number: Optional[str] = Field(None, description="Session number if it's a follow-up.")
    counselor_name: Optional[str] = Field(None, description="Name of the counselor handling the session.")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "date_of_session": "11th February 2025",
                "new_call_follow_up": "New Call",
                "session_number": None,
                "counselor_name": "Shruti"
            }
        }


class DemographicDetails(BaseModel):
    """
    A Pydantic model representing demographic details of a client.
    """
    client_id: Optional[str] = Field(None, description="Always assign null.")
    gender: Optional[str] = Field(None,
                                  description="Gender of the client (e.g., Male, Female, Non-binary, Prefer not to say).")
    age: Optional[int] = Field(None, description="Age of the client in years.")
    location: Optional[str] = Field(None, description="Client's geographical location (e.g., country or city).")
    working_status: Optional[str] = Field(None, description="Values can be 'Student', 'Working', or 'Other'.")
    any_formal_diagnosis: Optional[str] = Field(None,
                                                description="Any formally diagnosed condition of the client (if applicable).")
    code_of_concern: Optional[str] = Field(None,
                                           description="A predefined code representing the area of concern for the client.")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "client_id": None,
                "gender": "Female",
                "age": 28,
                "location": "Mumbai",
                "working_status": "Working",
                "any_formal_diagnosis": None,
                "code_of_concern": "Work-life Concerns"
            }
        }


class FollowUpPlan(BaseModel):
    """
    A Pydantic model capturing the follow-up plan and goals for the next session.
    """
    status: Optional[str] = Field(None, description="Status of the follow-up session (e.g., Scheduled, Not scheduled).")
    follow_up_date: Optional[str] = Field(None, description="Date and time for the follow-up session (if scheduled).")
    goals: Optional[List[str]] = Field(None, description="List of goals set for the next session.")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "Scheduled for next week",
                "follow_up_date": "4:00 pm | 15.02.2025",
                "goals": [
                    "Explore deeper emotional patterns.",
                    "Aid with decision making regarding her life plans."
                ]
            }
        }


class SessionWork(BaseModel):
    """
    A Pydantic model capturing counseling related work.
    """
    counseling_process_flow: Optional[str] = Field(None, description="Flow of the counseling process.")
    therapeutic_interventions: Optional[List[str]] = Field(None, description="Therapeutic interventions used.")
    issues_worked_on: Optional[List[str]] = Field(None, description="Issues worked on during the session.")
    homework: Optional[List[str]] = Field(None, description="Homework or tasks assigned to the client.")
    follow_up_plan: Optional[FollowUpPlan] = Field(None, description="Follow-up plan and goals for the next session.")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "counseling_process_flow": "- Initial client sharing\n- Discussion of concerns\n- Exploration of coping mechanisms",
                "therapeutic_interventions": [
                    "Cognitive Behavioral Therapy",
                    "Mindfulness exercises"
                ],
                "issues_worked_on": [
                    "Anxiety management",
                    "Sleep improvement"
                ],
                "homework": [
                    "Journaling daily",
                    "Practice meditation for 10 minutes"
                ],
                "follow_up_plan": {
                    "status": "Scheduled",
                    "follow_up_date": "2025-03-10 14:00",
                    "goals": [
                        "Review homework progress",
                        "Introduce new coping strategies"
                    ]
                }
            }
        }


class SessionDocumentation(BaseModel):
    """
    A Pydantic model capturing counseling related documentation.
    """
    key_concerns: Optional[str] = Field(None, description="Key concerns shared by the client.")
    dominant_feelings: Optional[List[str]] = Field(None, description="Dominant feelings expressed by the client.")
    work_done: Optional[SessionWork] = Field(None, description="Counseling related work.")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "key_concerns": "- Feeling overwhelmed by responsibilities\n- Difficulty in managing stress",
                "dominant_feelings": [
                    "Ashamed:Guilty",
                    "Grief:Despair"
                ],
                "work_done": {
                    "counseling_process_flow": "- Initial client sharing\n- Discussion of concerns\n- Exploration of coping mechanisms",
                    "therapeutic_interventions": [
                        "Guided meditation",
                        "Cognitive restructuring"
                    ],
                    "issues_worked_on": [
                        "Identifying stress triggers"
                    ],
                    "homework": [
                        "Daily relaxation exercises"
                    ],
                    "follow_up_plan": {
                        "status": "Pending",
                        "follow_up_date": "2025-03-12 15:00",
                        "goals": [
                            "Assess progress on relaxation",
                            "Plan further interventions"
                        ]
                    }
                }
            }
        }


class CounselorImpressions(BaseModel):
    """
    A Pydantic model capturing subjective impressions from the counselor.
    """
    client_attitude: Optional[str] = Field(None, description="Client's attitude towards therapy.")
    emotional_state_start: Optional[str] = Field(None,
                                                 description="Client’s emotional state at the beginning of the session.")
    emotional_state_change: Optional[str] = Field(None,
                                                  description="Changes in the client’s emotional state during the session.")
    problem_analysis: Optional[str] = Field(None, description="Counselor's conceptualization of the client’s problem.")
    additional_insights: Optional[str] = Field(None,
                                               description="Any other insights or observations noted by the counselor.")
    counselor_feelings: Optional[str] = Field(None, description="How the counselor felt during the session.")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "client_attitude": "Open but hesitant about practicing self-compassion due to guilt.",
                "emotional_state_start": "Overwhelmed, emotionally exhausted.",
                "emotional_state_change": "Slight relief after discussing challenges; moments of insight.",
                "problem_analysis": "Client feels deeply responsible for others, leading to self-neglect.",
                "additional_insights": "Client’s sense of identity is strongly tied to caregiving, making it difficult to prioritize personal well-being.",
                "counselor_feelings": "Feeling energetic and alert."
            }
        }


class SummaryNote(BaseModel):
    """
    A Pydantic model capturing the summary of the counseling session.
    """
    session_details: Optional[SessionDetails] = Field(None, description="Details of the counseling session.")
    demographic_details: Optional[DemographicDetails] = Field(None, description="Demographic details of the client.")
    session_documentation: Optional[SessionDocumentation] = Field(None,
                                                                  description="Documentation related to the counseling session.")
    counselor_impressions: Optional[CounselorImpressions] = Field(None,
                                                                  description="Subjective impressions from the counselor.")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "session_details": {
                    "date_of_session": "2025-02-11",
                    "new_call_follow_up": "Follow-up",
                    "session_number": None,
                    "counselor_name": "Shruti"
                },
                "demographic_details": {
                    "client_id": None,
                    "gender": "Female",
                    "age": 28,
                    "location": "Mumbai",
                    "working_status": "Working",
                    "any_formal_diagnosis": None,
                    "code_of_concern": "Work-life Concerns"
                },
                "session_documentation": {
                    "key_concerns": "- Feeling overwhelmed by responsibilities\n- Difficulty in managing stress",
                    "dominant_feelings": [
                        "Ashamed:Guilty",
                        "Grief:Despair"
                    ],
                    "work_done": {
                        "counseling_process_flow": "- Initial client sharing\n- Discussion of concerns\n- Exploration of coping mechanisms",
                        "therapeutic_interventions": [
                            "Mindfulness",
                            "Cognitive reframing"
                        ],
                        "issues_worked_on": [
                            "Work-life balance"
                        ],
                        "homework": [
                            "Practice meditation for 5 minutes daily"
                        ],
                        "follow_up_plan": {
                            "status": "Scheduled",
                            "follow_up_date": "2025-02-15 10:00",
                            "goals": [
                                "Monitor stress levels",
                                "Adjust coping strategies"
                            ]
                        }
                    }
                },
                "counselor_impressions": {
                    "client_attitude": "Cooperative",
                    "emotional_state_start": "Anxious",
                    "emotional_state_change": "Calmer after session",
                    "problem_analysis": "Client struggles with self-management",
                    "additional_insights": "Needs consistent follow-up",
                    "counselor_feelings": "Encouraged"
                }
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
    summary_note: SummaryNote = Field(..., description="Summary of the counseling session")
    tags: list[Tag] = Field(..., description="List of tags to summarize the chat messages")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "summary_note": {
                    "session_details": {
                        "date_of_session": "2025-02-11",
                        "new_call_follow_up": "Follow-up",
                        "session_number": None,
                        "counselor_name": "Shruti"
                    },
                    "demographic_details": {
                        "client_id": None,
                        "gender": "Female",
                        "age": 28,
                        "location": "Mumbai",
                        "working_status": "Working",
                        "any_formal_diagnosis": None,
                        "code_of_concern": "Work-life Concerns"
                    },
                    "session_documentation": {
                        "key_concerns": "- Feeling overwhelmed by responsibilities\n- Difficulty in managing stress",
                        "dominant_feelings": [
                            "Ashamed:Guilty",
                            "Grief:Despair"
                        ],
                        "work_done": {
                            "counseling_process_flow": "- Initial client sharing\n- Discussion of concerns\n- Exploration of coping mechanisms",
                            "therapeutic_interventions": [
                                "Mindfulness",
                                "Cognitive reframing"
                            ],
                            "issues_worked_on": [
                                "Work-life balance"
                            ],
                            "homework": [
                                "Practice meditation for 5 minutes daily"
                            ],
                            "follow_up_plan": {
                                "status": "Scheduled",
                                "follow_up_date": "2025-02-15 10:00",
                                "goals": [
                                    "Monitor stress levels",
                                    "Adjust coping strategies"
                                ]
                            }
                        }
                    },
                    "counselor_impressions": {
                        "client_attitude": "Cooperative",
                        "emotional_state_start": "Anxious",
                        "emotional_state_change": "Calmer after session",
                        "problem_analysis": "Client struggles with self-management",
                        "additional_insights": "Needs consistent follow-up",
                        "counselor_feelings": "Encouraged"
                    },
                },
                "tags": [
                    {
                        "tag": "Stress",
                        "positivity_rating": 2
                    },
                    {
                        "tag": "Anxiety",
                        "positivity_rating": 1
                    }
                ]
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
