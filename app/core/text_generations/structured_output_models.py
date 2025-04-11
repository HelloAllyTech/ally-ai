from typing import Optional, List, Literal
from pydantic import BaseModel, Field, conint

from app.schemas.common import ChatMessage

DominantFeelingLiteral = Literal[
    "Betrayed:Let Down",
    "Resentful:Let Down",
    "Disrespected:Humiliated",
    "Ridiculed:Humiliated",
    "Indignant:Bitter",
    "Violated:Bitter",
    "Furious:Mad",
    "Jealous:Mad",
    "Hostile:Aggressive",
    "Provoked:Aggressive",
    "Infuriated:Frustrated",
    "Annoyed:Frustrated",
    "Withdrawn:Distant",
    "Numb:Distant",
    "Dismissive:Critical",
    "Skeptical:Critical",
    "Helpless:Scared",
    "Frightened:Scared",
    "Overwhelmed:Anxious",
    "Worried:Anxious",
    "Inadequate:Insecure",
    "Inferior:Insecure",
    "Worthless:Weak",
    "Insignificant:Weak",
    "Excluded:Rejected",
    "Persecuted:Rejected",
    "Nervous:Threatened",
    "Exposed:Threatened",
    "Indifferent:Bored",
    "Apathetic:Bored",
    "Pressured:Busy",
    "Rushed:Busy",
    "Overwhelmed:Stressed",
    "Out of control:Stressed",
    "Sleepy:Tired",
    "Unfocused:Tired",
    "Shocked:Startled",
    "Dismayed:Startled",
    "Disillusioned:Confused",
    "Perplexed:Confused",
    "Astonished:Amazed",
    "Awe:Amazed",
    "Eager:Excited",
    "Energetic:Excited",
    "Aroused:Playful",
    "Cheeky:Playful",
    "Free:Content",
    "Joyful:Content",
    "Curious:Interested",
    "Inquisitive:Interested",
    "Successful:Proud",
    "Confident:Proud",
    "Respected:Accepted",
    "Valued:Accepted",
    "Courageous:Powerful",
    "Creative:Powerful",
    "Loving:Peaceful",
    "Thankful:Peaceful",
    "Sensitive:Trusting",
    "Intimate:Trusting",
    "Hopeful:Optimistic",
    "Inspired:Optimistic",
    "Judgmental:Disapproving",
    "Embarrassed:Disapproving",
    "Appalled:Disappointed",
    "Revolted:Disappointed",
    "Nauseated:Awful",
    "Detestable:Awful",
    "Horrified:Repelled",
    "Hesitant:Repelled",
    "Embarrassed:Hurt",
    "Disappointed:Hurt",
    "Inferior:Depressed",
    "Empty:Depressed",
    "Remorseful:Guilty",
    "Ashamed:Guilty",
    "Grief:Despair",
    "Powerless:Despair",
    "Victimized:Vulnerable",
    "Fragile:Vulnerable",
    "Isolated:Lonely",
    "Abandoned:Lonely"
]

CodeOfConcernLiteral = Literal[
    "Academic Concerns",
    "Addiction Issues",
    "Anger Management",
    "Bereavement / Loss",
    "Bullying / Harassment",
    "Career Related Concerns",
    "Cultural / Identity Concerns",
    "Disability",
    "Domestic Abuse",
    "Financial Crisis",
    "Emotional Regulation",
    "Eating Disorders",
    "Interpersonal Conflicts",
    "Legal Concerns",
    "Loneliness / Isolation",
    "Mental Health",
    "Overthinking / Worrying",
    "Phobias / Fears",
    "Pregnancy and Parenting Stress",
    "Prank",  # Prank / Sexual Gratification, removing Sexual Gratification to remove confusion with other categories for the LLM
    "Referral Request",
    "Non-suicidal Self-Harm",
    "Self-esteem/ image Concerns",
    "Sexual Health Issues",
    "Sleep Issues / Insomnia",
    "Suicidal self-harm",
    "Stress Management",
    "Vicarious Trauma / Compassion Fatigue",
    "Violence",
    "Work-life Concerns",
    "Seeking Information (General Inquiry)"
]

WorkingStatusLiteral = Literal[
    "Student",
    "Working",
    "Other"
]

GenderLiteral = Literal[
    "Male",
    "Female",
    "Non-binary",
    "Client Prefers Not to Say"
]

PositivityRatingLiteral = Literal[1, 2, 3, 4, 5]


class StructuredSessionDetails(BaseModel):
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


class StructuredDemographicDetails(BaseModel):
    """
    A Pydantic model representing demographic details of a client.
    """
    client_id: Optional[str] = Field(None, description="Always assign null.")
    gender: Optional[GenderLiteral] = Field(None,
                                            description="Gender of the client. If not mentioned keep it as null")
    age: Optional[int] = Field(None, description="Age of the client in years.")
    location: Optional[str] = Field(None, description="Client's geographical location (e.g., country or city).")
    working_status: Optional[str] = Field(None, description="Values can be 'Student', 'Working', or 'Other'.")
    any_formal_diagnosis: Optional[str] = Field(None,
                                                description="Any formally diagnosed condition of the client (if applicable).")
    code_of_concern: Optional[CodeOfConcernLiteral] = Field(None,
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


class StructuredFollowUpPlan(BaseModel):
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


class StructuredSessionWork(BaseModel):
    """
    A Pydantic model capturing counseling related work.
    """
    counseling_process_flow: Optional[List[str]] = Field(None, description="Flow of the counseling process.")
    therapeutic_interventions: Optional[List[str]] = Field(None, description="Therapeutic interventions used.")
    issues_worked_on: Optional[List[str]] = Field(None, description="Issues worked on during the session.")
    homework: Optional[List[str]] = Field(None, description="Homework or tasks assigned to the client.")
    follow_up_plan: StructuredFollowUpPlan = Field(...,
                                                   description="Follow-up plan and goals for the next session.")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "counseling_process_flow": [
                    "Initial client sharing",
                    "Discussion of concerns",
                    "Exploration of coping mechanisms"
                ],
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


class StructuredSessionDocumentation(BaseModel):
    """
    A Pydantic model capturing counseling related documentation.
    """
    key_concerns: Optional[List[str]] = Field(None, description="Key concerns shared by the client.")
    dominant_feelings: Optional[List[DominantFeelingLiteral]] = Field(None,
                                                                      description="Dominant feelings expressed by the client.")
    work_done: StructuredSessionWork = Field(..., description="Counseling related work.")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "key_concerns": [
                    "Feeling overwhelmed by responsibilities",
                    "Difficulty in managing stress"
                ],
                "dominant_feelings": [
                    "Ashamed:Guilty",
                    "Grief:Despair"
                ],
                "work_done": {
                    "counseling_process_flow": [
                        "Opened up discussion about stressors",
                        "Explored relaxation techniques"
                    ],
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


class StructuredCounselorImpressions(BaseModel):
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


class StructuredTag(BaseModel):
    """
    A Pydantic model capturing the summary of the counseling session.
    """
    tag: str = Field(..., description="A tag to summarize the chat messages. Tag shouldn't be lengthy")
    positivity_rating: PositivityRatingLiteral = Field(...,
                                                       description="Positivity rating of the tag, 5 for highly positive and 1 for highly negative."
                                                       )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "tag": "Stress",
                "positivity_rating": 2
            }
        }


class StructuredSummaryNote(BaseModel):
    """
    A Pydantic model capturing the summary of the counseling session.
    """
    session_details: StructuredSessionDetails = Field(..., description="Details of the counseling session.")
    demographic_details: StructuredDemographicDetails = Field(...,
                                                              description="Demographic details of the client.")
    session_documentation: StructuredSessionDocumentation = Field(...,
                                                                  description="Documentation related to the counseling session.")
    counselor_impressions: StructuredCounselorImpressions = Field(...,
                                                                  description="Subjective impressions from the counselor.")
    tags: list[StructuredTag] = Field(...,
                                      description="List of tags to summarize the chat messages. Each tag should be concise.")
    call_quality: int = Field(...,
                              description="Quality rating of the call from 0 to 100 from a client perspective. The minimum is 0 and the maximum is 100.")

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
                    "key_concerns": [
                        "Feeling overwhelmed",
                        "Difficulty managing time"
                    ],
                    "dominant_feelings": [
                        "Ashamed:Guilty",
                        "Grief:Despair"
                    ],
                    "work_done": {
                        "counseling_process_flow": [
                            "Discussed daily stressors",
                            "Identified coping mechanisms"
                        ],
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
                "call_quality": 90,
            }
        }
