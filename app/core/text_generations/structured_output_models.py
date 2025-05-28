from typing import Optional, List, Literal
from pydantic import BaseModel, Field

from app.core.constants import AgeRange

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


class StructuredTag(BaseModel):
    """
    A Pydantic model capturing the summary of the counseling session.
    """
    tag: str = Field(..., description="A tag to summarize the chat messages. Tag shouldn't be lengthy")
    positivity_rating: int = Field(...,
                                   description="Positivity rating of the tag, 5 for highly positive and 1 for highly negative.")


class StructuredSummaryNote(BaseModel):
    """
    A Pydantic model capturing the summary of the counseling session.
    """
    # Session Details
    date_of_session: Optional[str] = Field(None, description="Date of the counseling session.")
    new_call_follow_up: Optional[str] = Field(None, description="Indicates if this is a new or follow-up call.")
    session_number: Optional[str] = Field(None, description="Session number if it's a follow-up.")
    counselor_name: Optional[str] = Field(None, description="Name of the counselor handling the session.")

    # Demographic Details
    client_id: Optional[str] = Field(None, description="Client ID if available.")
    gender: Optional[GenderLiteral] = Field(None, description="Gender of the client.")
    age: Optional[AgeRange] = Field(None, description="Age range of the client.")
    location: Optional[str] = Field(None, description="Client's location.")
    working_status: Optional[WorkingStatusLiteral] = Field(None, description="Client's working status.")
    any_formal_diagnosis: Optional[str] = Field(None, description="Any formal diagnosis if available.")
    code_of_concern: Optional[CodeOfConcernLiteral] = Field(None, description="Code of concern for the session.")
    call_id: Optional[str] = Field(None, description="Call ID of the session.")
    call_duration: Optional[int] = Field(None, description="Duration of the call in seconds.")
    call_time: Optional[str] = Field(None, description="Time of the call.")
    counsellor: Optional[str] = Field(None, description="Counsellor of the session.")
    call_type: Optional[str] = Field(None, description="Type of the call.")
    profession: Optional[str] = Field(None, description="Profession of the client.")
    relationship_status: Optional[str] = Field(None, description="Relationship status of the client.")
    session_summary: Optional[str] = Field(None, description="Summary of the session.")
    counseling_process_flow: Optional[List[str]] = Field(None, description="Flow of the counseling process.")

    key_concerns: Optional[List[str]] = Field(None, description="Key concerns shared by the client.")
    subjective_observations: Optional[List[str]] = Field(None, description="Subjective observations of the client.")
    objective_observations: Optional[List[str]] = Field(None, description="Objective observations of the client.")
    assessment: Optional[str] = Field(None, description="Assessment of the client.")
    dominant_feelings: Optional[List[DominantFeelingLiteral]] = Field(None,
                                                                      description="Dominant feelings expressed by the client.")
    issues_worked_on: Optional[List[str]] = Field(None, description="Issues worked on during the session.")
    key_therapeutic_techniques: Optional[List[str]] = Field(None, description="Key therapeutic techniques used.")
    referrals_provided: Optional[List[str]] = Field(None, description="Referrals provided to the client.")
    homework: Optional[List[str]] = Field(None, description="Homework or tasks assigned to the client.")
    plan_for_next_call: Optional[List[str]] = Field(None, description="Plan for the next call.")
    tags: List[StructuredTag] = Field(..., description="List of tags to summarize the chat messages.")
    reflective_questions_asked: Optional[List[str]] = Field(None,
                                                            description="Reflective questions asked by the counselor.")
    open_ended_questions_asked: Optional[List[str]] = Field(None,
                                                            description="Open-ended questions asked by the counselor.")
    emotional_lift: Optional[str] = Field(None, description="Emotional lift of the client.")
    call_quality: int = Field(..., description="Quality rating of the call from 0 to 100.")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "call_id": "CALL123",
                "call_date": "2025-02-11",
                "call_duration": 1800,
                "call_time": "10:00 AM",
                "caller_id": "CL123",
                "counsellor": "Shruti",
                "call_type": "Follow-up",
                "age": "25-34",
                "gender": "Female",
                "profession": "Software Engineer",
                "relationship_status": "Single",
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
                "reflective_questions_asked": [
                    "What are your thoughts on the session?",
                    "What did you think of the strategies we discussed?"
                ],
                "open_ended_questions_asked": [
                    "Can you tell me more about your work-life balance challenges?",
                    "How do you think we can work together to improve your time management skills?"
                ],
                "emotional_lift": "Client felt more relaxed after the session",
                "call_quality": 90
            }
        }


UserIdentityLiteral = Literal[
    "client",
    "counselor",
    "unknown"
]


class StructuredIdentifyUsers(BaseModel):
    """
    A Pydantic model for structured output of user identification.
    """
    speaker0: UserIdentityLiteral = Field(
        ...,
        description="The role of speaker0 (client, counselor, or unknown)"
    )
    speaker1: UserIdentityLiteral = Field(
        ...,
        description="The role of speaker1 (client, counselor, or unknown)"
    )
    model_config = {
        "json_schema_extra": {
            "example": {
                "speaker0": "client",
                "speaker1": "counselor"
            }
        }
    }
