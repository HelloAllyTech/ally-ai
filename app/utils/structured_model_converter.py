from functools import singledispatch
from typing import Optional

from app.core.text_generations.structured_output_models import (
    DominantFeelingLiteral,
    StructuredSummaryNote,
    StructuredTag,
)
from app.schemas.summary import (
    SummaryNoteAndTagsResponse,
    Tag,
)

# Module-level constant for level mapping.
_LEVEL2_TO_LEVEL1: dict[str, str] = {
    "Let Down": "Angry",
    "Humiliated": "Angry",
    "Bitter": "Angry",
    "Mad": "Angry",
    "Aggressive": "Angry",
    "Frustrated": "Angry",
    "Distant": "Angry",
    "Critical": "Angry",
    "Scared": "Fearful",
    "Anxious": "Fearful",
    "Insecure": "Fearful",
    "Weak": "Fearful",
    "Rejected": "Fearful",
    "Threatened": "Fearful",
    "Bored": "Bad",
    "Busy": "Bad",
    "Stressed": "Bad",
    "Tired": "Bad",
    "Startled": "Surprised",
    "Confused": "Surprised",
    "Amazed": "Surprised",
    "Excited": "Surprised",
    "Playful": "Happy",
    "Content": "Happy",
    "Interested": "Happy",
    "Proud": "Happy",
    "Accepted": "Happy",
    "Powerful": "Happy",
    "Peaceful": "Happy",
    "Trusting": "Happy",
    "Optimistic": "Happy",
    "Disapproving": "Disgusted",
    "Disappointed": "Disgusted",
    "Awful": "Disgusted",
    "Repelled": "Disgusted",
    "Hurt": "Sad",
    "Depressed": "Sad",
    "Guilty": "Sad",
    "Despair": "Sad",
    "Vulnerable": "Sad",
    "Lonely": "Sad"
}


def fetch_levels_of_feeling_from_dominant_feelings(
        dominant_feeling: Optional[DominantFeelingLiteral]
) -> tuple[str, str, str] | tuple[None, None, None]:
    """
    Extracts the three levels of feeling from a dominant feeling string.

    The input should be a string formatted as "inner_feeling:middle_feeling".
    The function derives the outer feeling (level1) based on a mapping from the middle feeling (level2).

    Parameters:
        dominant_feeling (Optional[DominantFeelingLiteral]): A string in the format "inner_feeling:middle_feeling".
            If None, the function returns (None, None, None).

    Returns:
        tuple[str, str, str] or tuple[None, None, None]:
            A tuple (level1, level2, level3) where:
              - level1 is derived from level2 using a predefined mapping,
              - level2 is the trimmed middle feeling,
              - level3 is the trimmed inner feeling.
            Returns (None, None, None) if the input is None.

    Raises:
        ValueError: If the input format is invalid or if level2 is not recognized.
    """
    if not dominant_feeling:
        return None, None, None

    # Split the input using ":" as the separator.
    parts = dominant_feeling.split(":", 1)
    if len(parts) != 2:
        raise ValueError("Input string must be in the format 'level3:level2'.")

    level3, level2 = (part.strip() for part in parts)

    # Look up the corresponding level1 from the mapping.
    level1 = _LEVEL2_TO_LEVEL1.get(level2)
    if level1 is None:
        raise ValueError(f"Level2 value '{level2}' is not recognized.")

    return level1, level2, level3


def format_dominant_feeling(level1: str, level2: str, level3: str) -> str | None:
    """
    Formats the dominant feeling string in the format "Level 1 > Level 2 > Level 3".

    Parameters:
        level1 (str): The outer feeling.
        level2 (str): The middle feeling.
        level3 (str): The inner feeling.

    Returns:
        str | None: The formatted dominant feeling string, or None if any level is None.
    """
    if None in (level1, level2, level3):
        return None
    return f"{level1} > {level2} > {level3}"


def format_list_to_bullet_points(items: list[str]) -> str:
    """
    Formats a list of strings as a bullet-point string.
    Parameters:
        items (list[str]): The list of strings to format.
    Returns:
        str: The formatted bullet-point string.
    """
    return "\n".join(f"- {item}" for item in items)


@singledispatch
def structured_output_model_to_rest[T, U](sop_model: U) -> T:
    """
    Convert a structured output model to a REST response model.

    This generic function converts a structured output model (of type U) into its corresponding
    REST response model (of type T). The conversion behavior is determined by the actual type of
    the input model, with specific implementations registered via singledispatch.

    Type Parameters:
        T: The target REST response model type.
        U: The source structured output model type.

    Parameters:
        sop_model (U): The structured output model instance to be converted.

    Returns:
        T: The REST response model produced by the conversion.

    Raises:
        NotImplementedError: If conversion for the type of `sop_model` is not implemented.
    """
    raise NotImplementedError(f"Conversion for type {type(sop_model)} is not implemented.")


@structured_output_model_to_rest.register
def _convert_none(_: None) -> None:
    """
    Handling cases where model is None.
    """
    return None


@structured_output_model_to_rest.register
def _convert_structured_tag(sop_model: StructuredTag) -> Tag:
    """
    Convert a StructuredTag to a Tag.
    """
    return Tag(
        tag=sop_model.tag,
        positivity_rating=sop_model.positivity_rating,
    )


@structured_output_model_to_rest.register
def _convert_structured_summary_note(sop_model: StructuredSummaryNote) -> SummaryNoteAndTagsResponse:
    """
    Convert a StructuredSummaryNote to a SummaryNoteAndTagsResponse.
    """
    # Convert tags
    tags = [Tag(tag=tag.tag, positivity_rating=tag.positivity_rating) for tag in sop_model.tags]

    # Format dominant feelings
    if sop_model.dominant_feelings:
        dominant_feelings = [
            format_dominant_feeling(*fetch_levels_of_feeling_from_dominant_feelings(feeling))
            for feeling in sop_model.dominant_feelings
        ]
    else:
        dominant_feelings = []

    # Create flattened response
    return SummaryNoteAndTagsResponse(
        call_id=sop_model.call_id,
        call_duration=sop_model.call_duration,
        call_date=sop_model.date_of_session,
        call_time=sop_model.call_time,
        client_id=sop_model.client_id,
        counsellor=sop_model.counsellor,
        call_type=sop_model.call_type,
        age=sop_model.age,
        gender=sop_model.gender,
        profession=sop_model.profession,
        relationship_status=sop_model.relationship_status,
        languages=[
            {
                "language": "English",
                "percentage": 0.85
            }
        ],
        location=sop_model.location,
        code_of_concern=sop_model.code_of_concern,
        session_summary=sop_model.session_summary,
        counseling_process_flow=format_list_to_bullet_points(
            sop_model.counseling_process_flow) if sop_model.counseling_process_flow else None,

        key_concerns=format_list_to_bullet_points(sop_model.key_concerns) if sop_model.key_concerns else None,
        subjective_observations=format_list_to_bullet_points(
            sop_model.subjective_observations) if sop_model.subjective_observations else None,
        objective_observations=format_list_to_bullet_points(
            sop_model.objective_observations) if sop_model.objective_observations else None,
        assessment=sop_model.assessment,
        dominant_feelings=format_list_to_bullet_points(dominant_feelings),
        issues_worked_on=format_list_to_bullet_points(
            sop_model.issues_worked_on) if sop_model.issues_worked_on else None,
        key_therapeutic_techniques=format_list_to_bullet_points(
            sop_model.key_therapeutic_techniques) if sop_model.key_therapeutic_techniques else None,
        referrals_provided=format_list_to_bullet_points(
            sop_model.referrals_provided) if sop_model.referrals_provided else None,
        homework=format_list_to_bullet_points(sop_model.homework) if sop_model.homework else None,
        plan_for_next_call=format_list_to_bullet_points(
            sop_model.plan_for_next_call) if sop_model.plan_for_next_call else None,
        tags=tags,
        reflective_questions_asked=len(
            sop_model.reflective_questions_asked) if sop_model.reflective_questions_asked else 0,
        open_ended_questions_asked=len(
            sop_model.open_ended_questions_asked) if sop_model.open_ended_questions_asked else 0,
        listening_share=None,
        emotional_lift=sop_model.emotional_lift,
        call_quality=max(0, min(sop_model.call_quality, 100))
    )
