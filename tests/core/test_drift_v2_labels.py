"""Drift judge v2 labels — schema contract and rubric coverage.

The judge is only allowed to emit booleans, enum choices and counts; every rate
is computed downstream in SQL. These tests pin that contract, because the
temptation when adding a label is to ask the model for a score — and a score
baked into 17k judged turns cannot be re-weighted without re-judging them all.

Imports are limited to the drift schema/prompt modules so this runs without the
service's heavier dependencies.
"""

import pytest
from pydantic import ValidationError

from app.core.drift.prompt import DEFAULT_JUDGE_RUBRIC
from app.core.drift.schemas import JudgeOutput, PerTurnJudgment

V1_TURN = {
    "turn_index": 0,
    "coherence": "fully_coherent",
    "topic_label": "on_topic",
    "in_character": True,
    "counselor_utterance_garbled": "none",
    "stt_error_type": "none",
    "ai_reply_failure_mode": "none",
    "root_attribution": "none",
    "reasoning": "Clean turn.",
}


def test_v1_output_still_validates():
    """A response with no v2 labels must not fail the whole session — the
    labels are additive, and an older judge deployment has to keep working."""
    turn = PerTurnJudgment(**V1_TURN)
    assert turn.role_inversion is None
    assert turn.solutions_offered is None
    assert turn.stuck_is_appropriate is None


def test_v2_labels_round_trip():
    turn = PerTurnJudgment(
        **V1_TURN,
        role_inversion=True,
        offered_solution=False,
        solutions_offered=3,
        introduced_new_information=False,
        stuck_is_appropriate=False,
        resistance_briefed=True,
    )
    assert turn.role_inversion is True
    assert turn.solutions_offered == 3
    assert turn.stuck_is_appropriate is False


def test_absent_is_none_not_false():
    """The distinction the whole appropriate-stuckness exclusion rests on: a
    label the judge did not answer must be null, so downstream can tell "not
    observed" from "observed and negative"."""
    turn = PerTurnJudgment(**V1_TURN, introduced_new_information=False)
    assert turn.introduced_new_information is False
    assert turn.stuck_is_appropriate is None


def test_solutions_offered_is_a_count_not_a_score():
    """Counts are integers. If this ever accepts a float the judge has been
    asked to rate something."""
    turn = PerTurnJudgment(**V1_TURN, solutions_offered=0)
    assert turn.solutions_offered == 0
    assert isinstance(turn.solutions_offered, int)

    with pytest.raises(ValidationError):
        PerTurnJudgment(**V1_TURN, solutions_offered="many")


def test_no_scalar_score_fields_exist():
    """Guard the contract by name: nothing on this model may look like a
    score, rating or rate."""
    banned = ("score", "rating", "rate", "severity_weight", "quality")
    for field in PerTurnJudgment.model_fields:
        assert not any(b in field.lower() for b in banned), (
            f"{field} looks like a scalar judgement — the judge labels, "
            "SQL computes"
        )


def test_judge_output_wraps_per_turn():
    out = JudgeOutput(per_turn=[PerTurnJudgment(**V1_TURN)])
    assert len(out.per_turn) == 1


@pytest.mark.parametrize(
    "label",
    [
        "role_inversion",
        "offered_solution",
        "solutions_offered",
        "resistance_briefed",
        "introduced_new_information",
        "stuck_is_appropriate",
    ],
)
def test_rubric_documents_every_v2_label(label):
    """A field the rubric never mentions is a field the model will not fill."""
    assert label in DEFAULT_JUDGE_RUBRIC


def test_rubric_keeps_the_stuckness_distinction():
    """The rubric must keep telling the judge that being stuck is sometimes
    correct. Losing this line turns the progression metric into pressure
    toward an agreeable actor, which is the failure clienthood measures."""
    assert "not automatically a failure" in DEFAULT_JUDGE_RUBRIC
    assert "resistant client should NOT yield" in DEFAULT_JUDGE_RUBRIC


def test_rubric_excludes_client_questions_from_inversion():
    """The commonest way to get role inversion wrong is to count a client
    asking for help as the actor taking the counsellor's chair."""
    assert "is NOT inversion" in DEFAULT_JUDGE_RUBRIC
