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
from app.core.drift.schemas import (
    JudgeOutput,
    LiveJudgeOutput,
    LiveTurnJudgment,
    PerTurnJudgment,
)

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


# ---- the live path must compel an answer, exactly as the lean one does -----
#
# PerTurnJudgment stays lenient on purpose (a stored v1 row is read back through
# it), so leniency has to stop at the schema the LIVE judge hands to Gemini.
# Without that, the live path repeats the lean judge's production failure: over
# 155 turns it emitted role_inversion on 2, both true, and a rate that counts
# only turns carrying the label read 100%.


def test_live_turn_judgment_requires_the_five_unconditional_labels():
    with pytest.raises(ValidationError):
        LiveTurnJudgment(**V1_TURN)

    clean = LiveTurnJudgment(
        **V1_TURN,
        role_inversion=False,
        offered_solution=False,
        solutions_offered=0,
        resistance_briefed=True,
        introduced_new_information=True,
    )
    assert clean.role_inversion is False
    assert clean.solutions_offered == 0
    # Genuinely conditional — the rubric defines no answer on a turn that
    # advanced, so it must not be invented.
    assert clean.stuck_is_appropriate is None


@pytest.mark.parametrize(
    "omitted",
    [
        "role_inversion",
        "offered_solution",
        "solutions_offered",
        "resistance_briefed",
        "introduced_new_information",
    ],
)
def test_live_response_rejects_a_turn_missing_any_unconditional_label(omitted):
    answered = {
        "role_inversion": False,
        "offered_solution": False,
        "solutions_offered": 0,
        "resistance_briefed": True,
        "introduced_new_information": True,
    }
    answered.pop(omitted)

    with pytest.raises(ValidationError):
        LiveJudgeOutput.model_validate({"per_turn": [{**V1_TURN, **answered}]})


def test_live_turn_judgment_still_carries_the_v1_fields():
    """Unlike the lean schema, the live judge emits the whole row."""
    fields = set(LiveTurnJudgment.model_fields)

    assert {"coherence", "topic_label", "root_attribution"} <= fields
    # And it stays assignable to the result model the endpoint returns.
    assert issubclass(LiveTurnJudgment, PerTurnJudgment)


def test_rubric_compels_an_answer_on_every_turn():
    """Prose alone did not hold in the lean judge, but the rubric must still
    say it: a model told to omit reasoning on clean turns generalises that to
    every label whose answer is no."""
    assert "EVERY AI-CLIENT TURN" in DEFAULT_JUDGE_RUBRIC
    assert "never a missing field" in DEFAULT_JUDGE_RUBRIC


class _FakeUsage:
    prompt_token_count = 2400
    candidates_token_count = 3800
    total_token_count = 6200


class _FakeResponse:
    usage_metadata = _FakeUsage()

    def __init__(self, parsed):
        self.parsed = parsed


def test_judge_session_hands_gemini_the_strict_schema(monkeypatch):
    """The enforcement point. Gemini marks a field required only when the
    response_schema does, so the live call has to pass the strict model — a
    lenient one lets the labels come back only where they fired."""
    from app.core.drift import judge as judge_mod

    captured = {}

    fully_answered = dict(
        V1_TURN,
        role_inversion=False,
        offered_solution=False,
        solutions_offered=0,
        resistance_briefed=True,
        introduced_new_information=True,
    )

    class _FakeModels:
        def generate_content(self, model, contents, config):
            captured["schema"] = config.response_schema
            return _FakeResponse(
                LiveJudgeOutput(per_turn=[LiveTurnJudgment(**fully_answered)])
            )

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setattr(judge_mod, "_get_client", lambda: _FakeClient())

    result = judge_mod.judge_session(
        [{"role": "client", "turn_index": 0, "text": "hi"}],
        persona="a tired client",
        language="en",
    )

    assert captured["schema"] is LiveJudgeOutput
    # The rollup still comes out of the same per-turn rows.
    assert result.session.drifted is False
    assert result.per_turn[0].role_inversion is False
