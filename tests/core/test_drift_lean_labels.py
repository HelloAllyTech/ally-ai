"""The lean backfill judge: same rubric, smaller response.

What these guard is not the model's answers — it is the two properties that
make a cheap backfill safe to mix with full live judgments:

  1. the lean prompt carries the SAME rubric text as the full one, so the two
     cannot define the added labels differently; and
  2. the lean response cannot carry a value that reads as a negative, because a
     spurious `false` would enter a rate's numerator while a NULL correctly
     leaves the turn out of the denominator entirely.

If either breaks, backfilled history and live data disagree, and that shows up
on a chart as a step change indistinguishable from a real regression.
"""

from app.core.drift.prompt import (
    DEFAULT_JUDGE_RUBRIC,
    build_judge_prompt,
    build_lean_labels_prompt,
)
from app.core.drift.schemas import LeanJudgeOutput, LeanTurnLabels

TRANSCRIPT = [
    {"role": "counselor", "text": "What brings you in today?"},
    {"role": "client", "turn_index": 0, "text": "Everything is too much lately."},
    {"role": "counselor", "text": "Tell me more about that."},
    {"role": "client", "turn_index": 1, "text": "It is the same every day."},
]


def test_lean_prompt_carries_the_same_rubric_as_the_full_judge():
    full = build_judge_prompt(TRANSCRIPT, "a tired client", "en")
    lean = build_lean_labels_prompt(TRANSCRIPT, "a tired client", "en")

    # Not "contains something similar" — the full prompt verbatim, so the label
    # definitions cannot drift between the backfill and the live path.
    assert lean.startswith(full)
    assert DEFAULT_JUDGE_RUBRIC in lean


def test_lean_prompt_carries_a_supplied_rubric_rather_than_the_default():
    # Production fetches the rubric from prompt management; the lean path must
    # use whatever the caller passed, or it would judge against stale text.
    custom = "CUSTOM RUBRIC BODY — role_inversion means something specific here."
    lean = build_lean_labels_prompt(TRANSCRIPT, "p", "en", rubric=custom)

    assert custom in lean
    assert DEFAULT_JUDGE_RUBRIC not in lean


def test_lean_prompt_asks_for_only_the_added_labels():
    lean = build_lean_labels_prompt(TRANSCRIPT, "p", "en")

    for added in (
        "role_inversion",
        "offered_solution",
        "solutions_offered",
        "resistance_briefed",
        "introduced_new_information",
        "stuck_is_appropriate",
    ):
        assert added in lean

    # The instruction naming what to drop must be present, not merely implied.
    assert "Omit coherence" in lean
    assert "LABELS ONLY" in lean


def test_absent_labels_are_null_not_false():
    # A model that declines to answer must remove the turn from the
    # denominator, never contribute a negative to the numerator.
    turn = LeanTurnLabels(turn_index=3)

    assert turn.role_inversion is None
    assert turn.offered_solution is None
    assert turn.solutions_offered is None
    assert turn.resistance_briefed is None
    assert turn.introduced_new_information is None
    assert turn.stuck_is_appropriate is None
    assert turn.reasoning is None


def test_lean_output_parses_a_partial_turn():
    out = LeanJudgeOutput.model_validate(
        {
            "per_turn": [
                {"turn_index": 0, "role_inversion": True, "reasoning": "asked the counsellor about their own view"},
                {"turn_index": 1, "introduced_new_information": False, "stuck_is_appropriate": True},
            ]
        }
    )

    assert out.per_turn[0].role_inversion is True
    # Unmentioned on turn 0 — must not have been filled in as False.
    assert out.per_turn[0].introduced_new_information is None
    assert out.per_turn[1].reasoning is None


def test_lean_schema_does_not_require_the_v1_fields():
    # Inheriting from PerTurnJudgment would drag coherence/topic_label back in
    # as required output — the exact cost this path exists to avoid.
    fields = set(LeanTurnLabels.model_fields)

    assert "coherence" not in fields
    assert "topic_label" not in fields
    assert "ai_reply_failure_mode" not in fields


class _FakeUsage:
    prompt_token_count = 2400
    candidates_token_count = 700
    total_token_count = 3100


class _FakeResponse:
    usage_metadata = _FakeUsage()

    def __init__(self, parsed):
        self.parsed = parsed


def test_judge_session_labels_only_actually_runs(monkeypatch):
    """Exercise the function, not just the pieces it uses.

    This exists because the first deployed version of this path raised
    NameError on `build_lean_labels_prompt` — the import was missing — while
    898 tests passed, because every one of them tested the prompt builder and
    the schema DIRECTLY and none ever called the judge. A missing import is
    invisible until call time, so the call itself has to be under test.
    """
    from app.core.drift import judge as judge_mod

    captured = {}

    class _FakeModels:
        def generate_content(self, model, contents, config):
            captured["prompt"] = contents
            captured["schema"] = config.response_schema
            return _FakeResponse(
                LeanJudgeOutput(
                    per_turn=[
                        LeanTurnLabels(turn_index=0, role_inversion=False),
                        LeanTurnLabels(turn_index=1, introduced_new_information=True),
                    ]
                )
            )

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setattr(judge_mod, "_get_client", lambda: _FakeClient())

    per_turn = judge_mod.judge_session_labels_only(
        TRANSCRIPT, persona="a tired client", language="en"
    )

    assert [t.turn_index for t in per_turn] == [0, 1]
    assert per_turn[0].role_inversion is False
    # The response schema must be the LEAN one, or the saving evaporates.
    assert captured["schema"] is LeanJudgeOutput
    assert "LABELS ONLY" in captured["prompt"]


def test_judge_session_labels_only_raises_on_unparsable_output(monkeypatch):
    from app.core.drift import judge as judge_mod

    class _FakeModels:
        def generate_content(self, model, contents, config):
            return _FakeResponse(None)

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setattr(judge_mod, "_get_client", lambda: _FakeClient())

    # Must fail loudly so the backfill logs and counts it, rather than merging
    # an empty label set over a session and calling it judged.
    import pytest

    with pytest.raises(RuntimeError, match="no parsable output"):
        judge_mod.judge_session_labels_only(TRANSCRIPT, persona="p", language="en")
