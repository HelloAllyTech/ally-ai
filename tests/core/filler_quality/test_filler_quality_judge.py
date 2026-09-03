"""Unit tests for the thinking-filler judge's deterministic parts: repeat
detection, post-processing (validation / joining / the acceptability rule) and
the prompt builder. The LLM call itself is not exercised here."""

import pytest

from app.core.filler_quality.judge import compute_repeats, process_output
from app.core.filler_quality.prompt import (
    DEFAULT_FILLER_RUBRIC,
    FillerStyleParams,
    build_judge_prompt,
)
from app.core.filler_quality.schemas import (
    FillerFinding,
    FillerJudgment,
    FillerObservation,
)


def _obs(
    turn_index, filler_text, learner="I've been struggling.", reply="Tell me.", **kw
):
    return FillerObservation(
        turn_index=turn_index,
        learner_utterance=learner,
        filler_text=filler_text,
        reply_text=reply,
        **kw,
    )


def _finding(dimension, category, evidence="hmm"):
    return FillerFinding(
        dimension=dimension,
        category=category,
        severity="major",
        evidence_quote=evidence,
        reasoning="test",
    )


def _judgement(turn_index, findings=None):
    """A clean filler by default — which is what most of them are."""
    return FillerJudgment(turn_index=turn_index, findings=findings or [])


class TestComputeRepeats:
    def test_first_use_is_never_a_repeat(self):
        repeats = compute_repeats([_obs(0, "Hmm"), _obs(1, "I see")])
        assert repeats[0] == (False, None)
        assert repeats[1] == (False, None)

    def test_repeat_inside_the_window_is_flagged_with_its_distance(self):
        observations = [_obs(0, "Hmm"), _obs(1, "I see"), _obs(2, "Hmm")]
        repeats = compute_repeats(observations, window_plays=12)
        assert repeats[2] == (True, 2)

    def test_repeat_outside_the_window_is_not_flagged(self):
        """A phrase coming back after a long gap is variety, not repetition."""
        observations = (
            [_obs(0, "Hmm")]
            + [_obs(i, f"phrase {i}") for i in range(1, 6)]
            + [_obs(6, "Hmm")]
        )
        repeats = compute_repeats(observations, window_plays=3)
        assert repeats[6] == (False, 6)

    def test_window_is_counted_in_plays_not_turns(self):
        """One turn can play two fillers (the continuation), so a window in
        turns would be half as wide here as it is in the player."""
        observations = [_obs(5, "Hmm"), _obs(5, "Well"), _obs(6, "Hmm")]
        repeats = compute_repeats(observations, window_plays=2)
        # Two PLAYS apart, though only one turn apart.
        assert repeats[6] == (True, 2)

    def test_identity_ignores_case_and_spacing(self):
        repeats = compute_repeats([_obs(0, "Hmm"), _obs(1, "  hmm ")])
        assert repeats[1][0] is True


class TestProcessOutput:
    def test_joins_the_annotation_to_its_observation(self):
        observations = [_obs(0, "Hmm", source="in_turn", filler_type="hesitation")]
        result = process_output([_judgement(0)], observations)

        assert result.fillers_judged == 1
        row = result.per_filler[0]
        # The model is never asked to echo these back, so they must be joined.
        assert row.filler_text == "Hmm"
        assert row.source == "in_turn"
        assert row.filler_type == "hesitation"

    def test_a_clean_filler_has_no_findings(self):
        """Most fillers are fine, and the schema must make that the cheap case."""
        result = process_output([_judgement(0)], [_obs(0, "Hmm")])
        assert result.per_filler[0].findings == []
        assert result.dropped_annotations == 0

    def test_findings_survive_with_their_dimension(self):
        result = process_output(
            [
                _judgement(
                    0,
                    [
                        _finding("context_fit", "answers_earlier_turn"),
                        _finding("safety", "committed"),
                    ],
                )
            ],
            [_obs(0, "Yes, that's right")],
        )
        categories = {f.category for f in result.per_filler[0].findings}
        assert categories == {"answers_earlier_turn", "committed"}

    def test_category_must_belong_to_its_dimension(self):
        """Mirrors the language judge: an invalid pairing is dropped, not repaired."""
        result = process_output(
            [_judgement(0, [_finding("context_fit", "persona_break")])],
            [_obs(0, "Hmm")],
        )
        assert result.per_filler[0].findings == []
        assert result.dropped_annotations == 1

    def test_generic_finding_is_conditioned_out_when_no_style_configured(self):
        """Calling a filler generic on a character who was never given a voice
        blames the model for a configuration gap."""
        result = process_output(
            [_judgement(0, [_finding("character_fit", "generic_for_character")])],
            [_obs(0, "Hmm")],
            style_configured=False,
        )
        finding = result.per_filler[0].findings[0]
        # Kept, because it says something true about the scenario — but flagged,
        # so it is not counted against the model.
        assert finding.conditioned_out is True

    def test_generic_finding_counts_when_style_was_configured(self):
        result = process_output(
            [_judgement(0, [_finding("character_fit", "generic_for_character")])],
            [_obs(0, "Hmm")],
            style_configured=True,
        )
        assert result.per_filler[0].findings[0].conditioned_out is False

    def test_other_findings_are_never_conditioned_out(self):
        result = process_output(
            [_judgement(0, [_finding("safety", "committed")])],
            [_obs(0, "Yes")],
            style_configured=False,
        )
        assert result.per_filler[0].findings[0].conditioned_out is False

    def test_unknown_turn_is_dropped_and_counted(self):
        result = process_output([_judgement(99)], [_obs(0, "Hmm")])
        assert result.per_filler == []
        assert result.dropped_annotations == 1

    def test_evidence_is_clamped(self):
        result = process_output(
            [_judgement(0, [_finding("safety", "committed", evidence="x" * 1000)])],
            [_obs(0, "Hmm")],
        )
        assert len(result.per_filler[0].findings[0].evidence_quote) == 240

    def test_distinct_phrase_ratio_is_computed_from_what_played(self):
        observations = [_obs(0, "Hmm"), _obs(1, "Hmm"), _obs(2, "I see"), _obs(3, "Ah")]
        result = process_output([_judgement(i) for i in range(4)], observations)
        # 3 distinct of 4 played.
        assert result.distinct_phrase_ratio == pytest.approx(0.75)

    def test_repeat_facts_are_attached_from_code_not_the_model(self):
        observations = [_obs(0, "Hmm"), _obs(1, "Hmm")]
        result = process_output([_judgement(0), _judgement(1)], observations)
        by_turn = {row.turn_index: row for row in result.per_filler}
        assert by_turn[0].repeated_within_window is False
        assert by_turn[1].repeated_within_window is True
        assert by_turn[1].plays_since_last_use == 1

    def test_rows_come_back_in_turn_order(self):
        observations = [_obs(0, "a"), _obs(1, "b"), _obs(2, "c")]
        result = process_output(
            [_judgement(2), _judgement(0), _judgement(1)], observations
        )
        assert [row.turn_index for row in result.per_filler] == [0, 1, 2]

    def test_no_observations_yields_an_empty_result(self):
        result = process_output([], [])
        assert result.fillers_judged == 0
        assert result.distinct_phrase_ratio is None


class TestPromptBuilder:
    def test_includes_the_rubric_and_every_observation(self):
        prompt = build_judge_prompt(
            [_obs(0, "Hmm", learner="I lost my job."), _obs(1, "I see")],
            persona="A withdrawn client.",
            language="en-US",
        )
        assert DEFAULT_FILLER_RUBRIC.split("\n")[0] in prompt
        assert "[turn 0]" in prompt and "[turn 1]" in prompt
        assert "I lost my job." in prompt
        assert "A withdrawn client." in prompt

    def test_reply_is_shown_so_commitment_is_decidable(self):
        """A filler is only unsafe relative to the reply that followed it."""
        prompt = build_judge_prompt(
            [_obs(0, "Hmm", reply="No, I never said that.")],
            persona="",
            language="en",
        )
        assert "No, I never said that." in prompt

    def test_absent_style_is_stated_not_omitted(self):
        """Silence would read as a model failure on an unconfigured scenario."""
        prompt = build_judge_prompt([_obs(0, "Hmm")], persona="", language="en")
        assert "none configured" in prompt

    def test_configured_style_is_shown(self):
        prompt = build_judge_prompt(
            [_obs(0, "Hmm")],
            persona="",
            language="ta-IN",
            style_params=FillerStyleParams(
                language_label="Tamil (India)",
                style_exemplars=["I don't know, really."],
                allowed_fillers=["ஆமா", "சரி"],
            ),
        )
        assert "Tamil (India)" in prompt
        assert "I don't know, really." in prompt
        assert "ஆமா" in prompt

    def test_a_custom_rubric_replaces_the_default(self):
        prompt = build_judge_prompt(
            [_obs(0, "Hmm")], persona="", language="en", rubric="CUSTOM RUBRIC"
        )
        assert prompt.startswith("CUSTOM RUBRIC")
