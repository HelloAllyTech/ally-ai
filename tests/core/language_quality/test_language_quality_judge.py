"""Unit tests for the language-quality judge's deterministic parts:
post-processing (validation / layer derivation / conditioning) and the prompt
builder. The LLM call itself is not exercised here."""

import pytest

from app.core.language_quality.judge import process_output
from app.core.language_quality.prompt import (
    DEFAULT_JUDGE_RUBRIC,
    LanguageEvalParams,
    ScenarioStyleParams,
    build_judge_prompt,
)
from app.core.language_quality.schemas import (
    DIMENSION_CATEGORIES,
    DIMENSION_LAYER,
    ErrorAnnotation,
    TurnJudgment,
)


def _err(dimension, category, severity="major", basis="input_clean"):
    return ErrorAnnotation(
        dimension=dimension,
        category=category,
        severity=severity,
        evidence_quote="example span",
        isolation_basis=basis,
        reasoning="test",
    )


class TestProcessOutput:
    def test_layer_derived_from_dimension(self):
        turns = [
            TurnJudgment(
                turn_index=0,
                input_garbled="none",
                errors=[
                    _err("understanding", "ignored_context"),
                    _err("fluency", "grammar"),
                    _err("register", "too_formal_diglossia"),
                ],
            )
        ]
        result = process_output(turns)
        layers = [e.layer for e in result.per_turn[0].errors]
        assert layers == ["comprehension", "content", "appropriateness"]

    def test_invalid_category_for_dimension_is_dropped_and_counted(self):
        turns = [
            TurnJudgment(
                turn_index=0,
                input_garbled="none",
                errors=[
                    _err("register", "grammar"),  # grammar belongs to fluency
                    _err("register", "too_casual"),
                ],
            )
        ]
        result = process_output(turns)
        assert result.dropped_annotations == 1
        assert [e.category for e in result.per_turn[0].errors] == ["too_casual"]

    def test_conditioning_only_on_understanding_and_adequacy(self):
        turns = [
            TurnJudgment(
                turn_index=3,
                input_garbled="partial",
                errors=[
                    _err("adequacy", "off_topic", basis="input_garbled"),
                    _err("understanding", "misinterpreted_intent", basis="input_garbled"),
                    _err("fluency", "grammar"),
                    _err("register", "too_formal_diglossia"),
                ],
            )
        ]
        result = process_output(turns)
        by_dim = {e.dimension: e.conditioned_out for e in result.per_turn[0].errors}
        assert by_dim["adequacy"] is True
        assert by_dim["understanding"] is True
        assert by_dim["fluency"] is False
        assert by_dim["register"] is False

    def test_clean_input_never_conditions_out(self):
        turns = [
            TurnJudgment(
                turn_index=0,
                input_garbled="none",
                errors=[_err("adequacy", "hallucination")],
            )
        ]
        result = process_output(turns)
        assert result.per_turn[0].errors[0].conditioned_out is False

    def test_turns_sorted_and_counted(self):
        turns = [
            TurnJudgment(turn_index=2, input_garbled="none", errors=[]),
            TurnJudgment(turn_index=0, input_garbled="severe", errors=[]),
        ]
        result = process_output(turns)
        assert result.turns_judged == 2
        assert [t.turn_index for t in result.per_turn] == [0, 2]

    def test_evidence_quote_clamped(self):
        long_quote = "x" * 500
        err = _err("fluency", "grammar")
        err = err.model_copy(update={"evidence_quote": long_quote})
        turns = [TurnJudgment(turn_index=0, input_garbled="none", errors=[err])]
        result = process_output(turns)
        assert len(result.per_turn[0].errors[0].evidence_quote) <= 200


class TestTypologyConsistency:
    def test_every_dimension_has_layer_and_categories(self):
        assert set(DIMENSION_LAYER) == set(DIMENSION_CATEGORIES)

    def test_categories_do_not_overlap_across_dimensions(self):
        seen = set()
        for cats in DIMENSION_CATEGORIES.values():
            assert not (cats & seen)
            seen |= cats


class TestBuildJudgePrompt:
    TRANSCRIPT = [
        {"role": "counselor", "text": "How are you feeling?"},
        {"role": "client", "text": "I feel very alone.", "turn_index": 0},
    ]

    def test_contains_rubric_persona_and_turn_tags(self):
        prompt = build_judge_prompt(self.TRANSCRIPT, "A distressed teen", "ta-IN")
        assert DEFAULT_JUDGE_RUBRIC.splitlines()[0] in prompt
        assert "A distressed teen" in prompt
        assert "[turn 0] AI_CLIENT: I feel very alone." in prompt
        assert "COUNSELOR: How are you feeling?" in prompt

    def test_unknown_params_render_as_unknown(self):
        prompt = build_judge_prompt(self.TRANSCRIPT, "p", "kn-IN")
        assert "TARGET VARIETY: unknown" in prompt
        assert "DIGLOSSIA APPLIES: unknown" in prompt
        assert "ALLOWED FILLERS: none configured" in prompt

    def test_configured_params_rendered(self):
        prompt = build_judge_prompt(
            self.TRANSCRIPT,
            "p",
            "kn-IN",
            language_params=LanguageEvalParams(
                language_label="Kannada (India)",
                target_variety="colloquial spoken Kannada",
                diglossia=True,
                code_switch_partners=["en"],
            ),
            style_params=ScenarioStyleParams(
                register_directive_configured=True,
                style_exemplars_configured=False,
                allowed_fillers=["andre"],
                engine="SIMULATION",
                locked_content_exists=False,
            ),
        )
        assert "SESSION LANGUAGE: Kannada (India) (kn-IN)" in prompt
        assert "TARGET VARIETY: colloquial spoken Kannada" in prompt
        assert "DIGLOSSIA APPLIES: yes" in prompt
        assert "CODE-SWITCH PARTNERS: en" in prompt
        assert "REGISTER DIRECTIVE CONFIGURED: yes" in prompt
        assert "STYLE EXEMPLARS CONFIGURED: no" in prompt
        assert "ALLOWED FILLERS: andre" in prompt
        assert "ENGINE: SIMULATION" in prompt
        assert "LOCKED CONTENT EXISTS: no" in prompt


class TestSchemaValidation:
    def test_invalid_severity_rejected(self):
        with pytest.raises(Exception):
            _err("fluency", "grammar", severity="fatal")

    def test_invalid_dimension_rejected(self):
        with pytest.raises(Exception):
            _err("naturalness", "grammar")
