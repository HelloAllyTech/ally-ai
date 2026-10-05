"""Tests for HelplineCopilotService.

The load-bearing behaviours are the ones that keep a talker's words out of logs, keep a
bad draft from reaching a listener, and make a failure visible instead of silently
looking like "no risk": the signal is always a real substring of the message and goes to
the PHI log only; suggestions are filtered, trimmed and capped after generation; a nudge
exists only when asked for; and a failed call reports itself rather than raising.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import settings
from app.core.helpline.schemas import (
    SKILL_KEYS,
    HelplineCopilotTurn,
    HelplineRiskVerdict,
    HelplineSuggestion,
)
from app.core.helpline.service import (
    MAX_NUDGE_CHARS,
    MAX_SUGGESTION_CHARS,
    HelplineCopilotService,
    resolve_signal,
    truncate,
)
from app.core.llm_usage.tasks import LLMTask

MODULE = "app.core.helpline.service"

META = {"provider": "openai", "model": "gpt-4o-mini"}

TALKER_MESSAGE = "Honestly I just feel like there is no point anymore."
TURN_MESSAGES = [
    {"role": "talker", "content": "I failed my exam and I can't tell my parents."},
    {"role": "listener", "content": "That sounds like a lot to carry."},
    {"role": "talker", "content": "Yeah. I just feel so alone with it."},
]


@pytest.fixture
def service():
    return HelplineCopilotService()


def stub_generate(*results, meta=META):
    """An AsyncMock for generate_structured yielding (parsed, meta) in order."""
    return AsyncMock(side_effect=[(r, dict(meta)) for r in results])


def suggestion(text="It sounds like you're carrying a lot.", skill_key="empathy"):
    return HelplineSuggestion(text=text, skill_key=skill_key)


def turn(*suggestions, stage="Understand", nudge=""):
    return HelplineCopilotTurn(stage=stage, nudge=nudge, suggestions=list(suggestions))


class TestResolveSignal:
    """`signal` must be a verbatim substring of the message — ally-be stores only its
    offsets, so anything that is not literally in the message cannot be located."""

    def test_an_exact_phrase_is_kept(self):
        assert resolve_signal("no point anymore", TALKER_MESSAGE) == "no point anymore"

    def test_surrounding_whitespace_is_trimmed(self):
        assert resolve_signal("  no point anymore \n", TALKER_MESSAGE) == (
            "no point anymore"
        )

    def test_a_case_or_whitespace_variant_is_returned_as_the_messages_own_text(self):
        message = "I can't\ndo this  anymore, it's too much"

        assert resolve_signal("I CAN'T DO THIS ANYMORE", message) == (
            "I can't\ndo this  anymore"
        )

    def test_a_paraphrase_keeps_the_longest_matching_substring(self):
        message = "I want to end my life tonight, I have everything ready"

        result = resolve_signal("wants to end my life", message)

        assert result in message
        assert "end my life" in result

    def test_a_paraphrase_sharing_nothing_useful_is_blanked(self):
        assert resolve_signal("suicidal ideation", "my head hurts a lot") == ""

    def test_a_stray_fragment_is_blanked_rather_than_shown_as_a_signal(self):
        # "die" alone would read as a different message than the one that was sent.
        assert (
            resolve_signal("I am planning to die soon", "i feel fine, nice day") == ""
        )

    def test_an_empty_signal_or_message_is_empty(self):
        assert resolve_signal("", TALKER_MESSAGE) == ""
        assert resolve_signal("no point", "") == ""

    def test_a_long_signal_is_clipped_to_120_and_stays_a_substring(self):
        message = "x " * 200

        result = resolve_signal(message.strip(), message)

        assert len(result) == 120
        assert result in message

    @pytest.mark.parametrize(
        "message,signal",
        [
            ("मैं अब और जीना नहीं चाहता, सब खत्म कर दूंगा", "सब खत्म कर दूंगा"),
            ("ab jeene ka mann nahi karta", "jeene ka mann nahi karta"),
            ("நான் இனி வாழ விரும்பவில்லை", "வாழ விரும்பவில்லை"),
        ],
    )
    def test_indic_and_romanised_signals_survive_unchanged(self, message, signal):
        assert resolve_signal(signal, message) == signal


class TestTruncate:
    def test_short_text_is_untouched(self):
        assert truncate("  fine as it is  ", 50) == "fine as it is"

    def test_cuts_on_a_sentence_end_when_there_is_one_past_halfway(self):
        first = "That sounds really hard, and I can hear how long it has gone on."
        text = first + " and it keeps going " * 10

        assert truncate(text, 100) == first

    def test_a_sentence_end_before_the_halfway_mark_is_not_worth_cutting_at(self):
        text = "Hard. " + "and it keeps going " * 10

        result = truncate(text, 100)

        assert result != "Hard."
        assert result.endswith("…")

    def test_otherwise_cuts_on_a_word_with_an_ellipsis_and_stays_within_the_limit(self):
        text = "word " * 100

        result = truncate(text, 40)

        assert len(result) <= 40
        assert result.endswith("…")
        assert not result.rstrip("…").endswith("wor")

    def test_a_danda_counts_as_a_sentence_end(self):
        first = "आप बहुत थके हुए लगते हैं, और फिर भी आप यहाँ हैं।"
        text = first + " फिर भी आप यहाँ हैं" * 10

        assert len(first) >= 30
        assert truncate(text, 60) == first

    def test_never_exceeds_the_limit(self):
        for limit in (10, 40, 240, 300):
            assert len(truncate("abcdefghij " * 80, limit)) <= limit


class TestClassifyRisk:
    @pytest.mark.asyncio
    async def test_returns_the_verdict_with_a_verbatim_signal(self, service):
        gen = stub_generate(
            HelplineRiskVerdict(
                is_crisis=True,
                confidence=0.72,
                signal="no point anymore",
                subject="SELF",
            )
        )
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.classify_risk(TALKER_MESSAGE)

        assert result == {
            "is_crisis": True,
            "confidence": pytest.approx(0.72),
            "signal": "no point anymore",
            "subject": "SELF",
            "failed": False,
            "provider": "openai",
            "model": "gpt-4o-mini",
        }

    @pytest.mark.asyncio
    async def test_a_low_confidence_crisis_is_still_a_crisis(self, service):
        # No threshold is applied here. ally-be holds the per-org one; re-gating here
        # would silently undo the prompt's "true when unsure" instruction.
        gen = stub_generate(
            HelplineRiskVerdict(
                is_crisis=True, confidence=0.3, signal="no point", subject="UNCLEAR"
            )
        )
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.classify_risk(TALKER_MESSAGE)

        assert result["is_crisis"] is True
        assert result["confidence"] == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_not_a_crisis_has_no_signal_and_an_unclear_subject(self, service):
        # Even if the model fills them in: there is nothing to show a listener.
        gen = stub_generate(
            HelplineRiskVerdict(
                is_crisis=False, confidence=0.2, signal="no point", subject="SELF"
            )
        )
        with (
            patch(f"{MODULE}.generate_structured", gen),
            patch(f"{MODULE}.phi_logger") as mock_phi,
        ):
            mock_phi.log = AsyncMock()
            result = await service.classify_risk(TALKER_MESSAGE)

        assert result["is_crisis"] is False
        assert result["signal"] == ""
        assert result["subject"] == "UNCLEAR"
        assert result["failed"] is False
        mock_phi.log.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_signal_that_is_not_in_the_message_is_blanked(self, service):
        gen = stub_generate(
            HelplineRiskVerdict(
                is_crisis=True,
                confidence=0.8,
                signal="expresses suicidal ideation",
                subject="SELF",
            )
        )
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.classify_risk("my head hurts a lot today")

        assert result["is_crisis"] is True
        assert result["signal"] == ""

    @pytest.mark.asyncio
    async def test_a_partly_wrong_signal_keeps_the_longest_matching_substring(
        self, service
    ):
        message = "I want to end my life tonight, I have everything ready"
        gen = stub_generate(
            HelplineRiskVerdict(
                is_crisis=True,
                confidence=0.95,
                signal="wants to end my life tonight",
                subject="SELF",
            )
        )
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.classify_risk(message)

        assert result["signal"] in message
        assert "end my life tonight" in result["signal"]

    @pytest.mark.asyncio
    async def test_subject_is_normalised_and_unknown_values_become_unclear(
        self, service
    ):
        for raw, expected in [
            ("other", "OTHER"),
            (" Self ", "SELF"),
            ("THIRD_PARTY", "UNCLEAR"),
            ("", "UNCLEAR"),
        ]:
            gen = stub_generate(
                HelplineRiskVerdict(
                    is_crisis=True, confidence=0.7, signal="no point", subject=raw
                )
            )
            with patch(f"{MODULE}.generate_structured", gen):
                result = await service.classify_risk(TALKER_MESSAGE)
            assert result["subject"] == expected, raw

    @pytest.mark.asyncio
    async def test_confidence_is_clamped_to_zero_one(self, service):
        for raw, expected in [(1.7, 1.0), (-0.4, 0.0)]:
            gen = stub_generate(
                HelplineRiskVerdict(is_crisis=True, confidence=raw, signal="no point")
            )
            with patch(f"{MODULE}.generate_structured", gen):
                result = await service.classify_risk(TALKER_MESSAGE)
            assert result["confidence"] == expected

    @pytest.mark.asyncio
    async def test_a_failure_reports_itself_rather_than_raising(self, service):
        gen = AsyncMock(side_effect=RuntimeError("classifier model down"))
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.classify_risk(TALKER_MESSAGE)

        assert result["is_crisis"] is False
        # The distinction the caller needs: "looked and said no" vs "could not look".
        assert result["failed"] is True
        assert result["signal"] == ""
        assert result["subject"] == "UNCLEAR"

    @pytest.mark.asyncio
    async def test_a_failure_does_not_log_the_exception_text(self, service):
        # A validation error quotes its input, which here is a talker's message.
        gen = AsyncMock(side_effect=RuntimeError(f"bad input: {TALKER_MESSAGE}"))
        with (
            patch(f"{MODULE}.generate_structured", gen),
            patch(f"{MODULE}.logger") as mock_logger,
        ):
            await service.classify_risk(TALKER_MESSAGE)

        rendered = _rendered_calls(mock_logger)
        assert "RuntimeError" in rendered
        assert "no point" not in rendered

    @pytest.mark.asyncio
    async def test_a_missing_template_fails_rather_than_passing_the_message(
        self, service
    ):
        with patch(f"{MODULE}.build_risk_prompt", return_value=""):
            result = await service.classify_risk(TALKER_MESSAGE)

        assert result["failed"] is True
        assert result["is_crisis"] is False

    @pytest.mark.asyncio
    async def test_an_empty_message_is_not_a_failure_and_costs_no_call(self, service):
        with patch(f"{MODULE}.generate_structured") as gen:
            result = await service.classify_risk("   ")

        gen.assert_not_called()
        assert result["failed"] is False
        assert result["is_crisis"] is False

    @pytest.mark.asyncio
    async def test_the_signal_goes_to_the_phi_log_not_the_app_log(self, service):
        """The signal is a talker's verbatim disclosure — PHI. It goes to the PHI audit
        log and never the general application logger, however useful it would be there.
        """
        gen = stub_generate(
            HelplineRiskVerdict(
                is_crisis=True,
                confidence=0.9,
                signal="no point anymore",
                subject="SELF",
            )
        )
        with (
            patch(f"{MODULE}.generate_structured", gen),
            patch(f"{MODULE}.logger") as mock_logger,
            patch(f"{MODULE}.phi_logger") as mock_phi,
        ):
            mock_phi.log = AsyncMock()
            await service.classify_risk(TALKER_MESSAGE)

        rendered = _rendered_calls(mock_logger)
        assert "no point" not in rendered
        assert TALKER_MESSAGE not in rendered
        # The confidence is fine to log, and is what makes a firing visible.
        assert "0.9" in rendered

        mock_phi.log.assert_awaited_once()
        event = mock_phi.log.call_args.args[0]
        assert event.details["signal"] == "no point anymore"
        assert event.details["confidence"] == pytest.approx(0.9)
        assert event.details["component"] == "HelplineCopilotService.classify_risk"

    @pytest.mark.asyncio
    async def test_the_phi_log_gets_the_resolved_signal_not_the_models_text(
        self, service
    ):
        gen = stub_generate(
            HelplineRiskVerdict(
                is_crisis=True,
                confidence=0.9,
                signal="NO POINT ANYMORE",
                subject="SELF",
            )
        )
        with (
            patch(f"{MODULE}.generate_structured", gen),
            patch(f"{MODULE}.phi_logger") as mock_phi,
        ):
            mock_phi.log = AsyncMock()
            result = await service.classify_risk(TALKER_MESSAGE)

        assert result["signal"] == "no point anymore"
        assert mock_phi.log.call_args.args[0].details["signal"] == "no point anymore"


class TestRiskCallShape:
    @pytest.mark.asyncio
    async def test_uses_the_task_label_the_default_model_and_a_tight_cap(self, service):
        gen = stub_generate(HelplineRiskVerdict())
        with patch(f"{MODULE}.generate_structured", gen):
            await service.classify_risk(TALKER_MESSAGE)

        kwargs = gen.call_args.kwargs
        assert kwargs["task"] == LLMTask.HELPLINE_RISK_CLASSIFY.value
        assert kwargs["task"] == "helpline_risk_classify"
        assert kwargs["schema"] is HelplineRiskVerdict
        assert kwargs["model"] == settings.HELPLINE.RISK_MODEL == "gpt-4o-mini"
        assert kwargs["max_tokens"] == settings.HELPLINE.RISK_MAX_TOKENS == 200
        assert kwargs["provider"] is None

    @pytest.mark.asyncio
    async def test_temperature_is_small_but_not_zero(self, service):
        """generate_structured's OpenAI path sends temperature only when truthy, so 0.0
        is silently dropped and the call runs at the model default."""
        gen = stub_generate(HelplineRiskVerdict())
        with patch(f"{MODULE}.generate_structured", gen):
            await service.classify_risk(TALKER_MESSAGE)

        temperature = gen.call_args.kwargs["temperature"]
        assert 0 < temperature <= 0.05

    @pytest.mark.asyncio
    async def test_a_prompt_override_sets_provider_model_and_temperature(self, service):
        prompts = {
            "ally_ai_helpline_risk_classify": {
                "prompt": "Classify: {message}",
                "provider": "anthropic",
                "model": "claude-haiku-4-5",
                "temperature": 0.3,
            }
        }
        gen = stub_generate(HelplineRiskVerdict())
        with patch(f"{MODULE}.generate_structured", gen):
            await service.classify_risk(TALKER_MESSAGE, prompts=prompts)

        kwargs = gen.call_args.kwargs
        assert kwargs["provider"] == "anthropic"
        assert kwargs["model"] == "claude-haiku-4-5"
        assert kwargs["temperature"] == pytest.approx(0.3)
        assert kwargs["prompt"] == f"Classify: {TALKER_MESSAGE}"

    @pytest.mark.asyncio
    async def test_an_override_temperature_of_exactly_zero_is_floored(self, service):
        # 0.0 would be dropped by the OpenAI path and run at the model's default.
        prompts = {"ally_ai_helpline_risk_classify": {"prompt": "x", "temperature": 0}}
        gen = stub_generate(HelplineRiskVerdict())
        with patch(f"{MODULE}.generate_structured", gen):
            await service.classify_risk(TALKER_MESSAGE, prompts=prompts)

        assert gen.call_args.kwargs["temperature"] > 0

    @pytest.mark.asyncio
    async def test_message_context_and_language_reach_the_prompt(self, service):
        gen = stub_generate(HelplineRiskVerdict())
        recent = [
            {"role": "talker", "content": "everything feels pointless"},
            {"role": "listener", "content": "Are you thinking of ending your life?"},
        ]
        with patch(f"{MODULE}.generate_structured", gen):
            await service.classify_risk("yes", recent=recent, language="hi")

        prompt = gen.call_args.kwargs["prompt"]
        assert "Talker: everything feels pointless" in prompt
        assert "Listener: Are you thinking of ending your life?" in prompt
        assert "chose Hindi" in prompt
        assert "yes" in prompt[prompt.index("BEGIN MESSAGE") :]


class TestCopilotTurn:
    @pytest.mark.asyncio
    async def test_returns_stage_suggestions_and_provenance(self, service):
        gen = stub_generate(
            turn(
                suggestion("It sounds like you're carrying this alone.", "empathy"),
                suggestion("What has been the hardest part?", "verbal"),
                stage="Understand",
            )
        )
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES)

        assert result == {
            "stage": "Understand",
            "nudge": "",
            "suggestions": [
                {
                    "text": "It sounds like you're carrying this alone.",
                    "skill_key": "empathy",
                },
                {"text": "What has been the hardest part?", "skill_key": "verbal"},
            ],
            "failed": False,
            "provider": "openai",
            "model": "gpt-4o-mini",
        }

    @pytest.mark.asyncio
    async def test_a_failure_reports_itself_rather_than_raising(self, service):
        gen = AsyncMock(side_effect=RuntimeError("model down"))
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES, include_nudge=True)

        assert result["failed"] is True
        assert result["suggestions"] == []
        assert result["nudge"] == ""
        assert result["stage"] == ""

    @pytest.mark.asyncio
    async def test_a_missing_template_fails(self, service):
        with patch(f"{MODULE}.build_turn_prompt", return_value=""):
            result = await service.copilot_turn(TURN_MESSAGES)

        assert result["failed"] is True

    @pytest.mark.asyncio
    async def test_nudge_is_stripped_when_not_requested(self, service):
        # The model may write one anyway (an override can drop the instruction); the
        # caller asked for none, and ally-be counts nudges against a per-chat budget.
        gen = stub_generate(turn(suggestion(), nudge="Try reflecting the loneliness."))
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES, include_nudge=False)

        assert result["nudge"] == ""

    @pytest.mark.asyncio
    async def test_nudge_is_returned_when_requested(self, service):
        gen = stub_generate(
            turn(suggestion(), nudge="  Try reflecting the loneliness. ")
        )
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES, include_nudge=True)

        assert result["nudge"] == "Try reflecting the loneliness."
        assert "at most 240 characters" in gen.call_args.kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_a_long_nudge_is_truncated_to_240(self, service):
        gen = stub_generate(turn(suggestion(), nudge="Reflect the feeling. " * 40))
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES, include_nudge=True)

        assert 0 < len(result["nudge"]) <= MAX_NUDGE_CHARS

    @pytest.mark.asyncio
    async def test_a_nudge_naming_a_dosage_is_dropped(self, service):
        gen = stub_generate(turn(suggestion(), nudge="Suggest 25 mg of something."))
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES, include_nudge=True)

        assert result["nudge"] == ""

    @pytest.mark.asyncio
    async def test_a_nudge_about_confidentiality_is_kept(self, service):
        # As coaching it is exactly right; only suggestions get the strict filter.
        nudge = "They asked about privacy: explain the limits, don't promise secrecy."
        gen = stub_generate(turn(suggestion(), nudge=nudge))
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES, include_nudge=True)

        assert result["nudge"] == nudge

    @pytest.mark.asyncio
    async def test_suggestions_are_capped_at_three_and_each_truncated_to_300(
        self, service
    ):
        long_text = "It sounds like this has been weighing on you. " * 20
        gen = stub_generate(
            turn(*[suggestion(f"{long_text} option {i}") for i in range(5)])
        )
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES)

        assert len(result["suggestions"]) == 3
        assert all(
            0 < len(s["text"]) <= MAX_SUGGESTION_CHARS for s in result["suggestions"]
        )

    @pytest.mark.asyncio
    async def test_unsafe_suggestions_are_dropped_and_the_rest_kept_in_order(
        self, service
    ):
        gen = stub_generate(
            turn(
                suggestion(
                    "I won't tell anyone, this stays between us.", "confidentiality"
                ),
                suggestion("It sounds like you're really alone with this.", "empathy"),
                suggestion(
                    "You have depression, and that is treatable.", "explanation"
                ),
                suggestion("You could take 10 mg of something to sleep.", "coping"),
                suggestion("What would help most right now?", "goals"),
            )
        )
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES)

        assert [s["skill_key"] for s in result["suggestions"]] == ["empathy", "goals"]
        assert result["failed"] is False

    @pytest.mark.asyncio
    async def test_if_everything_is_filtered_nothing_is_invented(self, service):
        gen = stub_generate(
            turn(suggestion("This stays between us.", "confidentiality"))
        )
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES)

        # Not a failure — the model answered; there was just nothing safe in it.
        assert result["suggestions"] == []
        assert result["failed"] is False

    @pytest.mark.asyncio
    async def test_the_safety_question_survives_the_filter(self, service):
        gen = stub_generate(
            turn(
                suggestion(
                    "That sounds incredibly painful. Are you having thoughts of "
                    "ending your life?",
                    "harm",
                ),
                suggestion("I won't tell anyone about this.", "confidentiality"),
            )
        )
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES)

        assert [s["skill_key"] for s in result["suggestions"]] == ["harm"]

    @pytest.mark.asyncio
    async def test_the_filter_is_applied_before_truncation(self, service):
        # An unsafe promise at the END of a long draft would be cut off by truncation
        # and then look clean; it must be caught on the full text.
        text = "It sounds like you're exhausted. " * 12 + "I won't tell anyone."
        gen = stub_generate(turn(suggestion(text, "empathy")))
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES)

        assert result["suggestions"] == []

    @pytest.mark.asyncio
    async def test_skill_keys_are_normalised_and_unknown_ones_drop_the_suggestion(
        self, service
    ):
        gen = stub_generate(
            turn(
                suggestion("What has that been like?", " Verbal "),
                suggestion("Who is around you?", "FAMILY"),
                suggestion("A made-up tag.", "validation"),
                suggestion("No tag at all.", ""),
            )
        )
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES)

        assert [s["skill_key"] for s in result["suggestions"]] == ["verbal", "family"]
        assert all(s["skill_key"] in SKILL_KEYS for s in result["suggestions"])

    @pytest.mark.asyncio
    async def test_blank_and_duplicate_suggestions_are_dropped(self, service):
        gen = stub_generate(
            turn(
                suggestion("   ", "empathy"),
                suggestion("What has that been like?", "verbal"),
                suggestion("what has  that been like?", "feelings"),
            )
        )
        with patch(f"{MODULE}.generate_structured", gen):
            result = await service.copilot_turn(TURN_MESSAGES)

        assert result["suggestions"] == [
            {"text": "What has that been like?", "skill_key": "verbal"}
        ]

    @pytest.mark.asyncio
    async def test_stage_is_normalised_and_unknown_becomes_empty(self, service):
        for raw, expected in [
            ("support", "Support"),
            (" CLOSE ", "Close"),
            ("Wrap", ""),
        ]:
            gen = stub_generate(turn(suggestion(), stage=raw))
            with patch(f"{MODULE}.generate_structured", gen):
                result = await service.copilot_turn(TURN_MESSAGES)
            assert result["stage"] == expected, raw

    @pytest.mark.asyncio
    async def test_no_text_is_logged_when_suggestions_are_dropped(self, service):
        gen = stub_generate(
            turn(suggestion("I won't tell anyone about the pills.", "confidentiality"))
        )
        with (
            patch(f"{MODULE}.generate_structured", gen),
            patch(f"{MODULE}.logger") as mock_logger,
        ):
            await service.copilot_turn(TURN_MESSAGES)

        rendered = _rendered_calls(mock_logger)
        assert "confidentiality" in rendered  # the reason code, as a count
        assert "pills" not in rendered


class TestTurnCallShape:
    @pytest.mark.asyncio
    async def test_uses_the_task_label_the_default_model_and_the_configured_cap(
        self, service
    ):
        gen = stub_generate(turn(suggestion()))
        with patch(f"{MODULE}.generate_structured", gen):
            await service.copilot_turn(TURN_MESSAGES)

        kwargs = gen.call_args.kwargs
        assert kwargs["task"] == LLMTask.HELPLINE_COPILOT_TURN.value
        assert kwargs["task"] == "helpline_copilot_turn"
        assert kwargs["schema"] is HelplineCopilotTurn
        assert kwargs["model"] == settings.HELPLINE.TURN_MODEL == "gpt-4.1-mini"
        assert kwargs["max_tokens"] == settings.HELPLINE.TURN_MAX_TOKENS == 900
        assert kwargs["temperature"] > 0

    @pytest.mark.asyncio
    async def test_a_prompt_override_sets_provider_model_temperature_and_text(
        self, service
    ):
        prompts = {
            "ally_ai_helpline_copilot_turn": {
                "prompt": "Reply in {language_name} to: {conversation}",
                "provider": "google",
                "model": "gemini-2.5-flash",
                "temperature": 0.2,
            }
        }
        gen = stub_generate(turn(suggestion()))
        with patch(f"{MODULE}.generate_structured", gen):
            await service.copilot_turn(TURN_MESSAGES, language="ta", prompts=prompts)

        kwargs = gen.call_args.kwargs
        assert kwargs["provider"] == "google"
        assert kwargs["model"] == "gemini-2.5-flash"
        assert kwargs["temperature"] == pytest.approx(0.2)
        assert kwargs["prompt"].startswith("Reply in Tamil to: Talker: I failed")

    @pytest.mark.asyncio
    async def test_conversation_summary_and_language_reach_the_prompt(self, service):
        gen = stub_generate(turn(suggestion()))
        with patch(f"{MODULE}.generate_structured", gen):
            await service.copilot_turn(
                TURN_MESSAGES,
                rolling_summary="Worried about exam results.",
                language="mr",
            )

        prompt = gen.call_args.kwargs["prompt"]
        assert "Worried about exam results." in prompt
        assert "Write every suggestion in Marathi" in prompt
        assert "Talker: Yeah. I just feel so alone with it." in prompt


def _rendered_calls(mock_logger) -> str:
    """Everything passed to any logger method, flattened to one string."""
    parts = []
    for name in ("debug", "info", "warning", "error", "exception"):
        for call in getattr(mock_logger, name).call_args_list:
            parts.extend(str(a) for a in call.args)
    return " ".join(parts)
