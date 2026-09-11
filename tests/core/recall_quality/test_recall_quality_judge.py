"""Unit tests for the recall-quality judge: the prompt builder and the post-processing guards.

The LLM call itself is not exercised. What is, is the part that keeps a verdict honest — a
`missed_better` naming a fact that was never in the pool would send someone to retune a weight
over material that does not exist, and it would look exactly like a real finding.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.core.recall_quality.judge import judge_recall
from app.core.recall_quality.prompt import DEFAULT_JUDGE_RUBRIC, build_judge_prompt
from app.core.recall_quality.schemas import RecallJudgment, RecallQualityOutput


def _fact(text, score=1.5, cue_hits=2, similarity=0.4):
    return {
        "text": text,
        "score": score,
        "cue_hits": cue_hits,
        "similarity": similarity,
    }


SELECTED = [_fact("she ran a tailoring shop for thirty years")]
PASSED_OVER = [_fact("her son stopped visiting after the funeral", score=1.4)]


class TestPromptBuilder:
    def test_shows_both_lists_so_a_near_miss_can_be_named(self):
        prompt = build_judge_prompt(
            counsellor_turn="Tell me about your family.",
            client_reply="There's not much to tell.",
            selected=SELECTED,
            passed_over=PASSED_OVER,
        )
        assert "she ran a tailoring shop for thirty years" in prompt
        assert "her son stopped visiting after the funeral" in prompt
        assert "RECALLED" in prompt and "PASSED OVER" in prompt

    def test_carries_the_turn_verbatim(self):
        prompt = build_judge_prompt(
            counsellor_turn="Tell me about your family.",
            client_reply="There's not much to tell.",
            selected=SELECTED,
            passed_over=[],
        )
        assert "COUNSELLOR SAID: Tell me about your family." in prompt
        assert "CLIENT REPLIED: There's not much to tell." in prompt

    def test_states_the_stance_so_withholding_is_not_read_as_failure(self):
        # A guarded client who declines to volunteer a fact recalled it perfectly well.
        prompt = build_judge_prompt(
            counsellor_turn="q",
            client_reply="a",
            selected=SELECTED,
            passed_over=[],
            stance="guarded",
        )
        assert "CLIENT'S STANCE ON THIS TURN: guarded" in prompt

    def test_states_where_the_cues_came_from(self):
        prompt = build_judge_prompt(
            counsellor_turn="q",
            client_reply="a",
            selected=SELECTED,
            passed_over=[],
            cue_tier="scenario",
        )
        assert "CUES CAME FROM: scenario" in prompt

    def test_says_an_empty_list_is_empty_rather_than_omitting_it(self):
        prompt = build_judge_prompt(
            counsellor_turn="q", client_reply="a", selected=[], passed_over=[]
        )
        assert "RECALLED (what the client had in mind): none." in prompt

    def test_the_rubric_licenses_no_demand_freely(self):
        # Without this the rate is dominated by "mm-hmm" turns scored as recall failures.
        assert "USE THIS FREELY" in DEFAULT_JUDGE_RUBRIC
        assert "no_demand" in DEFAULT_JUDGE_RUBRIC

    def test_the_rubric_separates_a_corpus_gap_from_a_ranking_failure(self):
        assert "nothing_apt" in DEFAULT_JUDGE_RUBRIC
        assert "write more" in DEFAULT_JUDGE_RUBRIC
        assert "IGNORE THE SCORES" in DEFAULT_JUDGE_RUBRIC

    def test_a_custom_rubric_replaces_the_default(self):
        prompt = build_judge_prompt(
            counsellor_turn="q",
            client_reply="a",
            selected=SELECTED,
            passed_over=[],
            rubric="ONLY THIS",
        )
        assert prompt.startswith("ONLY THIS")
        assert "IGNORE THE SCORES" not in prompt


@pytest.mark.asyncio
class TestGuards:
    async def _run(self, output, selected=SELECTED, passed_over=PASSED_OVER):
        with patch(
            "app.core.llm.dispatch.generate_structured",
            new=AsyncMock(return_value=(output, {"model": "gemini-2.5-pro"})),
        ):
            return await judge_recall(
                counsellor_turn="Tell me about your family.",
                client_reply="There's not much to tell.",
                selected=selected,
                passed_over=passed_over,
            )

    def _out(self, **kw):
        kw.setdefault("verdict", "well_chosen")
        return RecallQualityOutput(judgment=RecallJudgment(**kw))

    async def test_keeps_a_better_fact_that_was_really_in_the_pool(self):
        judgment, model = await self._run(
            self._out(
                verdict="missed_better",
                better_fact="her son stopped visiting after the funeral",
            )
        )
        assert judgment.verdict == "missed_better"
        assert judgment.better_fact == "her son stopped visiting after the funeral"
        assert model == "gemini-2.5-pro"

    async def test_strips_a_better_fact_that_was_never_shown(self):
        # Stored as-is it would read as "the ranking buried this" and send someone to retune a
        # weight over a fact that does not exist. The verdict stands; the invented quote does
        # not.
        judgment, _ = await self._run(
            self._out(verdict="missed_better", better_fact="a fact nobody wrote")
        )
        assert judgment.verdict == "missed_better"
        assert judgment.better_fact is None

    async def test_strips_a_better_fact_from_a_verdict_that_cannot_have_one(self):
        judgment, _ = await self._run(
            self._out(verdict="well_chosen", better_fact="something")
        )
        assert judgment.better_fact is None

    async def test_keeps_only_unused_facts_that_were_actually_recalled(self):
        judgment, _ = await self._run(
            self._out(
                unused_selected=[
                    "she ran a tailoring shop for thirty years",
                    "never recalled at all",
                ]
            )
        )
        assert judgment.unused_selected == ["she ran a tailoring shop for thirty years"]

    async def test_returns_none_when_the_judge_returns_nothing(self):
        # Distinct from `no_demand`: this is OUR failure, and counting it as "the turn needed
        # nothing" would quietly inflate the healthy bucket.
        judgment, model = await self._run(None)
        assert judgment is None
        assert model == "gemini-2.5-pro"

    async def test_does_not_call_the_model_when_there_was_no_pool(self):
        # No pool was never a ranking decision. Different from a pool with nothing apt, which
        # is a real finding.
        with patch(
            "app.core.llm.dispatch.generate_structured", new=AsyncMock()
        ) as spy:
            judgment, model = await judge_recall(
                counsellor_turn="q",
                client_reply="a",
                selected=[],
                passed_over=[],
            )
        spy.assert_not_called()
        assert judgment is None
        assert model == "gemini-2.5-pro"

    async def test_reports_the_model_that_actually_ran(self):
        with patch(
            "app.core.llm.dispatch.generate_structured",
            new=AsyncMock(
                return_value=(
                    self._out(),
                    {
                        "model": "gemini-2.5-flash",
                        "provider": "gemini",
                        "fell_back_from": "gemini-2.5-pro",
                    },
                )
            ),
        ):
            _, model = await judge_recall(
                counsellor_turn="q",
                client_reply="a",
                selected=SELECTED,
                passed_over=[],
            )
        assert model == "gemini-2.5-flash"
