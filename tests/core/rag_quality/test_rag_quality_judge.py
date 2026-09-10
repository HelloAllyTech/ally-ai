"""Unit tests for the RAG-quality judge's deterministic parts: the prompt
builder and the post-processing guards. The LLM call itself is not exercised.

What these protect is the integrity of a rate. Every number this judge feeds is
computed downstream in SQL over its labels, so a hallucinated chunk_id or a
duplicated judgment does not produce a visible error — it produces a plausible
precision figure that is quietly wrong, which is worse than no figure at all.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.core.rag_quality.judge import judge_retrieval
from app.core.rag_quality.prompt import (
    CORPUS_PURPOSE,
    DEFAULT_JUDGE_RUBRIC,
    build_judge_prompt,
)
from app.core.rag_quality.schemas import (
    PassageJudgment,
    RagQualityOutput,
    RetrievalJudgment,
)


def _passage(chunk_id="c1", text="Word-finding difficulty sounds like circling.", **kw):
    base = {
        "chunk_id": chunk_id,
        "document_title": "Designing Clients",
        "section_path": "How speech changes",
        "similarity": 0.5123,
        "text": text,
    }
    base.update(kw)
    return base


def _judgment(chunk_id="c1", relevance="relevant", superficial=False):
    return PassageJudgment(
        chunk_id=chunk_id,
        relevance=relevance,
        superficial_match=superficial,
        reasoning="because",
    )


# ---- prompt builder ----


class TestPromptBuilder:
    def test_states_what_the_corpus_is_for(self):
        # Relevance is not corpus-independent: craft guidance answers a craft
        # question and is tangential to a subject question. Without the purpose
        # the judge cannot label either honestly.
        prompt = build_judge_prompt("q", "character_library", [_passage()], 0.35)
        assert CORPUS_PURPOSE["character_library"] in prompt

    def test_names_an_unrecognised_corpus_rather_than_guessing(self):
        prompt = build_judge_prompt("q", "not_a_corpus", [_passage()], 0.35)
        assert "unrecognised corpus (not_a_corpus)" in prompt

    def test_includes_the_query_verbatim_and_the_floor(self):
        prompt = build_judge_prompt(
            "how specific should a character be?", "character_library", [_passage()], 0.35
        )
        assert "QUERY AS ISSUED: how specific should a character be?" in prompt
        assert "SIMILARITY FLOOR IN FORCE: 0.35" in prompt

    def test_carries_each_passage_with_its_id_score_and_text(self):
        prompt = build_judge_prompt("q", "character_library", [_passage()], 0.35)
        assert "[c1]" in prompt
        assert "Designing Clients · How speech changes" in prompt
        assert "similarity 0.5123" in prompt
        assert "Word-finding difficulty sounds like circling." in prompt

    def test_tells_the_judge_to_ignore_the_scores_when_labelling(self):
        # The scores are shown so a score/substance mismatch can be flagged,
        # not so they can be deferred to. This audit exists because they are
        # not trustworthy alone.
        assert "IGNORE THE SIMILARITY SCORES" in DEFAULT_JUDGE_RUBRIC

    def test_says_tangential_is_not_a_softer_irrelevant(self):
        # The label that matters most, and the one a judge will otherwise treat
        # as a hedge.
        assert "TANGENTIAL IS NOT A SOFTER IRRELEVANT" in DEFAULT_JUDGE_RUBRIC

    def test_an_empty_retrieval_asks_for_nothing_useful_and_a_gap(self):
        prompt = build_judge_prompt("q", "character_library", [], 0.35)
        assert "PASSAGES RETURNED: none." in prompt
        assert "nothing_useful" in prompt
        assert "missing" in prompt

    def test_a_custom_rubric_replaces_the_default(self):
        prompt = build_judge_prompt(
            "q", "character_library", [_passage()], 0.35, rubric="ONLY THIS"
        )
        assert prompt.startswith("ONLY THIS")
        assert "IGNORE THE SIMILARITY SCORES" not in prompt


# ---- post-processing guards ----


@pytest.mark.asyncio
class TestJudgeGuards:
    async def _run(self, output, passages):
        with patch(
            "app.core.llm.dispatch.generate_structured",
            new=AsyncMock(return_value=(output, {"model": "gemini-2.5-pro"})),
        ):
            return await judge_retrieval("q", "character_library", passages, 0.35)

    async def test_keeps_a_judgment_for_a_passage_we_sent(self):
        out = RagQualityOutput(
            passages=[_judgment("c1")],
            retrieval=RetrievalJudgment(sufficiency="sufficient"),
        )
        kept, retrieval, model = await self._run(out, [_passage("c1")])
        assert [p.chunk_id for p in kept] == ["c1"]
        assert retrieval.sufficiency == "sufficient"
        assert model == "gemini-2.5-pro"

    async def test_drops_a_judgment_for_a_passage_we_never_sent(self):
        # A hallucinated id would otherwise be stored against a real passage's
        # row and misreport it.
        out = RagQualityOutput(
            passages=[_judgment("c1"), _judgment("invented")],
            retrieval=RetrievalJudgment(sufficiency="partial", missing="a case account"),
        )
        kept, _, _ = await self._run(out, [_passage("c1")])
        assert [p.chunk_id for p in kept] == ["c1"]

    async def test_drops_a_duplicate_judgment_for_the_same_passage(self):
        # A duplicate double-counts one passage in every downstream rate.
        out = RagQualityOutput(
            passages=[
                _judgment("c1", "relevant"),
                _judgment("c1", "irrelevant"),
            ],
            retrieval=RetrievalJudgment(sufficiency="sufficient"),
        )
        kept, _, _ = await self._run(out, [_passage("c1")])
        assert len(kept) == 1
        assert kept[0].relevance == "relevant"

    async def test_leaves_a_skipped_passage_unjudged_rather_than_guessing(self):
        # An unjudged passage must stay unjudged; backfilling a default would
        # put invented labels into a precision figure.
        out = RagQualityOutput(
            passages=[_judgment("c1")],
            retrieval=RetrievalJudgment(sufficiency="partial", missing="x"),
        )
        kept, _, _ = await self._run(out, [_passage("c1"), _passage("c2")])
        assert [p.chunk_id for p in kept] == ["c1"]

    async def test_judges_an_empty_retrieval_rather_than_skipping_it(self):
        # The most informative row in the log: a corpus gap and a too-tight
        # floor look identical in the count, and `missing` is what separates
        # them.
        out = RagQualityOutput(
            passages=[],
            retrieval=RetrievalJudgment(
                sufficiency="nothing_useful", missing="how specific to make an option"
            ),
        )
        kept, retrieval, _ = await self._run(out, [])
        assert kept == []
        assert retrieval.sufficiency == "nothing_useful"
        assert retrieval.missing

    async def test_returns_none_for_the_set_when_the_judge_returns_nothing(self):
        # Distinct from `nothing_useful`: this is OUR failure, and counting it
        # as a corpus gap would blame the library for our outage.
        kept, retrieval, model = await self._run(None, [_passage("c1")])
        assert kept == []
        assert retrieval is None
        assert model == "gemini-2.5-pro"

    async def test_does_not_call_the_model_for_a_blank_query(self):
        with patch(
            "app.core.llm.dispatch.generate_structured", new=AsyncMock()
        ) as spy:
            kept, retrieval, model = await judge_retrieval(
                "   ", "character_library", [_passage()], 0.35
            )
        spy.assert_not_called()
        assert kept == [] and retrieval is None
        assert model == "gemini-2.5-pro"

    async def test_reports_the_model_that_actually_ran_not_the_configured_one(self):
        # A fallback recorded under the configured model pollutes a pinned
        # series — the trap the language judge already hit.
        out = RagQualityOutput(
            passages=[_judgment("c1")],
            retrieval=RetrievalJudgment(sufficiency="sufficient"),
        )
        with patch(
            "app.core.llm.dispatch.generate_structured",
            new=AsyncMock(
                return_value=(
                    out,
                    {
                        "model": "gemini-2.5-flash",
                        "provider": "gemini",
                        "fell_back_from": "gemini-2.5-pro",
                    },
                )
            ),
        ):
            _, _, model = await judge_retrieval(
                "q", "character_library", [_passage("c1")], 0.35
            )
        assert model == "gemini-2.5-flash"
