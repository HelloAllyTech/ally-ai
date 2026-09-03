"""Tests for KnowledgeAgentService.

The load-bearing behaviours are the ones that keep a wrong answer from looking right:
citations validated in code rather than trusted, the decline gate firing before any LLM
call, and a non-English question being translated BEFORE it is embedded.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.core.knowledge_agent.agent import KnowledgeAgentService
from app.core.knowledge_agent.schemas import (
    AnswerIntent,
    CrisisVerdict,
    DeclineReason,
    KnowledgeAnswer,
    TranslatedQuery,
)
from app.exceptions.custom_exceptions import LLMInvocationFailedException

CHUNK_A = "11111111-1111-1111-1111-111111111111"
CHUNK_B = "22222222-2222-2222-2222-222222222222"
DOC_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

AGENT = "app.core.knowledge_agent.agent"


def passage(chunk_id, similarity, text="a passage", tokens=100, **overrides):
    p = {
        "chunk_id": chunk_id,
        "document_id": DOC_ID,
        "document_title": "WHO mhGAP Intervention Guide",
        "chunk_index": 0,
        "text": text,
        "char_start": 0,
        "char_end": len(text),
        "page_from": 44,
        "page_to": 44,
        "section_path": "Depression > Assessment",
        "source_url": "",
        "language": "en",
        "token_count": tokens,
        "similarity": similarity,
    }
    p.update(overrides)
    return p


@pytest.fixture
def chunk_service():
    svc = AsyncMock()
    svc.search.return_value = []
    return svc


@pytest.fixture
def agent(chunk_service):
    return KnowledgeAgentService(chunk_service)


def stub_generate(*results):
    """Return an AsyncMock for generate_structured yielding (parsed, meta) in order."""
    return AsyncMock(
        side_effect=[
            (r, {"provider": "anthropic", "model": "claude-sonnet-4-6"})
            for r in results
        ]
    )


class TestDeclineGate:
    @pytest.mark.asyncio
    async def test_no_hits_declines_without_calling_the_llm(self, agent, chunk_service):
        """Zero hits is decided in code — no generation tokens spent."""
        chunk_service.search.return_value = []

        with patch(f"{AGENT}.generate_structured") as gen:
            result = await agent.answer("anything", translate_query=False)

        gen.assert_not_called()
        assert result["intent"] == AnswerIntent.DECLINE
        assert result["decline_reason"] == DeclineReason.NO_HITS
        assert result["citations"] == []
        # No model ran, so no model is reported.
        assert result["provider"] == ""
        assert result["model"] == ""

    @pytest.mark.asyncio
    async def test_below_threshold_declines_without_calling_the_llm(
        self, agent, chunk_service
    ):
        chunk_service.search.return_value = [passage(CHUNK_A, 0.30)]

        with patch(f"{AGENT}.generate_structured") as gen:
            result = await agent.answer(
                "q", translate_query=False, min_similarity=0.2, decline_similarity=0.42
            )

        gen.assert_not_called()
        assert result["decline_reason"] == DeclineReason.BELOW_THRESHOLD
        assert result["retrieval"]["top_similarity"] == 0.3

    @pytest.mark.asyncio
    async def test_retrieval_floor_and_decline_threshold_are_separate(
        self, agent, chunk_service
    ):
        """A hit between the two thresholds still reaches the model.

        This is the whole point of the split: a relevant passage against a paraphrased
        question scores ~0.40-0.60, so a single hard floor at the decision value would
        decline constantly on legitimate rephrasings.
        """
        chunk_service.search.return_value = [passage(CHUNK_A, 0.45)]

        with patch(
            f"{AGENT}.generate_structured",
            stub_generate(
                KnowledgeAnswer(
                    intent=AnswerIntent.ANSWER, answer="Ask directly.", citations=[1]
                )
            ),
        ):
            result = await agent.answer(
                "q", translate_query=False, min_similarity=0.35, decline_similarity=0.42
            )

        assert result["intent"] == AnswerIntent.ANSWER
        # The floor is what was sent to retrieval, not the decision threshold.
        assert chunk_service.search.call_args.kwargs["min_similarity"] == 0.35


class TestCitations:
    @pytest.mark.asyncio
    async def test_citations_resolve_to_chunk_metadata(self, agent, chunk_service):
        chunk_service.search.return_value = [
            passage(CHUNK_A, 0.7, "first"),
            passage(CHUNK_B, 0.68, "second", page_from=51, page_to=52),
        ]

        with patch(
            f"{AGENT}.generate_structured",
            stub_generate(
                KnowledgeAnswer(
                    intent=AnswerIntent.ANSWER, answer="Both say so.", citations=[2, 1]
                )
            ),
        ):
            result = await agent.answer("q", translate_query=False)

        cites = result["citations"]
        # Order preserved as the model reasoned, not re-sorted.
        assert [c["passage_number"] for c in cites] == [2, 1]
        assert cites[0]["chunk_id"] == CHUNK_B
        assert cites[0]["page_from"] == 51
        assert cites[0]["page_to"] == 52
        assert cites[1]["chunk_id"] == CHUNK_A

    @pytest.mark.asyncio
    async def test_out_of_range_citations_are_dropped_not_clamped(
        self, agent, chunk_service
    ):
        """A citation naming a passage that was never supplied is discarded.

        Clamping would attribute the claim to a real but WRONG passage, producing a
        source line the reader cannot check while making the answer look better sourced.
        """
        chunk_service.search.return_value = [passage(CHUNK_A, 0.7)]

        with patch(
            f"{AGENT}.generate_structured",
            stub_generate(
                KnowledgeAnswer(
                    intent=AnswerIntent.ANSWER,
                    answer="Grounded.",
                    citations=[1, 9, 0, -3],
                )
            ),
        ):
            result = await agent.answer("q", translate_query=False)

        assert [c["passage_number"] for c in result["citations"]] == [1]

    @pytest.mark.asyncio
    async def test_duplicate_citations_collapse(self, agent, chunk_service):
        chunk_service.search.return_value = [passage(CHUNK_A, 0.7)]

        with patch(
            f"{AGENT}.generate_structured",
            stub_generate(
                KnowledgeAnswer(
                    intent=AnswerIntent.ANSWER, answer="x", citations=[1, 1, 1]
                )
            ),
        ):
            result = await agent.answer("q", translate_query=False)

        assert len(result["citations"]) == 1

    @pytest.mark.asyncio
    async def test_answer_without_citations_is_kept_but_flagged(
        self, agent, chunk_service
    ):
        """Cross-passage synthesis legitimately cites nothing, so the answer survives.

        It is still counted, because that count is what shows whether the grounding
        instruction is holding.
        """
        chunk_service.search.return_value = [passage(CHUNK_A, 0.7)]

        with patch(
            f"{AGENT}.generate_structured",
            stub_generate(
                KnowledgeAnswer(
                    intent=AnswerIntent.ANSWER, answer="Synthesised.", citations=[]
                )
            ),
        ):
            result = await agent.answer("q", translate_query=False)

        assert result["intent"] == AnswerIntent.ANSWER
        assert result["answer"] == "Synthesised."
        assert result["retrieval"]["unsupported"] is True


class TestPostValidation:
    @pytest.mark.asyncio
    async def test_answer_with_empty_text_becomes_a_decline(self, agent, chunk_service):
        """'I answered' with nothing in it would reach the worker as silence."""
        chunk_service.search.return_value = [passage(CHUNK_A, 0.7)]

        with patch(
            f"{AGENT}.generate_structured",
            stub_generate(
                KnowledgeAnswer(intent=AnswerIntent.ANSWER, answer="   ", citations=[1])
            ),
        ):
            result = await agent.answer("q", translate_query=False)

        assert result["intent"] == AnswerIntent.DECLINE
        assert result["decline_reason"] == DeclineReason.MODEL_DECLINED
        assert result["citations"] == []

    @pytest.mark.asyncio
    async def test_model_decline_carries_its_own_reason(self, agent, chunk_service):
        chunk_service.search.return_value = [passage(CHUNK_A, 0.7)]

        with patch(
            f"{AGENT}.generate_structured",
            stub_generate(
                KnowledgeAnswer(
                    intent=AnswerIntent.DECLINE,
                    answer="My material doesn't cover that.",
                    citations=[1],
                )
            ),
        ):
            result = await agent.answer("q", translate_query=False)

        # MODEL_DECLINED, not BELOW_THRESHOLD: the model saw the passages and judged
        # them insufficient, which is a different tuning signal from a threshold
        # rejection.
        assert result["decline_reason"] == DeclineReason.MODEL_DECLINED
        assert result["citations"] == []

    @pytest.mark.asyncio
    async def test_clarify_carries_no_citations_and_no_decline_reason(
        self, agent, chunk_service
    ):
        """A vague question is not a corpus gap, so it must not look like one."""
        chunk_service.search.return_value = [passage(CHUNK_A, 0.7)]

        with patch(
            f"{AGENT}.generate_structured",
            stub_generate(
                KnowledgeAnswer(
                    intent=AnswerIntent.CLARIFY,
                    answer="Which age group do you mean?",
                    citations=[1],
                )
            ),
        ):
            result = await agent.answer("help with a client", translate_query=False)

        assert result["intent"] == AnswerIntent.CLARIFY
        assert result["decline_reason"] == DeclineReason.NONE
        assert result["citations"] == []

    @pytest.mark.asyncio
    async def test_missing_prompt_template_raises(self, agent, chunk_service):
        """An empty template must not become an unguided free-text answer."""
        chunk_service.search.return_value = [passage(CHUNK_A, 0.7)]

        with (
            patch(f"{AGENT}.build_answer_prompt", return_value=""),
            patch(f"{AGENT}.generate_structured") as gen,
        ):
            with pytest.raises(LLMInvocationFailedException):
                await agent.answer("q", translate_query=False)

        gen.assert_not_called()


class TestPassageSelection:
    @pytest.mark.asyncio
    async def test_similarity_band_drops_weak_hits(self, agent, chunk_service):
        """One strong match beats a strong match diluted by weak ones."""
        chunk_service.search.return_value = [
            passage(CHUNK_A, 0.80),
            passage(CHUNK_B, 0.78),
            passage("cccccccc-cccc-cccc-cccc-cccccccccccc", 0.50),
        ]

        gen = stub_generate(
            KnowledgeAnswer(intent=AnswerIntent.ANSWER, answer="x", citations=[1])
        )
        with patch(f"{AGENT}.generate_structured", gen):
            result = await agent.answer(
                "q", translate_query=False, similarity_band=0.08
            )

        # 0.80 and 0.78 are within the band; 0.50 is not.
        assert result["retrieval"]["passages_used"] == 2

    @pytest.mark.asyncio
    async def test_max_passages_caps_selection(self, agent, chunk_service):
        chunk_service.search.return_value = [
            passage(f"{i}" * 8 + "-0000-0000-0000-000000000000", 0.80) for i in range(6)
        ]

        with patch(
            f"{AGENT}.generate_structured",
            stub_generate(
                KnowledgeAnswer(intent=AnswerIntent.ANSWER, answer="x", citations=[1])
            ),
        ):
            result = await agent.answer("q", translate_query=False, max_passages=3)

        assert result["retrieval"]["passages_used"] == 3

    @pytest.mark.asyncio
    async def test_context_token_budget_stops_adding_passages(
        self, agent, chunk_service
    ):
        """
        Budgeting uses the token_count ally-be already computed — no tokeniser needed.
        """
        chunk_service.search.return_value = [
            passage(CHUNK_A, 0.80, tokens=400),
            passage(CHUNK_B, 0.79, tokens=400),
            passage("cccccccc-cccc-cccc-cccc-cccccccccccc", 0.78, tokens=400),
        ]

        with patch(
            f"{AGENT}.generate_structured",
            stub_generate(
                KnowledgeAnswer(intent=AnswerIntent.ANSWER, answer="x", citations=[1])
            ),
        ):
            result = await agent.answer(
                "q", translate_query=False, max_context_tokens=900
            )

        # 400 + 400 fits in 900; the third would exceed it.
        assert result["retrieval"]["passages_used"] == 2

    @pytest.mark.asyncio
    async def test_first_passage_is_always_included(self, agent, chunk_service):
        """A budget smaller than the best passage still yields one passage, not zero.

        Returning nothing here would turn a covered question into a decline purely
        because of a misconfigured budget.
        """
        chunk_service.search.return_value = [passage(CHUNK_A, 0.80, tokens=5000)]

        with patch(
            f"{AGENT}.generate_structured",
            stub_generate(
                KnowledgeAnswer(intent=AnswerIntent.ANSWER, answer="x", citations=[1])
            ),
        ):
            result = await agent.answer(
                "q", translate_query=False, max_context_tokens=200
            )

        assert result["retrieval"]["passages_used"] == 1


class TestQueryTranslation:
    @pytest.mark.asyncio
    async def test_non_english_question_is_embedded_in_english(
        self, agent, chunk_service
    ):
        """Translation happens BEFORE retrieval, which is the only place it helps.

        Cross-lingual embedding alignment is weak, so the search text — not just the
        answer language — is what has to change.
        """
        chunk_service.search.return_value = [passage(CHUNK_A, 0.7)]

        gen = stub_generate(
            TranslatedQuery(
                is_english=False,
                language="hi",
                english_query="How do I ask about suicidal intent?",
            ),
            KnowledgeAnswer(
                intent=AnswerIntent.ANSWER,
                answer="सीधे पूछें।",
                citations=[1],
                language="hi",
            ),
        )
        with patch(f"{AGENT}.generate_structured", gen):
            result = await agent.answer("आत्महत्या के बारे में कैसे पूछूं?")

        assert (
            chunk_service.search.call_args.kwargs["query"]
            == "How do I ask about suicidal intent?"
        )
        assert result["language"] == "hi"
        assert result["retrieval"]["query_language"] == "hi"
        assert (
            result["retrieval"]["translated_query"]
            == "How do I ask about suicidal intent?"
        )

    @pytest.mark.asyncio
    async def test_english_question_is_searched_as_written(self, agent, chunk_service):
        chunk_service.search.return_value = [passage(CHUNK_A, 0.7)]

        gen = stub_generate(
            TranslatedQuery(is_english=True, language="en", english_query=""),
            KnowledgeAnswer(intent=AnswerIntent.ANSWER, answer="Ask.", citations=[1]),
        )
        with patch(f"{AGENT}.generate_structured", gen):
            result = await agent.answer("How do I ask about intent?")

        assert (
            chunk_service.search.call_args.kwargs["query"]
            == "How do I ask about intent?"
        )
        # None means "searched as asked" — distinguishable from a translation that
        # happened to be identical.
        assert result["retrieval"]["translated_query"] is None

    @pytest.mark.asyncio
    async def test_translation_failure_degrades_to_the_original_text(
        self, agent, chunk_service
    ):
        """A translation outage costs retrieval quality, not availability."""
        chunk_service.search.return_value = [passage(CHUNK_A, 0.7)]

        gen = AsyncMock(
            side_effect=[
                RuntimeError("translate model down"),
                (
                    KnowledgeAnswer(
                        intent=AnswerIntent.ANSWER, answer="Ask.", citations=[1]
                    ),
                    {"provider": "anthropic", "model": "claude-sonnet-4-6"},
                ),
            ]
        )
        with patch(f"{AGENT}.generate_structured", gen):
            result = await agent.answer("आत्महत्या के बारे में कैसे पूछूं?")

        assert chunk_service.search.call_args.kwargs["query"] == (
            "आत्महत्या के बारे में कैसे पूछूं?"
        )
        assert result["intent"] == AnswerIntent.ANSWER

    @pytest.mark.asyncio
    async def test_translate_disabled_skips_the_call(self, agent, chunk_service):
        chunk_service.search.return_value = [passage(CHUNK_A, 0.7)]

        gen = stub_generate(
            KnowledgeAnswer(intent=AnswerIntent.ANSWER, answer="Ask.", citations=[1])
        )
        with patch(f"{AGENT}.generate_structured", gen):
            await agent.answer("q", translate_query=False)

        # Exactly one call: the answer. No translation round trip.
        assert gen.await_count == 1


class TestTraceability:
    @pytest.mark.asyncio
    async def test_reports_the_model_that_actually_ran(self, agent, chunk_service):
        """
        ally-be stores this so an answer that changed after a model swap is explainable.
        """
        chunk_service.search.return_value = [passage(CHUNK_A, 0.7)]

        gen = AsyncMock(
            return_value=(
                KnowledgeAnswer(intent=AnswerIntent.ANSWER, answer="x", citations=[1]),
                {
                    "provider": "gemini",
                    "model": "gemini-2.5-flash",
                    "fell_back_from": "anthropic",
                },
            )
        )
        with patch(f"{AGENT}.generate_structured", gen):
            result = await agent.answer("q", translate_query=False)

        assert result["provider"] == "gemini"
        assert result["model"] == "gemini-2.5-flash"
        assert result["prompt_version"]


class TestCrisisClassifier:
    """The second layer of the safety net.

    The keyword rules in ally-be catch messages that say it outright. This exists for
    the ones that do not, which is the normal case for people whose job is noticing it
    in others. Two behaviours matter enough to pin:

    A failure must return ``is_crisis: False`` with ``failed: True`` rather than raising
    or defaulting to True. Raising takes the whole question down over the classifier;
    defaulting to True answers every question with a crisis message the moment a key
    expires, and a bot that only ever says "call a crisis line" is one workers stop
    reading. The keyword rules are what still hold in that window.

    And the verdict must come back untouched — no confidence threshold applied here. The
    prompt instructs the model to choose crisis when uncertain, so re-gating on
    confidence downstream would silently undo the instruction that makes the net work.
    """

    @pytest.mark.asyncio
    async def test_returns_the_verdict_with_the_signal_phrase(self, agent):
        gen = stub_generate(
            CrisisVerdict(
                is_crisis=True, signal="I can't keep doing this", confidence=0.55
            )
        )
        with patch(f"{AGENT}.generate_structured", gen):
            result = await agent.classify_crisis("I can't keep doing this")

        assert result["is_crisis"] is True
        assert result["signal"] == "I can't keep doing this"
        assert result["failed"] is False

    @pytest.mark.asyncio
    async def test_a_low_confidence_crisis_is_still_a_crisis(self, agent):
        # The borderline message is the whole point. If a threshold were applied here,
        # the prompt's "choose true when uncertain" instruction would be dead code.
        gen = stub_generate(
            CrisisVerdict(is_crisis=True, signal="tired of all of this", confidence=0.2)
        )
        with patch(f"{AGENT}.generate_structured", gen):
            result = await agent.classify_crisis("I'm so tired of all of this")

        assert result["is_crisis"] is True
        assert result["confidence"] == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_an_ordinary_reference_question_is_not_a_crisis(self, agent):
        gen = stub_generate(CrisisVerdict(is_crisis=False, confidence=0.9))
        with patch(f"{AGENT}.generate_structured", gen):
            result = await agent.classify_crisis(
                "How do I carry out a suicide risk assessment?"
            )

        assert result["is_crisis"] is False
        assert result["signal"] == ""
        assert result["failed"] is False

    @pytest.mark.asyncio
    async def test_a_failure_reports_itself_rather_than_raising(self, agent):
        gen = AsyncMock(side_effect=RuntimeError("classifier model down"))
        with patch(f"{AGENT}.generate_structured", gen):
            result = await agent.classify_crisis("anything")

        assert result["is_crisis"] is False
        # The distinction the caller needs: "looked and said no" versus "could not
        # look".
        assert result["failed"] is True

    @pytest.mark.asyncio
    async def test_a_missing_template_does_not_silently_pass_the_message(self, agent):
        with patch(f"{AGENT}.build_crisis_prompt", return_value=""):
            result = await agent.classify_crisis("anything")

        assert result == {
            "is_crisis": False,
            "signal": "",
            "confidence": 0.0,
            "failed": True,
        }

    @pytest.mark.asyncio
    async def test_a_positive_signal_goes_to_the_phi_log_not_the_app_log(self, agent):
        """The signal phrase is a worker's verbatim disclosure — PHI. It must go to
        the PHI audit log, never the general application logger, no matter how
        operationally useful a positive verdict is to see in the app log."""
        gen = stub_generate(
            CrisisVerdict(
                is_crisis=True, signal="I want to end my life", confidence=0.9
            )
        )
        with patch(f"{AGENT}.generate_structured", gen), patch(
            f"{AGENT}.logger"
        ) as mock_logger, patch(f"{AGENT}.phi_logger") as mock_phi_logger:
            mock_phi_logger.log = AsyncMock()
            await agent.classify_crisis("I want to end my life")

        for call in mock_logger.info.call_args_list:
            rendered = " ".join(str(a) for a in call.args)
            assert "I want to end my life" not in rendered

        mock_phi_logger.log.assert_awaited_once()
        logged_event = mock_phi_logger.log.call_args.args[0]
        assert logged_event.details["signal"] == "I want to end my life"
