"""The quality judges never run on a substitute model.

Dispatch normally retries a failed provider once on OpenAI, and substitutes
OpenAI for a provider with no key. For the judges both are pure cost: ally-be
selects work by the pinned judge model (`judgeModel` in its backlog targets, or
the configured model it reads back from the filler endpoint), so a session
judged by the fallback is still "unjudged" under the pinned model and is judged
again on Gemini. The substitute's call is paid for and its rows feed no pinned
series — and for the filler judge, whose rates are read without a model filter,
both rows would count.

Pinned two ways: every judge call site asks for `never_fallback`, and through
the real dispatch a judge whose Gemini call fails (or has no key) raises without
OpenAI ever being called.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.filler_quality.schemas import FillerObservation
from app.core.llm import dispatch
from app.exceptions.custom_exceptions import LLMInvocationFailedException

TRANSCRIPT = [
    {"role": "counselor", "text": "What brings you in today?"},
    {"role": "client", "turn_index": 0, "text": "Everything is too much lately."},
]


async def _drift():
    from app.core.drift.judge import judge_session

    return await judge_session(TRANSCRIPT, persona="p", language="en")


async def _drift_labels():
    from app.core.drift.judge import judge_session_labels_only

    return await judge_session_labels_only(TRANSCRIPT, persona="p", language="en")


async def _language():
    from app.core.language_quality.judge import judge_session

    return await judge_session(TRANSCRIPT, persona="p", language="ta-IN")


async def _groundedness():
    from app.core.feedback_groundedness.judge import judge_feedback

    return await judge_feedback(
        TRANSCRIPT,
        [{"claim_index": 0, "kind": "positive", "text": "Listened well."}],
        "en",
    )


async def _recall():
    from app.core.recall_quality.judge import judge_recall

    return await judge_recall(
        counsellor_turn="Tell me about work.",
        client_reply="I sewed.",
        selected=[{"text": "she ran a tailoring shop", "score": 1.5}],
        passed_over=[],
    )


async def _rag():
    from app.core.rag_quality.judge import judge_retrieval

    return await judge_retrieval(
        "how do I ask about intent?",
        "whatsapp_qa",
        [{"chunk_id": "c1", "text": "Ask directly.", "similarity": 0.8}],
        0.3,
    )


async def _filler():
    from app.core.filler_quality.judge import judge_session

    return await judge_session(
        [FillerObservation(turn_index=0, filler_text="hmm")], "p", "en"
    )


JUDGES = {
    "drift": _drift,
    "drift_labels": _drift_labels,
    "language": _language,
    "feedback_groundedness": _groundedness,
    "recall_quality": _recall,
    "rag_quality": _rag,
    "filler": _filler,
}


class TestEveryJudgeAsksForIt:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(JUDGES))
    async def test_call_site_sets_never_fallback(self, monkeypatch, name):
        captured = {}

        async def _fake(**kwargs):
            captured.update(kwargs)
            return None, {
                "provider": "gemini",
                "model": "gemini-2.5-pro",
                "fell_back_from": None,
            }

        # The judges import dispatch inside the function, so the module
        # attribute is what the call resolves to.
        monkeypatch.setattr(dispatch, "generate_structured", _fake)

        try:
            await JUDGES[name]()
        except RuntimeError as e:
            # Some judges raise on an empty output, after the call under test.
            assert "no parsable output" in str(e)

        assert captured["never_fallback"] is True
        assert captured["provider"] == dispatch.PROVIDER_GEMINI


class TestThroughTheRealDispatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(JUDGES))
    async def test_a_failed_gemini_call_is_not_retried_on_openai(self, name):
        # A 503: exactly the failure dispatch would otherwise retry elsewhere.
        unavailable = type("ServerError", (Exception,), {"code": 503})("down")
        gemini = AsyncMock(side_effect=unavailable)
        openai = AsyncMock()
        with (
            patch.object(dispatch.settings.GEMINI, "API_KEY", "g"),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "o"),
            patch.dict(dispatch._GENERATORS, {"gemini": gemini, "openai": openai}),
        ):
            with pytest.raises(LLMInvocationFailedException):
                await JUDGES[name]()

        assert gemini.await_count == 1
        openai.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(JUDGES))
    async def test_a_missing_gemini_key_is_not_substituted(self, name):
        gemini = AsyncMock()
        openai = AsyncMock()
        with (
            patch.object(dispatch.settings.GEMINI, "API_KEY", None),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "o"),
            patch.dict(dispatch._GENERATORS, {"gemini": gemini, "openai": openai}),
        ):
            with pytest.raises(LLMInvocationFailedException):
                await JUDGES[name]()

        gemini.assert_not_awaited()
        openai.assert_not_awaited()


class TestOtherCallersKeepTheirFallback:
    """Out of scope on purpose: the WhatsApp bot and the analytics agent are
    better served by a different model than by no answer."""

    @pytest.mark.asyncio
    async def test_the_default_still_falls_back(self):
        gemini = AsyncMock(
            side_effect=type("ServerError", (Exception,), {"code": 503})("down")
        )
        openai_response = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="{}"), finish_reason="stop"
                )
            ],
        )
        openai_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(return_value=openai_response)
                )
            )
        )

        from pydantic import BaseModel

        class Out(BaseModel):
            answer: str = ""

        with (
            patch.object(dispatch.settings.GEMINI, "API_KEY", "g"),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "o"),
            patch.dict(dispatch._GENERATORS, {"gemini": gemini}),
            patch.object(dispatch, "_get_openai_client", return_value=openai_client),
        ):
            _, meta = await dispatch.generate_structured(
                schema=Out, prompt="q", provider="gemini", model="gemini-2.5-pro"
            )

        assert meta["fell_back_from"] == "gemini"
