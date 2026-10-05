"""`scenario_session_id` on the judges: request -> judge -> dispatch -> usage row.

The judges run outside any LiveKit room, so the usage rows they emit had nothing
to tie them to a session, and judge cost could not be read per session. ally-be
now sends the session id; these tests pin the whole path it travels, and that
leaving it out (every caller that predates it) changes nothing.

Each judge is exercised through its real function, because a parameter that is
accepted and then dropped is invisible to any test of the pieces.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.drift.judge import compute_session_rollup
from app.core.filler_quality.schemas import FillerJudgmentResult, FillerObservation
from app.core.language_quality.schemas import LanguageJudgmentResult

TRANSCRIPT = [
    {"role": "counselor", "text": "What brings you in today?"},
    {"role": "client", "turn_index": 0, "text": "Everything is too much lately."},
]
CLAIMS = [{"claim_index": 0, "kind": "positive", "text": "Listened well."}]
FACT = {"text": "she ran a tailoring shop", "score": 1.5}


async def _drift(**kw):
    from app.core.drift.judge import judge_session

    return await judge_session(TRANSCRIPT, persona="p", language="en", **kw)


async def _drift_labels(**kw):
    from app.core.drift.judge import judge_session_labels_only

    return await judge_session_labels_only(TRANSCRIPT, persona="p", language="en", **kw)


async def _language(**kw):
    from app.core.language_quality.judge import judge_session

    return await judge_session(TRANSCRIPT, persona="p", language="ta-IN", **kw)


async def _groundedness(**kw):
    from app.core.feedback_groundedness.judge import judge_feedback

    return await judge_feedback(TRANSCRIPT, CLAIMS, "en", **kw)


async def _recall(**kw):
    from app.core.recall_quality.judge import judge_recall

    return await judge_recall(
        counsellor_turn="Tell me about work.",
        client_reply="I sewed.",
        selected=[FACT],
        passed_over=[],
        **kw,
    )


async def _filler(**kw):
    from app.core.filler_quality.judge import judge_session

    return await judge_session(
        [FillerObservation(turn_index=0, filler_text="hmm")], "p", "en", **kw
    )


JUDGES = {
    "drift": _drift,
    "drift_labels": _drift_labels,
    "language": _language,
    "feedback_groundedness": _groundedness,
    "recall_quality": _recall,
    "filler": _filler,
}


@pytest.fixture
def captured(monkeypatch):
    """Stand in for generate_structured, recording how each judge called it.

    The judges import dispatch INSIDE the function, so patching the attribute
    on the dispatch module is what the call resolves to. Returning no output
    makes some judges raise afterwards ("no parsable output"); the call under
    test has happened by then, so that is swallowed below.
    """
    from app.core.llm import dispatch

    calls = {}

    async def _fake(**kwargs):
        calls.update(kwargs)
        return None, {
            "provider": "gemini",
            "model": "gemini-2.5-pro",
            "fell_back_from": None,
        }

    monkeypatch.setattr(dispatch, "generate_structured", _fake)
    return calls


async def _run(judge, **kw):
    try:
        await judge(**kw)
    except RuntimeError as e:
        assert "no parsable output" in str(e)


class TestJudgeFunctionsPassItToDispatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(JUDGES))
    async def test_threads_the_session_id(self, captured, name):
        await _run(JUDGES[name], scenario_session_id="sess-123")
        assert captured["scenario_session_id"] == "sess-123"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(JUDGES))
    async def test_defaults_to_none(self, captured, name):
        await _run(JUDGES[name])
        assert "task" in captured, "the judge never reached dispatch"
        assert captured["scenario_session_id"] is None


# --- endpoints -------------------------------------------------------------

_DRIFT_RESULT = (
    {"per_turn": [], "session": compute_session_rollup([])},
    "gemini-2.5-pro",
)

#: path -> (patch target, request body, value the patched judge returns)
ENDPOINTS = {
    "/api/v1/drift/judge": (
        "app.api.v1.endpoints.drift.judge_session",
        {"transcript": TRANSCRIPT},
        _DRIFT_RESULT,
    ),
    "/api/v1/drift/judge-labels": (
        "app.api.v1.endpoints.drift.judge_session_labels_only",
        {"transcript": TRANSCRIPT},
        ([], "gemini-2.5-pro"),
    ),
    "/api/v1/language-quality/judge": (
        "app.api.v1.endpoints.language_quality.judge_session",
        {"transcript": TRANSCRIPT},
        (LanguageJudgmentResult(per_turn=[], turns_judged=0), "gemini-2.5-pro"),
    ),
    "/api/v1/feedback-groundedness/judge": (
        "app.api.v1.endpoints.feedback_groundedness.judge_feedback",
        {"transcript": TRANSCRIPT, "claims": CLAIMS},
        ([], "gemini-2.5-pro"),
    ),
    "/api/v1/recall-quality/judge": (
        "app.api.v1.endpoints.recall_quality.judge_recall",
        {
            "counsellor_turn": "Tell me about work.",
            "client_reply": "I sewed.",
            "selected": [FACT],
        },
        (None, "gemini-2.5-pro"),
    ),
    "/api/v1/filler-quality/judge": (
        "app.api.v1.endpoints.filler_quality.judge_session",
        {"observations": [{"turn_index": 0, "filler_text": "hmm"}]},
        (FillerJudgmentResult(), "gemini-2.5-flash"),
    ),
}


@pytest.fixture
def client():
    """API key read from settings, as the middleware does — a literal 401s
    wherever a local .env supplies a different one."""
    from app.core.config import settings
    from app.main import app

    test_client = TestClient(app)
    test_client.headers.update({"x-api-key": settings.API.X_API_KEY})
    return test_client


class TestEndpointsAcceptIt:
    @pytest.mark.parametrize("path", list(ENDPOINTS))
    def test_request_field_reaches_the_judge(self, client, path):
        target, body, result = ENDPOINTS[path]
        with patch(target, new=AsyncMock(return_value=result)) as judge:
            response = client.post(
                path, json={**body, "scenario_session_id": "sess-123"}
            )

        assert response.status_code == 200, response.text
        assert judge.await_args.kwargs["scenario_session_id"] == "sess-123"

    @pytest.mark.parametrize("path", list(ENDPOINTS))
    def test_omitting_it_still_judges(self, client, path):
        """Today's ally-be sends no session id; the request must be judged
        exactly as before."""
        target, body, result = ENDPOINTS[path]
        with patch(target, new=AsyncMock(return_value=result)) as judge:
            response = client.post(path, json=body)

        assert response.status_code == 200, response.text
        assert judge.await_args.kwargs["scenario_session_id"] is None
