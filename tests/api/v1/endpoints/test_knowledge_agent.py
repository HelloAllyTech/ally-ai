"""Tests for the /knowledge-agent/answer endpoint.

The central contract: all three outcomes are 200. A decline is a correct result, not an
error — ally-be sends the worker a real reply in every case, and only a genuine failure
should divert it onto the fallback path.
"""

from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.knowledge_agent.schemas import AnswerIntent, DeclineReason
from app.exceptions.custom_exceptions import (
    EmbeddingFailedException,
    LLMInvocationFailedException,
    VectorDBSearchFailedException,
)
from tests.api.v1.endpoints.base import BaseAPITest

SERVICE = "app.core.knowledge_agent.agent.KnowledgeAgentService"


class _ResolvedKeyAPITest(BaseAPITest):
    """A client whose API key is read from settings rather than hardcoded.

    AuthMiddleware compares the header against ``settings.API.X_API_KEY``. Which source
    supplies that differs by environment — in CI it is the env vars tests/conftest.py
    sets, on a developer machine a repo-root .env wins — so any literal in a test is
    right in one place and 401s in the other. Reading the same value the middleware
    reads is correct in both.
    """

    @pytest.fixture
    def client(self):
        from app.core.config import settings
        from app.main import app

        test_client = TestClient(app)
        test_client.headers.update({"x-api-key": settings.API.X_API_KEY})
        return test_client


def agent_result(**overrides):
    result = {
        "intent": AnswerIntent.ANSWER,
        "answer": "Ask directly about intent and plan.",
        "language": "en",
        "confidence": 0.8,
        "citations": [
            {
                "passage_number": 1,
                "chunk_id": str(uuid4()),
                "document_id": str(uuid4()),
                "document_title": "WHO mhGAP Intervention Guide",
                "page_from": 44,
                "page_to": 44,
                "section_path": "Depression > Assessment",
                "source_url": "",
                "similarity": 0.61,
            }
        ],
        "decline_reason": DeclineReason.NONE,
        "retrieval": {
            "top_k": 8,
            "min_similarity": 0.35,
            "decline_similarity": 0.42,
            "hit_count": 3,
            "top_similarity": 0.61,
            "passages_used": 2,
            "query_language": "en",
            "translated_query": None,
            "unsupported": False,
        },
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "prompt_version": "v1",
    }
    result.update(overrides)
    return result


class TestAnswerEndpoint(_ResolvedKeyAPITest):
    def test_answer_success(self, client: TestClient):
        with patch(f"{SERVICE}.answer") as mock_answer:
            mock_answer.return_value = agent_result()
            response = client.post(
                "/api/v1/knowledge-agent/answer",
                json={"question": "How do I ask about suicidal intent?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "answer"
        assert data["citations"][0]["page_from"] == 44
        # The model that actually ran is echoed, so a fallback stays traceable.
        assert data["provider"] == "anthropic"
        assert data["model"] == "claude-sonnet-4-6"
        assert data["prompt_version"] == "v1"

    def test_decline_is_200_not_an_error(self, client: TestClient):
        """A decline is a correct outcome.

        A non-2xx would push ally-be onto its fallback message and lose the specific
        'not in my material' wording.
        """
        with patch(f"{SERVICE}.answer") as mock_answer:
            mock_answer.return_value = agent_result(
                intent=AnswerIntent.DECLINE,
                answer="",
                citations=[],
                decline_reason=DeclineReason.BELOW_THRESHOLD,
                provider="",
                model="",
            )
            response = client.post(
                "/api/v1/knowledge-agent/answer", json={"question": "unrelated"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "decline"
        assert data["decline_reason"] == "below_threshold"

    def test_clarify_is_200(self, client: TestClient):
        with patch(f"{SERVICE}.answer") as mock_answer:
            mock_answer.return_value = agent_result(
                intent=AnswerIntent.CLARIFY,
                answer="Which age group do you mean?",
                citations=[],
            )
            response = client.post(
                "/api/v1/knowledge-agent/answer",
                json={"question": "help with a client"},
            )

        assert response.status_code == 200
        assert response.json()["intent"] == "clarify"

    def test_thresholds_are_forwarded(self, client: TestClient):
        """
        ally-be reads these from the whatsapp_bot settings row, so tuning needs no
        deploy.
        """
        with patch(f"{SERVICE}.answer") as mock_answer:
            mock_answer.return_value = agent_result()
            response = client.post(
                "/api/v1/knowledge-agent/answer",
                json={
                    "question": "q",
                    "top_k": 12,
                    "min_similarity": 0.3,
                    "decline_similarity": 0.5,
                    "max_passages": 4,
                    "translate_query": False,
                },
            )

        assert response.status_code == 200
        kwargs = mock_answer.call_args.kwargs
        assert kwargs["top_k"] == 12
        assert kwargs["min_similarity"] == 0.3
        assert kwargs["decline_similarity"] == 0.5
        assert kwargs["max_passages"] == 4
        assert kwargs["translate_query"] is False

    def test_translate_query_defaults_on(self, client: TestClient):
        """
        Default-on matters: default-off would silently degrade every non-English
        question.
        """
        with patch(f"{SERVICE}.answer") as mock_answer:
            mock_answer.return_value = agent_result()
            client.post("/api/v1/knowledge-agent/answer", json={"question": "q"})

        assert mock_answer.call_args.kwargs["translate_query"] is True

    def test_history_is_forwarded(self, client: TestClient):
        with patch(f"{SERVICE}.answer") as mock_answer:
            mock_answer.return_value = agent_result()
            client.post(
                "/api/v1/knowledge-agent/answer",
                json={
                    "question": "what about for children?",
                    "history": [
                        {"role": "user", "content": "How do I ask about intent?"},
                        {"role": "assistant", "content": "Ask directly."},
                    ],
                },
            )

        history = mock_answer.call_args.kwargs["history"]
        assert len(history) == 2
        assert history[0]["content"] == "How do I ask about intent?"

    @pytest.mark.parametrize(
        "body",
        [
            {"question": ""},
            {"question": "q", "top_k": 0},
            {"question": "q", "top_k": 51},
            {"question": "q", "min_similarity": 1.5},
            {"question": "q", "max_answer_chars": 50},
        ],
    )
    def test_validation(self, client: TestClient, body):
        assert (
            client.post("/api/v1/knowledge-agent/answer", json=body).status_code == 422
        )

    def test_embedding_failure_is_502(self, client: TestClient):
        with patch(f"{SERVICE}.answer") as mock_answer:
            mock_answer.side_effect = EmbeddingFailedException("nope")
            response = client.post(
                "/api/v1/knowledge-agent/answer", json={"question": "q"}
            )
        assert response.status_code == 502

    def test_vector_db_failure_is_503(self, client: TestClient):
        with patch(f"{SERVICE}.answer") as mock_answer:
            mock_answer.side_effect = VectorDBSearchFailedException("down")
            response = client.post(
                "/api/v1/knowledge-agent/answer", json={"question": "q"}
            )
        assert response.status_code == 503

    def test_llm_failure_is_502_not_a_decline(self, client: TestClient):
        """A model outage is NOT 'the corpus does not cover this'.

        Reporting it as a decline would quietly fill the unanswered-question queue with
        questions the corpus answers perfectly well.
        """
        with patch(f"{SERVICE}.answer") as mock_answer:
            mock_answer.side_effect = LLMInvocationFailedException("model down")
            response = client.post(
                "/api/v1/knowledge-agent/answer", json={"question": "q"}
            )
        assert response.status_code == 502
