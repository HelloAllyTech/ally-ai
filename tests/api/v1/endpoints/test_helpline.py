"""Tests for the /helpline/risk and /helpline/turn endpoints.

The central contract: both are ALWAYS 200. A failure is reported in `failed`, never as a
5xx — a 5xx would push ally-be onto its generic error path on a message that may be a
crisis. The service is patched; its behaviour is covered in tests/core/helpline.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.api.v1.endpoints.base import BaseAPITest

SERVICE = "app.core.helpline.service.HelplineCopilotService"


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


RISK_URL = "/api/v1/helpline/risk"
TURN_URL = "/api/v1/helpline/turn"

RISK_PAYLOAD = {
    "message": "honestly there's no point anymore",
    "recent": [
        {"role": "talker", "content": "everything is so heavy"},
        {"role": "listener", "content": "That sounds really hard."},
    ],
    "language": "en",
}

TURN_PAYLOAD = {
    "messages": [
        {"role": "talker", "content": "I failed my exam and I can't tell my parents"},
        {"role": "listener", "content": "That sounds like a lot to carry."},
        {"role": "talker", "content": "I just feel so alone"},
    ],
    "rolling_summary": "Worried about exam results.",
    "language": "en",
    "include_nudge": True,
}


def risk_result(**overrides):
    result = {
        "is_crisis": True,
        "confidence": 0.72,
        "signal": "no point anymore",
        "subject": "SELF",
        "failed": False,
        "provider": "openai",
        "model": "gpt-4o-mini",
    }
    result.update(overrides)
    return result


def turn_result(**overrides):
    result = {
        "stage": "Understand",
        "nudge": "Try reflecting the loneliness before asking about the exam.",
        "suggestions": [
            {
                "text": "It sounds like you're carrying this on your own.",
                "skill_key": "empathy",
            },
            {"text": "What has felt hardest about it?", "skill_key": "verbal"},
        ],
        "failed": False,
        "provider": "openai",
        "model": "gpt-4o-mini",
    }
    result.update(overrides)
    return result


class TestRiskEndpoint(_ResolvedKeyAPITest):
    def test_success_returns_the_contract_shape(self, client: TestClient):
        with patch(f"{SERVICE}.classify_risk", new_callable=AsyncMock) as mock:
            mock.return_value = risk_result()
            response = client.post(RISK_URL, json=RISK_PAYLOAD)

        assert response.status_code == 200
        assert response.json() == {
            "is_crisis": True,
            "confidence": 0.72,
            "signal": "no point anymore",
            "subject": "SELF",
            "failed": False,
            "provider": "openai",
            "model": "gpt-4o-mini",
        }

    def test_request_fields_reach_the_service(self, client: TestClient):
        prompts = {"ally_ai_helpline_risk_classify": {"prompt": "x {message}"}}
        with patch(f"{SERVICE}.classify_risk", new_callable=AsyncMock) as mock:
            mock.return_value = risk_result()
            client.post(
                RISK_URL, json={**RISK_PAYLOAD, "language": "hi", "prompts": prompts}
            )

        mock.assert_awaited_once()
        assert mock.call_args.args == ("honestly there's no point anymore",)
        assert mock.call_args.kwargs == {
            "recent": RISK_PAYLOAD["recent"],
            "language": "hi",
            "prompts": prompts,
        }

    def test_only_the_message_is_required(self, client: TestClient):
        with patch(f"{SERVICE}.classify_risk", new_callable=AsyncMock) as mock:
            mock.return_value = risk_result(
                is_crisis=False, signal="", subject="UNCLEAR"
            )
            response = client.post(RISK_URL, json={"message": "hello"})

        assert response.status_code == 200
        assert mock.call_args.kwargs == {
            "recent": [],
            "language": "en",
            "prompts": None,
        }

    def test_a_classifier_failure_is_200_with_failed_true(self, client: TestClient):
        failed = risk_result(
            is_crisis=False, confidence=0.0, signal="", subject="UNCLEAR", failed=True
        )
        with patch(f"{SERVICE}.classify_risk", new_callable=AsyncMock) as mock:
            mock.return_value = failed
            response = client.post(RISK_URL, json=RISK_PAYLOAD)

        assert response.status_code == 200
        assert response.json()["failed"] is True
        assert response.json()["is_crisis"] is False

    def test_an_unexpected_exception_is_200_with_failed_true_not_a_500(
        self, client: TestClient
    ):
        with patch(f"{SERVICE}.classify_risk", new_callable=AsyncMock) as mock:
            mock.side_effect = RuntimeError("something exploded")
            response = client.post(RISK_URL, json=RISK_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert data["failed"] is True
        assert data["is_crisis"] is False
        assert data["signal"] == ""
        assert data["subject"] == "UNCLEAR"

    def test_a_malformed_service_result_is_also_a_reported_failure(
        self, client: TestClient
    ):
        with patch(f"{SERVICE}.classify_risk", new_callable=AsyncMock) as mock:
            mock.return_value = risk_result(subject="NOT_A_SUBJECT")
            response = client.post(RISK_URL, json=RISK_PAYLOAD)

        assert response.status_code == 200
        assert response.json()["failed"] is True

    def test_the_exception_text_is_not_logged(self, client: TestClient):
        """A validation error quotes its input, which here is a talker's message."""
        with (
            patch(f"{SERVICE}.classify_risk", new_callable=AsyncMock) as mock,
            patch("app.api.v1.endpoints.helpline.logger") as mock_logger,
        ):
            mock.side_effect = RuntimeError("input was: no point anymore")
            client.post(RISK_URL, json=RISK_PAYLOAD)

        rendered = " ".join(
            str(a) for c in mock_logger.error.call_args_list for a in c.args
        )
        assert "RuntimeError" in rendered
        assert "no point" not in rendered

    def test_rejects_a_request_without_the_api_key(self, client: TestClient):
        with patch(f"{SERVICE}.classify_risk", new_callable=AsyncMock) as mock:
            response = client.post(
                RISK_URL, json=RISK_PAYLOAD, headers={"x-api-key": ""}
            )

        assert response.status_code == 401
        mock.assert_not_awaited()

    def test_rejects_a_request_with_no_api_key_header_at_all(self, client: TestClient):
        client.headers.pop("x-api-key", None)

        response = client.post(RISK_URL, json=RISK_PAYLOAD)

        assert response.status_code == 401

    def test_rejects_a_wrong_api_key(self, client: TestClient):
        response = client.post(
            RISK_URL, json=RISK_PAYLOAD, headers={"x-api-key": "not-the-key"}
        )

        assert response.status_code == 401

    def test_an_unknown_role_is_a_422_not_silently_accepted(self, client: TestClient):
        payload = {**RISK_PAYLOAD, "recent": [{"role": "bot", "content": "hi"}]}

        response = client.post(RISK_URL, json=payload)

        assert response.status_code == 422


class TestTurnEndpoint(_ResolvedKeyAPITest):
    def test_success_returns_the_contract_shape(self, client: TestClient):
        with patch(f"{SERVICE}.copilot_turn", new_callable=AsyncMock) as mock:
            mock.return_value = turn_result()
            response = client.post(TURN_URL, json=TURN_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert data["stage"] == "Understand"
        assert data["nudge"].startswith("Try reflecting")
        assert data["suggestions"] == [
            {
                "text": "It sounds like you're carrying this on your own.",
                "skill_key": "empathy",
            },
            {"text": "What has felt hardest about it?", "skill_key": "verbal"},
        ]
        assert data["failed"] is False
        assert (data["provider"], data["model"]) == ("openai", "gpt-4o-mini")

    def test_request_fields_reach_the_service(self, client: TestClient):
        prompts = {"ally_ai_helpline_copilot_turn": {"prompt": "x {conversation}"}}
        with patch(f"{SERVICE}.copilot_turn", new_callable=AsyncMock) as mock:
            mock.return_value = turn_result()
            client.post(
                TURN_URL, json={**TURN_PAYLOAD, "language": "ta", "prompts": prompts}
            )

        mock.assert_awaited_once()
        assert mock.call_args.args == (TURN_PAYLOAD["messages"],)
        assert mock.call_args.kwargs == {
            "rolling_summary": "Worried about exam results.",
            "language": "ta",
            "include_nudge": True,
            "prompts": prompts,
        }

    def test_defaults_when_only_messages_are_sent(self, client: TestClient):
        with patch(f"{SERVICE}.copilot_turn", new_callable=AsyncMock) as mock:
            mock.return_value = turn_result(nudge="")
            response = client.post(
                TURN_URL, json={"messages": TURN_PAYLOAD["messages"]}
            )

        assert response.status_code == 200
        assert mock.call_args.kwargs == {
            "rolling_summary": "",
            "language": "en",
            "include_nudge": False,
            "prompts": None,
        }

    def test_an_empty_suggestion_list_is_a_normal_200(self, client: TestClient):
        # Everything may be filtered out; the caller must not backfill it.
        with patch(f"{SERVICE}.copilot_turn", new_callable=AsyncMock) as mock:
            mock.return_value = turn_result(suggestions=[], nudge="", stage="")
            response = client.post(TURN_URL, json=TURN_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert data["suggestions"] == []
        assert data["stage"] == ""
        assert data["failed"] is False

    def test_a_turn_failure_is_200_with_failed_true(self, client: TestClient):
        with patch(f"{SERVICE}.copilot_turn", new_callable=AsyncMock) as mock:
            mock.return_value = turn_result(
                stage="", nudge="", suggestions=[], failed=True, provider="", model=""
            )
            response = client.post(TURN_URL, json=TURN_PAYLOAD)

        assert response.status_code == 200
        assert response.json()["failed"] is True
        assert response.json()["suggestions"] == []

    def test_an_unexpected_exception_is_200_with_failed_true_not_a_500(
        self, client: TestClient
    ):
        with patch(f"{SERVICE}.copilot_turn", new_callable=AsyncMock) as mock:
            mock.side_effect = RuntimeError("something exploded")
            response = client.post(TURN_URL, json=TURN_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert data["failed"] is True
        assert data["suggestions"] == []
        assert data["nudge"] == ""
        assert data["stage"] == ""

    def test_a_malformed_service_result_is_also_a_reported_failure(
        self, client: TestClient
    ):
        with patch(f"{SERVICE}.copilot_turn", new_callable=AsyncMock) as mock:
            mock.return_value = turn_result(stage="Wrap-up")
            response = client.post(TURN_URL, json=TURN_PAYLOAD)

        assert response.status_code == 200
        assert response.json()["failed"] is True

    def test_rejects_a_request_without_the_api_key(self, client: TestClient):
        with patch(f"{SERVICE}.copilot_turn", new_callable=AsyncMock) as mock:
            response = client.post(
                TURN_URL, json=TURN_PAYLOAD, headers={"x-api-key": ""}
            )

        assert response.status_code == 401
        mock.assert_not_awaited()

    def test_rejects_a_wrong_api_key(self, client: TestClient):
        response = client.post(
            TURN_URL, json=TURN_PAYLOAD, headers={"x-api-key": "not-the-key"}
        )

        assert response.status_code == 401

    def test_messages_are_required(self, client: TestClient):
        response = client.post(TURN_URL, json={"language": "en"})

        assert response.status_code == 422


@pytest.mark.parametrize("url", [RISK_URL, TURN_URL])
def test_both_routes_are_registered_under_the_helpline_prefix(url):
    from app.main import app

    paths = {route.path for route in app.routes}

    assert url in paths
