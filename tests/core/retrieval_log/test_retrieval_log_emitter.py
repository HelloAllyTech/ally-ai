"""Unit tests for the retrieval-log emitter: payload shape and the privacy default.

The emitter has one job that matters beyond plumbing — marking whose words the query is. A
health worker's question rendered in an admin panel is not a cosmetic mistake, so the default
is tested rather than assumed.
"""

import json
from unittest.mock import patch

from app.core.retrieval_log import emitter as emitter_module
from app.core.retrieval_log.emitter import MAX_PASSAGES, emit_retrieval_log


def _sent(mock_send):
    assert mock_send.called, "nothing was sent"
    return json.loads(mock_send.call_args[0][0])["data"]["retrieval_log"]


def _emit(**over):
    args = {
        "corpus": "whatsapp_qa",
        "consumer": "whatsapp_bot",
        "query": "how do I support someone refusing medication?",
        "min_similarity": 0.35,
        "requested_limit": 8,
        "returned_count": 3,
        "latency_ms": 412,
        "hits": [
            {"chunk_id": "c1", "document_id": "d1", "similarity": 0.62},
            {"chunk_id": "c2", "document_id": "d1", "similarity": 0.41},
        ],
        "disposition": "answered",
        "query_language": "ta",
    }
    args.update(over)
    with patch.object(emitter_module, "_queue_url", return_value="https://sqs/test"), patch.object(
        emitter_module, "_send_blocking"
    ) as send:
        emit_retrieval_log(**args)
    return send


class TestPrivacyDefault:
    def test_a_query_is_sensitive_unless_told_otherwise(self):
        # Wrongly marking an operator's query sensitive costs one withheld string; wrongly
        # marking a worker's question public renders it in a console.
        send = _emit()
        assert _sent(send)["query_sensitive"] is True

    def test_an_operator_query_can_opt_out_explicitly(self):
        send = _emit(query_sensitive=False, consumer="admin_preview")
        assert _sent(send)["query_sensitive"] is False


class TestPayload:
    def test_carries_both_thresholds(self):
        # The search floor and the refusal threshold are different numbers, and the bot can
        # retrieve a passage it then refuses to answer from.
        send = _emit(min_similarity=0.35, decline_similarity=0.5)
        body = _sent(send)
        assert body["min_similarity"] == 0.35
        assert body["decline_similarity"] == 0.5

    def test_reports_the_disposition_the_caller_decided(self):
        send = _emit(returned_count=0, disposition="declined_below_threshold")
        body = _sent(send)
        assert body["disposition"] == "declined_below_threshold"
        assert body["returned_count"] == 0

    def test_ranks_passages_as_retrieval_returned_them(self):
        send = _emit()
        passages = _sent(send)["passages"]
        assert [p["rank"] for p in passages] == [1, 2]
        assert passages[0]["similarity"] == 0.62

    def test_drops_a_hit_with_no_ids_rather_than_inventing_them(self):
        send = _emit(hits=[{"similarity": 0.5}, {"chunk_id": "c1", "document_id": "d1"}])
        assert [p["chunk_id"] for p in _sent(send)["passages"]] == ["c1"]

    def test_caps_the_passages_one_message_can_carry(self):
        hits = [
            {"chunk_id": f"c{i}", "document_id": "d1", "similarity": 0.4}
            for i in range(MAX_PASSAGES + 20)
        ]
        send = _emit(hits=hits)
        assert len(_sent(send)["passages"]) == MAX_PASSAGES

    def test_travels_as_its_own_message_type(self):
        # Same queue as llm_usage — dispatched by type on the other side, so there is no new
        # queue to provision. The knowledge base already shipped once against a queue that did
        # not exist and 500'd every upload for a fortnight.
        with patch.object(
            emitter_module, "_queue_url", return_value="https://sqs/test"
        ), patch.object(emitter_module, "_send_blocking") as send:
            emit_retrieval_log(
                corpus="whatsapp_qa",
                consumer="whatsapp_bot",
                query="q",
                min_similarity=0.35,
                requested_limit=8,
                returned_count=1,
                latency_ms=10,
            )
        assert json.loads(send.call_args[0][0])["message_type"] == "retrieval_log"


class TestNeverInTheWay:
    def test_no_ops_without_a_queue_configured(self):
        with patch.object(emitter_module, "_queue_url", return_value=""), patch.object(
            emitter_module, "_send_blocking"
        ) as send:
            emit_retrieval_log(
                corpus="whatsapp_qa",
                consumer="whatsapp_bot",
                query="q",
                min_similarity=0.35,
                requested_limit=8,
                returned_count=1,
                latency_ms=10,
            )
        send.assert_not_called()

    def test_says_nothing_about_an_empty_query(self):
        send = _emit(query="   ")
        send.assert_not_called()

    def test_swallows_a_send_failure(self):
        # Analytics are worth a table, not a worker's answer.
        with patch.object(
            emitter_module, "_queue_url", return_value="https://sqs/test"
        ), patch.object(
            emitter_module, "_send_blocking", side_effect=RuntimeError("sqs down")
        ):
            emit_retrieval_log(
                corpus="whatsapp_qa",
                consumer="whatsapp_bot",
                query="q",
                min_similarity=0.35,
                requested_limit=8,
                returned_count=1,
                latency_ms=10,
            )
