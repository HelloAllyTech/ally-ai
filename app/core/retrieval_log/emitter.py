"""Best-effort emitter for retrieval-log events (ally-ai).

Reports a retrieval this service performed to ally-be, which owns the retrieval log. Wire
shape mirrors ``llm_usage`` (``data.retrieval_log = {...}``) and it travels on the SAME queue,
so there is no new infrastructure to provision — a lesson from the knowledge base shipping
against a queue that did not exist, which 500'd every upload for a fortnight.

WHY THIS EXISTS. The WhatsApp Q&A bot retrieves here, in one call, and never passes through
ally-be's search path — the only writer of that log. So the platform's highest-volume RAG
surface was the one nothing measured, while the character corpus had a judge and a precision
curve. The floor that governs the bot was the number with least evidence behind it.

PRIVACY. ``query_sensitive`` marks a query as someone's own words rather than an operator's.
A health worker's question is PHI-adjacent by default here, so callers on that path MUST pass
True; ally-be treats an ABSENT flag as sensitive, and every read surface withholds the text.
Never blocks or fails the retrieval path, and no-ops unless the queue is configured.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import settings
from app.core.queue.sqs_queue_client import SQSQueueClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

#: Cap on passages reported per retrieval. Generous next to any real top-k, and it keeps one
#: malformed call from writing hundreds of rows on the other side.
MAX_PASSAGES = 60


def _queue_url() -> str:
    """Shares llm_usage's queue: one consumer, dispatched on `message_type`."""
    cfg = getattr(settings, "LLM_USAGE", None)
    return str(getattr(cfg, "QUEUE_URL", "") or "") if cfg else ""


def _enabled() -> bool:
    return bool(_queue_url())


def _shape_passages(hits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Passage rows from raw search hits, ranked as retrieval returned them."""
    shaped: List[Dict[str, Any]] = []
    for index, hit in enumerate(list(hits)[:MAX_PASSAGES]):
        chunk_id = hit.get("chunk_id") or hit.get("id")
        document_id = hit.get("document_id")
        if not chunk_id or not document_id:
            continue
        shaped.append(
            {
                "chunk_id": str(chunk_id),
                "document_id": str(document_id),
                "rank": index + 1,
                "similarity": float(hit.get("similarity") or 0.0),
            }
        )
    return shaped


def _send_blocking(body: str) -> None:
    """Send via the shared boto3 SQS client (lazily created). Never raises."""
    try:
        SQSQueueClient.create_client()  # idempotent
        client = SQSQueueClient.get_client()
        client.send_message(QueueUrl=_queue_url(), MessageBody=body)
    except Exception:
        logger.debug("retrieval_log send failed (best-effort)", exc_info=True)


def emit_retrieval_log(
    corpus: str,
    consumer: str,
    query: str,
    *,
    min_similarity: float,
    requested_limit: int,
    returned_count: int,
    latency_ms: int,
    hits: Optional[Sequence[Dict[str, Any]]] = None,
    decline_similarity: Optional[float] = None,
    disposition: Optional[str] = None,
    query_language: Optional[str] = None,
    query_sensitive: bool = True,
    session_id: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
) -> None:
    """Report one retrieval. Never raises, never blocks the caller.

    `query_sensitive` defaults to True rather than False on purpose: the cost of wrongly
    marking an operator's query sensitive is that a panel withholds one string, and the cost of
    wrongly marking a worker's question public is that it renders in an admin console.
    """
    try:
        if not _enabled() or not corpus or not consumer or not (query or "").strip():
            return

        body = json.dumps(
            {
                "message_type": "retrieval_log",
                "timestamp": int(time.time()),
                "data": {
                    "retrieval_log": {
                        "corpus": corpus,
                        "consumer": consumer,
                        "query": query,
                        "query_sensitive": bool(query_sensitive),
                        "query_language": query_language,
                        "min_similarity": float(min_similarity),
                        "decline_similarity": (
                            None if decline_similarity is None else float(decline_similarity)
                        ),
                        "disposition": disposition,
                        "requested_limit": int(requested_limit),
                        "fetch_limit": int(requested_limit),
                        "returned_count": int(returned_count),
                        "latency_ms": int(latency_ms),
                        "tags": [str(t) for t in (tags or [])],
                        "session_id": session_id,
                        "passages": _shape_passages(hits or []),
                    }
                },
            }
        )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop (a sync caller): send inline rather than dropping the row.
            _send_blocking(body)
            return
        asyncio.create_task(asyncio.to_thread(_send_blocking, body))
    except Exception:
        logger.debug("emit_retrieval_log skipped (best-effort)", exc_info=True)
