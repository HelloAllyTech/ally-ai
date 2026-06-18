"""Best-effort emitter for LLM token-usage events (ally-ai).

Sends an ``llm_usage`` SQS message (wire shape mirrors turn_metrics:
``data.llm_usage = {...}``) to the queue ally-be consumes. Never blocks or
fails the LLM call path. No-ops unless ``settings.LLM_USAGE.QUEUE_URL`` is set.

The FastAPI process has no SQS client by default, so we lazily create the
shared singleton on first use (idempotent — reuses the worker's if present).
"""

import asyncio
import json
import time
from typing import Optional, Tuple

from app.core.config import settings
from app.core.queue.sqs_queue_client import SQSQueueClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _enabled() -> bool:
    cfg = getattr(settings, "LLM_USAGE", None)
    return bool(cfg and getattr(cfg, "ENABLED", False) and getattr(cfg, "QUEUE_URL", ""))


def _build_body(
    service: str,
    provider: str,
    model: str,
    task: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    audio_ms: Optional[int] = None,
    characters: Optional[int] = None,
    room_id: Optional[str] = None,
    scenario_id: Optional[int] = None,
) -> str:
    env = None
    try:
        env = settings.ENV.ENV
    except Exception:
        env = None
    return json.dumps(
        {
            "message_type": "llm_usage",
            "timestamp": int(time.time()),
            "room_id": room_id,
            "data": {
                "llm_usage": {
                    "service": service,
                    "provider": provider or "unknown",
                    "model": model or "unknown",
                    "task": task,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "audio_ms": audio_ms,
                    "characters": characters,
                    "env": env,
                    "scenario_id": scenario_id,
                }
            },
        }
    )


def _send_blocking(body: str) -> None:
    """Send via the shared boto3 SQS client (lazily created). Never raises."""
    try:
        SQSQueueClient.create_client()  # idempotent
        client = SQSQueueClient.get_client()
        client.send_message(
            QueueUrl=settings.LLM_USAGE.QUEUE_URL, MessageBody=body
        )
    except Exception:
        logger.debug("llm_usage send failed (best-effort)", exc_info=True)


def _has_quantity(
    service: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    audio_ms: Optional[int],
    characters: Optional[int],
) -> bool:
    if service == "stt":
        return bool(audio_ms)
    if service == "tts":
        return bool(characters)
    return bool(prompt_tokens or completion_tokens or total_tokens)


def emit_ai_usage(
    service: str,
    provider: Optional[str],
    model: Optional[str],
    task: Optional[str],
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    audio_ms: Optional[int] = None,
    characters: Optional[int] = None,
    room_id: Optional[str] = None,
    scenario_id: Optional[int] = None,
    blocking: bool = False,
) -> None:
    """Best-effort emit for any AI service ('llm' | 'stt' | 'tts'). Never raises."""
    try:
        if not task or not _enabled():
            return
        if room_id and room_id.startswith("preview-"):
            return
        if not _has_quantity(
            service, prompt_tokens, completion_tokens, total_tokens, audio_ms, characters
        ):
            return
        body = _build_body(
            service,
            provider,
            model,
            task,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            audio_ms=audio_ms,
            characters=characters,
            room_id=room_id,
            scenario_id=scenario_id,
        )
        if blocking:
            _send_blocking(body)
        else:
            # Off the hot path: run the blocking boto3 send in a thread.
            asyncio.create_task(asyncio.to_thread(_send_blocking, body))
    except Exception:
        logger.debug("emit_ai_usage skipped (best-effort)", exc_info=True)


def emit_llm_usage(
    provider: Optional[str],
    model: Optional[str],
    task: Optional[str],
    usage: Optional[Tuple[int, int, int]],
    room_id: Optional[str] = None,
    scenario_id: Optional[int] = None,
) -> None:
    """Back-compat LLM helper (async). Forwards to emit_ai_usage(service='llm')."""
    if not usage:
        return
    prompt, completion, total = usage
    emit_ai_usage(
        "llm",
        provider,
        model,
        task,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        room_id=room_id,
        scenario_id=scenario_id,
    )


def emit_llm_usage_blocking(
    provider: Optional[str],
    model: Optional[str],
    task: Optional[str],
    usage: Optional[Tuple[int, int, int]],
    room_id: Optional[str] = None,
    scenario_id: Optional[int] = None,
) -> None:
    """Back-compat LLM helper (sync) for non-async call sites (e.g. drift judge)."""
    if not usage:
        return
    prompt, completion, total = usage
    emit_ai_usage(
        "llm",
        provider,
        model,
        task,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        room_id=room_id,
        scenario_id=scenario_id,
        blocking=True,
    )
