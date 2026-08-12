"""Multi-provider structured generation for the knowledge agent.

One entry point — ``generate_structured`` — takes a Pydantic schema and returns a
validated instance of it, whichever provider served the call. Exists because the
WhatsApp bot's answer model is ADMIN-SELECTABLE per prompt (ally-be's prompt management
stores provider/model/temperature per prompt row), so the calling code cannot know at
import time which SDK will run.

Both providers are driven with structured output rather than free-text-then-parse, but
they reach it differently, and that difference is the whole reason this module exists:

  * Gemini takes a ``response_schema`` directly and returns a parsed object.
  * Anthropic has NO equivalent. Structured output goes through a single tool the model
    is FORCED
    to call, and the tool's input is the structured payload.

Calls are made on each SDK's ASYNC client. The older analytics agent uses the blocking
Gemini client inside an async endpoint, which stalls the event loop for the duration of
the call; that is tolerable for an admin screen with one user, but this path serves
concurrent WhatsApp messages against a 25-second budget, so it cannot afford to
serialise them behind one another.

Provider selection never silently substitutes a model without saying so: the provider
and model that ACTUALLY ran come back in the metadata, are stored on
wa_messages.retrieval_meta by ally-be, and are what the admin conversation log displays.
An answer that changed because a fallback kicked in must be explainable after the fact.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from app.core.config import settings
from app.exceptions.custom_exceptions import LLMInvocationFailedException
from app.utils.logger import get_logger

logger = get_logger(__name__)

TSchema = TypeVar("TSchema", bound=BaseModel)

PROVIDER_GEMINI = "gemini"
PROVIDER_ANTHROPIC = "anthropic"

# Alternative spellings accepted for a provider, mapped to the canonical name.
#
# `google` matters for interoperability, not tidiness: ally-be's LLM_CONFIG_SCHEMA and
# admin dropdown store Gemini as `google`, and its llm-model-registry treats the two as
# one via `canonicalProvider`. A prompt override arriving here as `google` must resolve,
# or an admin who picks Gemini in the UI silently gets the default provider instead.
_PROVIDER_ALIASES: Dict[str, str] = {
    "google": PROVIDER_GEMINI,
    "google-genai": PROVIDER_GEMINI,
    "claude": PROVIDER_ANTHROPIC,
}

SUPPORTED_PROVIDERS = (PROVIDER_GEMINI, PROVIDER_ANTHROPIC)

# Anthropic REQUIRES max_tokens on every request — there is no "as long as it needs"
# default, and omitting it is a 400. Sized for a WhatsApp-length answer plus its JSON
# envelope: replies are composed to 1600 characters, so this is generous headroom rather
# than a limit the model will hit.
DEFAULT_MAX_TOKENS = 2048

# The forced-tool name for Anthropic structured output. Arbitrary but stable — it
# appears in the request and in the returned tool_use block, and the response parser
# looks for it.
_STRUCTURED_TOOL_NAME = "emit_result"

_anthropic_client = None
_gemini_client = None


def canonical_provider(provider: Optional[str]) -> Optional[str]:
    """Normalise a provider name, resolving known aliases. None when unrecognised."""
    if not provider:
        return None
    name = provider.strip().lower()
    name = _PROVIDER_ALIASES.get(name, name)
    return name if name in SUPPORTED_PROVIDERS else None


def infer_provider_from_model(model: Optional[str]) -> Optional[str]:
    """
    Infer the provider from a model id, for overrides that set a model but no provider.
    """
    if not model:
        return None
    name = model.strip().lower()
    if name.startswith(("claude", "anthropic")):
        return PROVIDER_ANTHROPIC
    if name.startswith(("gemini", "models/gemini")):
        return PROVIDER_GEMINI
    return None


def _is_configured(provider: str) -> bool:
    """Whether a provider has a usable API key."""
    if provider == PROVIDER_ANTHROPIC:
        return bool(settings.ANTHROPIC.API_KEY)
    if provider == PROVIDER_GEMINI:
        return bool(settings.GEMINI.API_KEY)
    return False


def resolve_target(
    provider: Optional[str] = None, model: Optional[str] = None
) -> Tuple[str, str, Optional[str]]:
    """
    Decide which (provider, model) will actually run.

    Returns ``(provider, model, fell_back_from)``. ``fell_back_from`` is the provider
    that was asked for but could not run, or None when the request was honoured — it
    exists so a fallback is reportable rather than invisible.

    Resolution order: explicit provider, then inferred from the model id, then the
    configured default. A requested provider with no API key falls back to the other
    supported provider rather than failing the question outright, because for a worker
    waiting on WhatsApp a slightly different model is a much better outcome than no
    answer. It is logged at WARNING and surfaced in the metadata, never swallowed.
    """
    requested = canonical_provider(provider) or infer_provider_from_model(model)
    default_provider = (
        canonical_provider(settings.KNOWLEDGE_AGENT.DEFAULT_PROVIDER)
        or PROVIDER_ANTHROPIC
    )
    target = requested or default_provider

    if _is_configured(target):
        return target, _resolve_model(target, model), None

    alternative = next(
        (p for p in SUPPORTED_PROVIDERS if p != target and _is_configured(p)), None
    )
    if alternative is None:
        raise LLMInvocationFailedException(
            "No LLM provider is configured — set ANTHROPIC__API_KEY or GEMINI__API_KEY."
        )

    logger.warning(
        "Provider %s has no API key configured; falling back to %s. The answer will be "
        "generated by a different model than requested.",
        target,
        alternative,
    )
    # The requested model belongs to the unavailable provider, so it cannot carry over —
    # the fallback provider's own default is used instead.
    return alternative, _resolve_model(alternative, None), target


def _resolve_model(provider: str, model: Optional[str]) -> str:
    """The model to use, honouring an override only when it belongs to this provider."""
    if model and infer_provider_from_model(model) in (provider, None):
        return model.strip()
    if provider == PROVIDER_ANTHROPIC:
        return settings.KNOWLEDGE_AGENT.DEFAULT_MODEL
    # Gemini has no dedicated default in KnowledgeAgentSettings; reuse the analytics
    # agent's answer model, which is the cheap-and-fast tier rather than the reasoning
    # tier.
    return settings.ANALYTICS_AGENT.ANSWER_MODEL


def _get_anthropic_client():
    """Lazily build the async Anthropic client; clear error when unusable."""
    global _anthropic_client
    if _anthropic_client is None:
        if not settings.ANTHROPIC.API_KEY:
            raise LLMInvocationFailedException(
                "ANTHROPIC__API_KEY is not configured — cannot run a Claude model."
            )
        try:
            from anthropic import AsyncAnthropic  # imported lazily; optional dependency
        except Exception as e:  # pragma: no cover - optional dependency
            raise LLMInvocationFailedException(
                "The anthropic package is not installed — cannot run a Claude model."
            ) from e
        _anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC.API_KEY)
    return _anthropic_client


def _get_gemini_client():
    """Lazily build the Gemini client; clear error when unusable."""
    global _gemini_client
    if _gemini_client is None:
        if not settings.GEMINI.API_KEY:
            raise LLMInvocationFailedException(
                "GEMINI__API_KEY is not configured — cannot run a Gemini model."
            )
        try:
            from google import genai  # imported lazily; optional dependency
        except Exception as e:  # pragma: no cover - optional dependency
            raise LLMInvocationFailedException(
                "The google-genai package is not installed — cannot run a Gemini model."
            ) from e
        _gemini_client = genai.Client(api_key=settings.GEMINI.API_KEY)
    return _gemini_client


def _emit_usage(
    provider: str, model: str, task: Optional[str], usage: Tuple[int, int, int]
) -> None:
    """Best-effort token-usage emission for the cost-by-model/task dashboard."""
    try:
        from app.core.llm_usage.emitter import emit_llm_usage_blocking

        emit_llm_usage_blocking(provider=provider, model=model, task=task, usage=usage)
    except Exception:  # noqa: BLE001 — usage accounting never fails a request
        logger.debug("llm usage emission skipped", exc_info=True)


async def _generate_anthropic(
    *,
    schema: Type[TSchema],
    prompt: str,
    model: str,
    temperature: float,
    system: Optional[str],
    max_tokens: int,
    task: Optional[str],
) -> TSchema:
    """
    Structured output from Claude via a single FORCED tool call.

    Anthropic has no `response_schema` equivalent, and asking for JSON in the prose
    prompt gets JSON most of the time — which is the worst failure rate to have, because
    the occasional prose preamble or trailing explanation only shows up under load.
    Declaring the schema as a tool and setting `tool_choice` to that tool makes the
    model's only available move to emit conforming arguments, so validity is enforced by
    the API rather than hoped for.
    """
    client = _get_anthropic_client()

    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "tools": [
            {
                "name": _STRUCTURED_TOOL_NAME,
                "description": (
                    "Return the result. This is the only way to respond; every "
                    "field of the schema must be populated."
                ),
                "input_schema": schema.model_json_schema(),
            }
        ],
        "tool_choice": {"type": "tool", "name": _STRUCTURED_TOOL_NAME},
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = await client.messages.create(**kwargs)

    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        # Anthropic reports input and output separately and NO total, unlike Gemini's
        # total_token_count. Compute it, or the cost dashboard reads every Claude call
        # as zero.
        _emit_usage(
            PROVIDER_ANTHROPIC,
            model,
            task,
            (prompt_tokens, completion_tokens, prompt_tokens + completion_tokens),
        )

    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "tool_use" and (
            getattr(block, "name", None) == _STRUCTURED_TOOL_NAME
        ):
            return schema.model_validate(block.input)

    # Reachable when the model hits max_tokens mid-tool-call, which truncates the block.
    stop_reason = getattr(response, "stop_reason", None)
    logger.error(
        "Anthropic returned no usable tool_use block (stop_reason=%s)", stop_reason
    )
    raise LLMInvocationFailedException(
        f"Claude returned no structured output (stop_reason={stop_reason})."
    )


async def _generate_gemini(
    *,
    schema: Type[TSchema],
    prompt: str,
    model: str,
    temperature: float,
    system: Optional[str],
    max_tokens: int,
    task: Optional[str],
) -> TSchema:
    """
    Structured output from Gemini via `response_schema`, as the analytics agent does.
    """
    from google.genai import types  # imported lazily; optional dependency

    client = _get_gemini_client()

    config_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "response_mime_type": "application/json",
        "response_schema": schema,
        "max_output_tokens": max_tokens,
    }
    if system:
        config_kwargs["system_instruction"] = system

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    um = getattr(response, "usage_metadata", None)
    if um is not None:
        prompt_tokens = int(getattr(um, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(um, "candidates_token_count", 0) or 0)
        total_tokens = int(getattr(um, "total_token_count", 0) or 0) or (
            prompt_tokens + completion_tokens
        )
        _emit_usage(
            PROVIDER_GEMINI,
            model,
            task,
            (prompt_tokens, completion_tokens, total_tokens),
        )

    parsed = getattr(response, "parsed", None)
    if parsed is None:
        logger.error("Gemini returned no parsable structured output")
        raise LLMInvocationFailedException("Gemini returned no structured output.")
    # `parsed` is already an instance of `schema`, but revalidate so both providers are
    # guaranteed to have gone through the same validation before anything downstream
    # trusts the shape.
    return schema.model_validate(
        parsed if isinstance(parsed, dict) else parsed.model_dump()
    )


async def generate_structured(
    *,
    schema: Type[TSchema],
    prompt: str,
    task: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    system: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Tuple[TSchema, Dict[str, Any]]:
    """
    Generate a validated instance of `schema` from whichever provider is selected.

    Parameters:
        schema: Pydantic model the output must conform to.
        prompt: The user-role prompt content.
        task: LLMTask value for cost accounting. Omit to skip emission.
        provider: 'anthropic' | 'gemini' (aliases accepted). None resolves from
            `model`, then the configured default.
        model: Explicit model id. Honoured only when it belongs to the resolved
            provider.
        temperature: Sampling temperature. Defaults to 0 — for a grounded answer
            over fixed passages, the same question should not produce a materially
            different answer minute to minute, or an admin comparing two
            conversation-log entries cannot tell a corpus change from a sampling
            wobble.
        system: Optional system instruction.
        max_tokens: Output cap. Required by Anthropic; applied to Gemini too so
            both providers truncate at the same point rather than differing
            invisibly.

    Returns:
        (parsed, meta) where meta is {"provider", "model", "fell_back_from"}
        describing what ACTUALLY ran, so a fallback or an admin model change stays
        traceable.

    Raises:
        LLMInvocationFailedException: If no provider is configured, the SDK is
            missing, or the provider returned nothing usable.
    """
    resolved_provider, resolved_model, fell_back_from = resolve_target(provider, model)

    meta: Dict[str, Any] = {
        "provider": resolved_provider,
        "model": resolved_model,
        "fell_back_from": fell_back_from,
    }

    generator = (
        _generate_anthropic
        if resolved_provider == PROVIDER_ANTHROPIC
        else _generate_gemini
    )

    try:
        parsed = await generator(
            schema=schema,
            prompt=prompt,
            model=resolved_model,
            temperature=temperature,
            system=system,
            max_tokens=max_tokens,
            task=task,
        )
    except LLMInvocationFailedException:
        raise
    except Exception as e:
        # Includes schema validation failures: a response that does not fit the schema
        # is a failed call, not something to coerce into a half-populated object and
        # pass downstream.
        logger.exception(
            "Structured generation failed on %s/%s: %s",
            resolved_provider,
            resolved_model,
            type(e).__name__,
        )
        raise LLMInvocationFailedException(
            f"{resolved_provider} failed to produce a valid response."
        ) from e

    return parsed, meta


def reset_clients() -> None:
    """Drop the cached SDK clients. For tests; not used on the request path."""
    global _anthropic_client, _gemini_client
    _anthropic_client = None
    _gemini_client = None
