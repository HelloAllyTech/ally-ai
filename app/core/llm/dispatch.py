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

import re
from typing import Any, Dict, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from app.core.config import settings
from app.exceptions.custom_exceptions import LLMInvocationFailedException
from app.utils.logger import get_logger

logger = get_logger(__name__)

TSchema = TypeVar("TSchema", bound=BaseModel)

PROVIDER_GEMINI = "gemini"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"

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
    "gpt": PROVIDER_OPENAI,
}

SUPPORTED_PROVIDERS = (PROVIDER_GEMINI, PROVIDER_ANTHROPIC, PROVIDER_OPENAI)

# The provider every fallback lands on.
#
# Not a preference between vendors: OpenAI is the only one whose key is a
# REQUIRED setting in this service (`OpenAISettings.API_KEY`), while the other
# two are Optional so the service can boot without them. A fallback pointing at
# a provider whose credentials can be absent — or present and expired, which is
# what actually happened — is not a fallback.
FALLBACK_PROVIDER = PROVIDER_OPENAI

# Anthropic REQUIRES max_tokens on every request — there is no "as long as it needs"
# default, and omitting it is a 400. Sized for a WhatsApp-length answer plus its JSON
# envelope: replies are composed to 1600 characters, so this is generous headroom rather
# than a limit the model will hit.
DEFAULT_MAX_TOKENS = 2048

# Used when a caller passes `max_tokens=None` — "no explicit cap" — and the
# provider will not accept that. Anthropic REQUIRES max_tokens on every request,
# so there is no way to express "as long as it needs"; Gemini and OpenAI simply
# have the field omitted. Sized for a long structured judgment rather than a
# chat reply, because the callers that pass None are the judges, whose output is
# a per-turn array over a whole session.
UNCAPPED_ANTHROPIC_MAX_TOKENS = 16384

# The forced-tool name for Anthropic structured output. Arbitrary but stable — it
# appears in the request and in the returned tool_use block, and the response parser
# looks for it.
_STRUCTURED_TOOL_NAME = "emit_result"

# HTTP statuses that mean the PROVIDER is unusable rather than the request being
# wrong, so the same call is worth one attempt somewhere else.
#
#   401/403 — the credential is dead or revoked. This is the case that started
#             the whole migration: a key that is PRESENT, so `_is_configured`
#             passes, and invalid, so every call fails. A missing-key fallback
#             cannot catch it, which is why this classification exists.
#   404     — the model id was retired under us.
#   408/429/5xx — transient or capacity.
#
# 400 is deliberately absent: the request was rejected, and retrying a malformed
# request elsewhere just fails twice, slower.
_RETRYABLE_STATUSES = frozenset({401, 403, 404, 408, 429})


def _is_retryable_provider_failure(error: BaseException) -> bool:
    """
    Whether `error` says the PROVIDER is unusable, as opposed to the response
    being unusable.

    Deliberately conservative: anything unrecognised is NOT retried. The first
    version of this defaulted to True for an error with no HTTP status, which
    swept up two content failures — a Pydantic ValidationError, and our own
    "returned no structured output" — and retried them on another provider. For
    a judge that is the worst possible behaviour: it changes the model
    (and therefore the `judgeModel` a row is filed under) because one response
    did not parse, mixing a content problem into a series as if it were an
    infrastructure one.

    So a fallback fires only for a dead credential, a retired model, capacity,
    or a request that never arrived.
    """
    # Our own exception, raised by the generators above when a provider ANSWERED
    # but the answer was unusable — no tool block, nothing parsable. It carries
    # an internal 500 for the API layer, which would otherwise read as a server
    # error here. The provider is fine; the response is not.
    if isinstance(error, LLMInvocationFailedException):
        return False

    status = (
        getattr(error, "status_code", None)
        or getattr(error, "status", None)
        or getattr(getattr(error, "response", None), "status_code", None)
    )
    if isinstance(status, int):
        return status in _RETRYABLE_STATUSES or status >= 500

    # No status: retry only for the transport-level failures, matched by class
    # name so this module needs no SDK imports. `APIConnectionError`,
    # `ConnectError`, `ReadTimeout`, `TimeoutException` and friends all qualify;
    # ValidationError and LLMInvocationFailedException deliberately do not.
    name = type(error).__name__
    return "Connection" in name or "Timeout" in name or isinstance(error, TimeoutError)


_anthropic_client = None
_gemini_client = None
_openai_client = None


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
    # `o1`/`o3`/`o4` alongside `gpt-`: the reasoning family does not carry the
    # gpt prefix, and an unrecognised model id silently resolves to the default
    # provider instead of the one whose name is on it.
    if name.startswith("gpt") or re.match(r"^o\d", name):
        return PROVIDER_OPENAI
    return None


def _is_configured(provider: str) -> bool:
    """Whether a provider has a usable API key."""
    if provider == PROVIDER_ANTHROPIC:
        return bool(settings.ANTHROPIC.API_KEY)
    if provider == PROVIDER_GEMINI:
        return bool(settings.GEMINI.API_KEY)
    if provider == PROVIDER_OPENAI:
        return bool(settings.OPENAI.API_KEY)
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
    configured default. A requested provider with no API key falls back rather than
    failing the question outright, because for a worker waiting on WhatsApp a slightly
    different model is a much better outcome than no answer. It is logged at WARNING
    and surfaced in the metadata, never swallowed.

    The fallback prefers OpenAI specifically, rather than "whichever other provider
    has a key". Picking arbitrarily made the substitute depend on iteration order of
    a tuple, so which model answered a question could change with an unrelated edit
    to that tuple — and the whole point of reporting `fell_back_from` is that the
    substitution is explainable afterwards.
    """
    requested = canonical_provider(provider) or infer_provider_from_model(model)
    default_provider = (
        canonical_provider(settings.KNOWLEDGE_AGENT.DEFAULT_PROVIDER)
        or FALLBACK_PROVIDER
    )
    target = requested or default_provider

    if _is_configured(target):
        return target, _resolve_model(target, model), None

    ordered = (
        FALLBACK_PROVIDER,
        *(p for p in SUPPORTED_PROVIDERS if p != FALLBACK_PROVIDER),
    )
    alternative = next((p for p in ordered if p != target and _is_configured(p)), None)
    if alternative is None:
        raise LLMInvocationFailedException(
            "No LLM provider is configured — set OPENAI__API_KEY, "
            "ANTHROPIC__API_KEY or GEMINI__API_KEY."
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
    # The knowledge agent's own default belongs to whichever provider it names,
    # so it is only the right answer for that provider.
    if provider == canonical_provider(settings.KNOWLEDGE_AGENT.DEFAULT_PROVIDER):
        return settings.KNOWLEDGE_AGENT.DEFAULT_MODEL
    if provider == PROVIDER_GEMINI:
        # Reuse the analytics agent's answer model — the cheap-and-fast tier
        # rather than the reasoning tier.
        return settings.ANALYTICS_AGENT.ANSWER_MODEL
    if provider == PROVIDER_OPENAI:
        return settings.KNOWLEDGE_AGENT.FALLBACK_MODEL
    return settings.KNOWLEDGE_AGENT.ANTHROPIC_MODEL


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


def _get_openai_client():
    """Lazily build the async OpenAI client; clear error when unusable."""
    global _openai_client
    if _openai_client is None:
        if not settings.OPENAI.API_KEY:
            raise LLMInvocationFailedException(
                "OPENAI__API_KEY is not configured — cannot run an OpenAI model."
            )
        try:
            from openai import AsyncOpenAI  # imported lazily; optional dependency
        except Exception as e:  # pragma: no cover - optional dependency
            raise LLMInvocationFailedException(
                "The openai package is not installed — cannot run an OpenAI model."
            ) from e
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI.API_KEY)
    return _openai_client


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
    max_tokens: Optional[int],
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
        "max_tokens": max_tokens or UNCAPPED_ANTHROPIC_MAX_TOKENS,
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
    max_tokens: Optional[int],
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
    }
    # Omitted entirely when the caller passes None, rather than substituted with
    # a default. A judge emits one element per conversational turn, so a cap
    # sized for a chat reply truncates the array mid-object on a long session —
    # which surfaces as a schema validation failure, not as "too long".
    if max_tokens:
        config_kwargs["max_output_tokens"] = max_tokens
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


async def _generate_openai(
    *,
    schema: Type[TSchema],
    prompt: str,
    model: str,
    temperature: float,
    system: Optional[str],
    max_tokens: Optional[int],
    task: Optional[str],
) -> TSchema:
    """
    Structured output from OpenAI via a json_schema response format.

    The third distinct mechanism in this module, which is the reason the module
    exists: Gemini takes a `response_schema`, Anthropic needs a forced tool, and
    OpenAI takes a JSON Schema on `response_format`. None of the three
    translates to the others, and a caller should not have to know which one ran.

    `strict` is deliberately NOT set. Strict mode requires every property to be
    required and `additionalProperties: false` throughout, which a Pydantic
    schema with any optional field does not satisfy — and the failure is a 400
    on the whole call rather than a laxer match, so opting in would break the
    schemas most likely to use it. Validation still happens on our side, the
    same `schema.model_validate` every provider goes through.
    """
    client = _get_openai_client()

    messages: list[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": _STRUCTURED_TOOL_NAME,
                "schema": schema.model_json_schema(),
            },
        },
    }
    if max_tokens:
        # max_completion_tokens, not max_tokens: the reasoning family rejects
        # the older name outright.
        kwargs["max_completion_tokens"] = max_tokens
    # The reasoning family accepts only the default temperature, and passing one
    # is a 400 rather than something it ignores.
    if temperature and not re.match(r"^(o\d|gpt-5)", model.strip().lower()):
        kwargs["temperature"] = temperature

    response = await client.chat.completions.create(**kwargs)

    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) or (
            prompt_tokens + completion_tokens
        )
        _emit_usage(
            PROVIDER_OPENAI,
            model,
            task,
            (prompt_tokens, completion_tokens, total_tokens),
        )

    choices = getattr(response, "choices", None) or []
    content = (
        getattr(getattr(choices[0], "message", None), "content", None)
        if choices
        else None
    )
    if not content:
        # Reachable when the completion hits its token cap mid-object, which
        # returns a truncated string rather than an error.
        finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
        logger.error(
            "OpenAI returned no usable content (finish_reason=%s)", finish_reason
        )
        raise LLMInvocationFailedException(
            f"OpenAI returned no structured output (finish_reason={finish_reason})."
        )

    return schema.model_validate_json(content)


def _fallback_after_failure(
    failed_provider: str, *, never_fallback: bool
) -> Optional[Tuple[str, str]]:
    """
    Where to retry after a provider failed, or None when retrying is wrong.

    None means: the caller opted out, the provider that failed IS the fallback
    (retrying it would fail again for the same reason), or the fallback holds no
    key here — in which case a second attempt fails for a second, more
    confusing reason.
    """
    if never_fallback:
        return None
    if failed_provider == FALLBACK_PROVIDER:
        return None
    if not _is_configured(FALLBACK_PROVIDER):
        return None
    return FALLBACK_PROVIDER, _resolve_model(FALLBACK_PROVIDER, None)


async def generate_structured(
    *,
    schema: Type[TSchema],
    prompt: str,
    task: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    system: Optional[str] = None,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
    never_fallback: bool = False,
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
        max_tokens: Output cap, or None for no explicit cap. None is for output
            whose length is a property of the input rather than a choice — a
            judge emits one element per conversational turn, so a cap sized for
            a chat reply truncates the array mid-object and surfaces as a schema
            validation failure rather than as "too long". Anthropic has no way
            to express it and gets UNCAPPED_ANTHROPIC_MAX_TOKENS instead.

        never_fallback: Refuse to retry elsewhere when the selected provider
            fails. For a call whose whole point is exercising ONE named model,
            where a substitute would make the result a lie rather than a
            degradation.

    Returns:
        (parsed, meta) where meta is {"provider", "model", "fell_back_from"}
        describing what ACTUALLY ran, so a fallback or an admin model change stays
        traceable. Callers that STORE the model — the judges write it to
        `judgeModel`, which is part of a judgment row's uniqueness key — must
        record `meta["model"]` and not their own setting, or a fallback lands
        mislabelled and silently pollutes a pinned series.

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

    async def attempt(prov: str, mdl: str) -> TSchema:
        return await _GENERATORS[prov](
            schema=schema,
            prompt=prompt,
            model=mdl,
            temperature=temperature,
            system=system,
            max_tokens=max_tokens,
            task=task,
        )

    try:
        parsed = await attempt(resolved_provider, resolved_model)
    except Exception as e:
        # `resolve_target` only falls back when a key is ABSENT. An expired key
        # is present, so it passes that check and fails here instead — which is
        # exactly the incident this module was changed for. One retry on the
        # fallback provider turns a dead credential into degraded quality.
        retry = _fallback_after_failure(
            resolved_provider, never_fallback=never_fallback
        )
        if retry is None or not _is_retryable_provider_failure(e):
            # Includes schema validation failures: a response that does not fit the
            # schema is a failed call, not something to coerce into a
            # half-populated object and pass downstream.
            logger.exception(
                "Structured generation failed on %s/%s: %s",
                resolved_provider,
                resolved_model,
                type(e).__name__,
            )
            if isinstance(e, LLMInvocationFailedException):
                raise
            raise LLMInvocationFailedException(
                f"{resolved_provider} failed to produce a valid response."
            ) from e

        logger.warning(
            "%s/%s failed (%s: %s); retrying on %s/%s. The answer will be "
            "generated by a different model than requested.",
            resolved_provider,
            resolved_model,
            type(e).__name__,
            e,
            retry[0],
            retry[1],
        )
        parsed = await attempt(*retry)
        meta["provider"], meta["model"] = retry
        meta["fell_back_from"] = resolved_provider

    return parsed, meta


_GENERATORS = {
    PROVIDER_ANTHROPIC: _generate_anthropic,
    PROVIDER_GEMINI: _generate_gemini,
    PROVIDER_OPENAI: _generate_openai,
}


def reset_clients() -> None:
    """Drop the cached SDK clients. For tests; not used on the request path."""
    global _anthropic_client, _gemini_client, _openai_client
    _anthropic_client = None
    _gemini_client = None
    _openai_client = None
