"""Token-usage extraction helpers (never raise; return None when unavailable)."""

from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple


def extract_usage_from_aimessage(
    response: Any,
) -> Optional[Tuple[int, int, int]]:
    """Return (prompt, completion, total) from a LangChain AIMessage, or None.

    Works for plain `_invoke_llm` (output_class=None) and `bind_tools` calls
    where an AIMessage is returned. Structured-output calls return the parsed
    object (no usage) — use `normalize_callback_usage` for those.
    """
    try:
        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt = int(usage.get("input_tokens", 0) or 0)
            completion = int(usage.get("output_tokens", 0) or 0)
            total = int(usage.get("total_tokens", 0) or 0) or (prompt + completion)
            if prompt or completion or total:
                return prompt, completion, total

        meta = getattr(response, "response_metadata", None) or {}
        token_usage = meta.get("token_usage") or meta.get("usage") or {}
        if token_usage:
            prompt = int(token_usage.get("prompt_tokens", 0) or 0)
            completion = int(token_usage.get("completion_tokens", 0) or 0)
            total = int(token_usage.get("total_tokens", 0) or 0) or (
                prompt + completion
            )
            if prompt or completion or total:
                return prompt, completion, total
    except Exception:
        return None
    return None


def normalize_callback_usage(cb: Any) -> Optional[Tuple[int, int, int]]:
    """Sum a UsageMetadataCallbackHandler's per-model usage into one tuple.

    `cb.usage_metadata` is a dict: {model_name: {input_tokens, output_tokens,
    total_tokens, ...}}. Returns None when nothing was captured.
    """
    try:
        per_model = getattr(cb, "usage_metadata", None)
        if not per_model:
            return None
        prompt = completion = total = 0
        for u in per_model.values():
            prompt += int(u.get("input_tokens", 0) or 0)
            completion += int(u.get("output_tokens", 0) or 0)
            total += int(u.get("total_tokens", 0) or 0)
        if not total:
            total = prompt + completion
        if prompt or completion or total:
            return prompt, completion, total
    except Exception:
        return None
    return None


def extract_usage_details(
    cb: Any = None, response: Any = None
) -> Tuple[Optional[int], Optional[Dict[str, int]]]:
    """Return (cached_tokens, metadata) for a LangChain call, never raising.

    Reads the same source the token counts come from — the usage callback when
    it captured anything, else the AIMessage — so the detail always describes
    the counts it travels with.

    * ``cached_tokens``: LangChain's ``input_token_details.cache_read``, the
      prompt tokens served from the provider's cache. A SUBSET of the prompt
      count for both OpenAI and Gemini. None when the provider reported no
      such detail, which is not the same as reporting 0.
    * ``metadata``: ``{"reasoning_tokens": N}`` from
      ``output_token_details.reasoning`` when N > 0, else None. Informational
      only: LangChain's output count already includes reasoning for OpenAI and
      thinking for Gemini, so nothing here is added to it.
    """
    try:
        per_model = getattr(cb, "usage_metadata", None) if cb is not None else None
        if per_model and isinstance(per_model, Mapping):
            usages = list(per_model.values())
        else:
            usages = [getattr(response, "usage_metadata", None)]
        usages = [u for u in usages if isinstance(u, Mapping)]

        cached: Optional[int] = None
        reasoning = 0
        if usages:
            for u in usages:
                cache_read = (u.get("input_token_details") or {}).get("cache_read")
                if isinstance(cache_read, int):
                    cached = (cached or 0) + cache_read
                thinking = (u.get("output_token_details") or {}).get("reasoning")
                if isinstance(thinking, int):
                    reasoning += thinking
        else:
            # Raw OpenAI shape, for a response LangChain did not normalise.
            meta = getattr(response, "response_metadata", None)
            meta = meta if isinstance(meta, Mapping) else {}
            token_usage = meta.get("token_usage") or meta.get("usage") or {}
            if isinstance(token_usage, Mapping):
                prompt_details = token_usage.get("prompt_tokens_details") or {}
                cache_read = prompt_details.get("cached_tokens")
                if isinstance(cache_read, int):
                    cached = cache_read
                completion_details = token_usage.get("completion_tokens_details") or {}
                thinking = completion_details.get("reasoning_tokens")
                if isinstance(thinking, int):
                    reasoning = thinking

        metadata = {"reasoning_tokens": reasoning} if reasoning > 0 else None
        return cached, metadata
    except Exception:
        return None, None


def make_usage_callback() -> Optional[Any]:
    """Return a UsageMetadataCallbackHandler instance, or None if unavailable.

    Guarded import so a langchain-core version without the handler degrades to
    no usage capture (rather than breaking the LLM call path).
    """
    try:
        from langchain_core.callbacks import UsageMetadataCallbackHandler

        return UsageMetadataCallbackHandler()
    except Exception:
        return None
