"""Token-usage extraction helpers (never raise; return None when unavailable)."""

from typing import Any, Optional, Tuple


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
