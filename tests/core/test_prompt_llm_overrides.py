"""Tests for per-prompt LLM model/temperature override resolution (ally-ai)."""

import pytest

from app.prompts.resolver import _clamp_temperature, get_backend_llm_overrides


class TestGetBackendLlmOverrides:
    def test_reads_provider_model_and_temperature(self):
        prompts = {
            "ally_ai_analysis_counselor_analysis": {
                "prompt": "x",
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.4,
            }
        }
        assert get_backend_llm_overrides("analysis/counselor_analysis", prompts) == (
            "openai",
            "gpt-4o",
            0.4,
        )

    def test_string_entry_or_missing_returns_none(self):
        assert get_backend_llm_overrides(
            "nudge/nudge", {"ally_ai_nudge_nudge": "txt"}
        ) == (None, None, None)
        assert get_backend_llm_overrides("nudge/nudge", {}) == (None, None, None)
        assert get_backend_llm_overrides("nudge/nudge", None) == (None, None, None)

    def test_temperature_clamped_and_invalid_handled(self):
        assert _clamp_temperature(5) == 2.0
        assert _clamp_temperature(-1) == 0.0
        assert _clamp_temperature("hot") is None
        assert _clamp_temperature(None) is None
        assert _clamp_temperature(0) == 0.0

    def test_key_alignment_matches_resolver(self):
        # The override key must be the same ally_ai_-prefixed code the resolver
        # uses for text, so model/temp align with whichever entry supplied text.
        prompts = {"ally_ai_summary_summary": {"prompt": "p", "model": "gpt-5"}}
        assert get_backend_llm_overrides("summary/summary", prompts) == (
            None,
            "gpt-5",
            None,
        )


class TestIsOpenAiModelGuard:
    def test_guard(self):
        # Imported lazily: the client module pulls in langchain_openai, which may
        # be absent in a bare local env. Skip cleanly if so.
        try:
            from app.core.text_generations.openai_text_generation_client import (
                _is_openai_model,
            )
        except Exception:  # pragma: no cover - env-dependent
            pytest.skip("openai client module not importable in this env")

        assert _is_openai_model("gpt-4o") is True
        assert _is_openai_model("gpt-5-mini") is True
        assert _is_openai_model("o3-mini") is True
        assert _is_openai_model("gemini-2.5-pro") is False
        assert _is_openai_model("claude-sonnet-4-6") is False
        assert _is_openai_model(None) is False


class TestModelSupportsTemperature:
    def test_guard(self):
        try:
            from app.core.text_generations.openai_text_generation_client import (
                _model_supports_temperature,
            )
        except Exception:  # pragma: no cover - env-dependent
            pytest.skip("openai client module not importable in this env")

        assert _model_supports_temperature("gpt-4o") is True
        assert _model_supports_temperature("gpt-4.1-mini") is True
        assert _model_supports_temperature("gpt-5") is False
        assert _model_supports_temperature("o1") is False
        assert _model_supports_temperature(None) is True


class TestResolveProvider:
    def test_explicit_provider_wins(self):
        try:
            from app.core.text_generations.openai_text_generation_client import (
                _resolve_provider,
            )
        except Exception:  # pragma: no cover - env-dependent
            pytest.skip("client module not importable in this env")

        # Explicit provider wins over model-name inference.
        assert _resolve_provider("gemini-2.0-flash", "openai") == "openai"
        assert _resolve_provider("gpt-4o", "gemini") == "gemini"
        # Unrunnable explicit provider -> None (override ignored).
        assert _resolve_provider("claude-3", "anthropic") is None
        # Inference fallback when no explicit provider.
        assert _resolve_provider("gpt-4o", None) == "openai"
        assert _resolve_provider("gemini-2.5-pro", None) == "gemini"
        assert _resolve_provider("claude-3", None) is None
        assert _resolve_provider(None, None) is None


class TestGeminiClientBranch:
    def test_gemini_override_builds_gemini_client(self, monkeypatch):
        try:
            import app.core.text_generations.openai_text_generation_client as mod
        except Exception:  # pragma: no cover - env-dependent
            pytest.skip("client module not importable in this env")

        # Stub the Gemini builder so the test doesn't need a real key/SDK call.
        sentinel = object()
        built = {}

        def fake_build(model, temperature):
            built["model"] = model
            built["temperature"] = temperature
            return sentinel

        monkeypatch.setattr(mod, "_build_gemini_client", fake_build)
        mod._override_client_cache.clear()

        client = mod.OpenAITextGenerationClient.get_or_create_client(
            model="gemini-2.0-flash", temperature=0.4, provider="gemini"
        )
        assert client is sentinel
        assert built == {"model": "gemini-2.0-flash", "temperature": 0.4}
        # Cached by (provider, model, temperature).
        assert mod._override_client_cache[("gemini", "gemini-2.0-flash", 0.4)] is (
            sentinel
        )

    def test_gemini_unavailable_falls_back_to_default_openai(self, monkeypatch):
        try:
            import app.core.text_generations.openai_text_generation_client as mod
        except Exception:  # pragma: no cover - env-dependent
            pytest.skip("client module not importable in this env")

        # Gemini builder returns None (missing key/dep) -> must not raise; falls
        # through to the default OpenAI client path.
        monkeypatch.setattr(mod, "_build_gemini_client", lambda m, t: None)

        class _DefaultClient:
            model_name = "gpt-4o-mini"

        default_sentinel = _DefaultClient()
        monkeypatch.setattr(
            mod.OpenAITextGenerationClient,
            "get_client",
            staticmethod(lambda: default_sentinel),
        )
        mod._override_client_cache.clear()

        # No temperature override -> degrades to the exact default client.
        client = mod.OpenAITextGenerationClient.get_or_create_client(
            model="gemini-2.0-flash", temperature=None, provider="gemini"
        )
        assert client is default_sentinel
