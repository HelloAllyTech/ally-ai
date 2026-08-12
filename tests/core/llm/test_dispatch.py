"""Tests for the multi-provider structured-generation dispatch.

Two classes of behaviour matter here. First, provider RESOLUTION: the answer model is
admin-selectable, so a mis-resolved provider means an admin changes the model in the UI
and nothing happens. Second, the fact that a fallback is REPORTED — an answer generated
by a different model than the one configured has to be explainable afterwards from the
conversation log.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from app.core.llm import dispatch
from app.exceptions.custom_exceptions import LLMInvocationFailedException


class Answer(BaseModel):
    answer: str = ""
    confidence: float = 0.0


@pytest.fixture(autouse=True)
def reset_clients():
    dispatch.reset_clients()
    yield
    dispatch.reset_clients()


@pytest.fixture
def both_keys():
    """Both providers configured."""
    with (
        patch.object(dispatch.settings.ANTHROPIC, "API_KEY", "anthropic-key"),
        patch.object(dispatch.settings.GEMINI, "API_KEY", "gemini-key"),
    ):
        yield


class TestProviderNaming:
    @pytest.mark.parametrize(
        "given,expected",
        [
            ("anthropic", "anthropic"),
            ("ANTHROPIC", "anthropic"),
            ("  claude ", "anthropic"),
            ("gemini", "gemini"),
            # ally-be stores Gemini as `google` in llm_configs and the admin dropdown.
            # If this alias stops resolving, an admin who selects Gemini silently gets
            # the default provider instead.
            ("google", "gemini"),
            ("google-genai", "gemini"),
            ("openai", None),
            ("", None),
            (None, None),
        ],
    )
    def test_canonical_provider(self, given, expected):
        assert dispatch.canonical_provider(given) == expected

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("claude-sonnet-4-6", "anthropic"),
            ("claude-haiku-4-5-20251001", "anthropic"),
            ("gemini-2.5-pro", "gemini"),
            ("models/gemini-2.5-flash", "gemini"),
            ("gpt-4o-mini", None),
            (None, None),
        ],
    )
    def test_infer_provider_from_model(self, model, expected):
        assert dispatch.infer_provider_from_model(model) == expected


class TestResolveTarget:
    def test_honours_explicit_provider_and_model(self, both_keys):
        provider, model, fell_back = dispatch.resolve_target("gemini", "gemini-2.5-pro")
        assert (provider, model, fell_back) == ("gemini", "gemini-2.5-pro", None)

    def test_infers_provider_from_model_alone(self, both_keys):
        provider, model, fell_back = dispatch.resolve_target(None, "claude-sonnet-4-6")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-6"
        assert fell_back is None

    def test_ignores_model_belonging_to_another_provider(self, both_keys):
        """
        An explicit provider wins; a model from a different provider cannot ride along.
        """
        provider, model, _ = dispatch.resolve_target("gemini", "claude-sonnet-4-6")
        assert provider == "gemini"
        assert model != "claude-sonnet-4-6"

    def test_falls_back_and_reports_when_key_missing(self):
        """A missing key degrades to the other provider rather than failing outright.

        For a worker waiting on WhatsApp a different model beats no answer — but the
        substitution must be reported, never silent.
        """
        with (
            patch.object(dispatch.settings.ANTHROPIC, "API_KEY", None),
            patch.object(dispatch.settings.GEMINI, "API_KEY", "gemini-key"),
        ):
            provider, model, fell_back = dispatch.resolve_target(
                "anthropic", "claude-sonnet-4-6"
            )

        assert provider == "gemini"
        assert fell_back == "anthropic"
        # The requested model belonged to the unavailable provider, so it must not carry
        # over.
        assert model != "claude-sonnet-4-6"

    def test_raises_when_no_provider_configured(self):
        with (
            patch.object(dispatch.settings.ANTHROPIC, "API_KEY", None),
            patch.object(dispatch.settings.GEMINI, "API_KEY", None),
        ):
            with pytest.raises(LLMInvocationFailedException) as exc:
                dispatch.resolve_target("anthropic")

        assert "no llm provider is configured" in str(exc.value).lower()


def anthropic_response(*, tool_input=None, stop_reason="tool_use", usage=(11, 7)):
    content = []
    if tool_input is not None:
        content.append(
            SimpleNamespace(
                type="tool_use", name=dispatch._STRUCTURED_TOOL_NAME, input=tool_input
            )
        )
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
    )


class TestAnthropicPath:
    @pytest.mark.asyncio
    async def test_forces_the_structured_tool(self, both_keys):
        """Structured output is enforced by the API, not requested in prose.

        Asking for JSON in the prompt yields JSON *most* of the time, which is the worst
        failure rate to have — the occasional prose preamble only appears under load.
        """
        client = SimpleNamespace(
            messages=SimpleNamespace(
                create=AsyncMock(
                    return_value=anthropic_response(
                        tool_input={"answer": "Ask directly.", "confidence": 0.8}
                    )
                )
            )
        )

        with patch.object(dispatch, "_get_anthropic_client", return_value=client):
            parsed, meta = await dispatch.generate_structured(
                schema=Answer,
                prompt="How should I ask about intent?",
                provider="anthropic",
                model="claude-sonnet-4-6",
            )

        assert parsed == Answer(answer="Ask directly.", confidence=0.8)
        assert meta == {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "fell_back_from": None,
        }

        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["tool_choice"] == {
            "type": "tool",
            "name": dispatch._STRUCTURED_TOOL_NAME,
        }
        assert kwargs["tools"][0]["input_schema"] == Answer.model_json_schema()
        # Anthropic rejects a request without max_tokens outright.
        assert kwargs["max_tokens"] == dispatch.DEFAULT_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_computes_total_tokens(self, both_keys):
        """Anthropic reports input/output but NO total; unlike Gemini it must be summed.

        Without this the cost dashboard reads every Claude call as zero.
        """
        client = SimpleNamespace(
            messages=SimpleNamespace(
                create=AsyncMock(
                    return_value=anthropic_response(
                        tool_input={"answer": "x"}, usage=(100, 25)
                    )
                )
            )
        )

        with (
            patch.object(dispatch, "_get_anthropic_client", return_value=client),
            patch("app.core.llm_usage.emitter.emit_llm_usage_blocking") as emit,
        ):
            await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="anthropic",
                task="whatsapp_rag_answer",
            )

        assert emit.call_args.kwargs["usage"] == (100, 25, 125)

    @pytest.mark.asyncio
    async def test_raises_when_tool_block_missing(self, both_keys):
        """
        Hitting max_tokens mid-tool-call truncates the block; that is a failed call.
        """
        client = SimpleNamespace(
            messages=SimpleNamespace(
                create=AsyncMock(
                    return_value=anthropic_response(
                        tool_input=None, stop_reason="max_tokens"
                    )
                )
            )
        )

        with patch.object(dispatch, "_get_anthropic_client", return_value=client):
            with pytest.raises(LLMInvocationFailedException) as exc:
                await dispatch.generate_structured(
                    schema=Answer, prompt="q", provider="anthropic"
                )

        assert "max_tokens" in str(exc.value)

    @pytest.mark.asyncio
    async def test_schema_violation_is_a_failure_not_a_partial_object(self, both_keys):
        """A response that does not fit the schema fails.

        Coercing it into a half-populated object would send a confident empty answer to
        a worker.
        """
        client = SimpleNamespace(
            messages=SimpleNamespace(
                create=AsyncMock(
                    return_value=anthropic_response(
                        tool_input={"confidence": "not-a-number"}
                    )
                )
            )
        )

        with patch.object(dispatch, "_get_anthropic_client", return_value=client):
            with pytest.raises(LLMInvocationFailedException):
                await dispatch.generate_structured(
                    schema=Answer, prompt="q", provider="anthropic"
                )


class TestGeminiPath:
    @pytest.mark.asyncio
    async def test_passes_response_schema_and_returns_parsed(self, both_keys):
        response = SimpleNamespace(
            parsed=Answer(answer="From the passages.", confidence=0.5),
            usage_metadata=SimpleNamespace(
                prompt_token_count=30,
                candidates_token_count=10,
                total_token_count=40,
            ),
        )
        client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=AsyncMock(return_value=response)
                )
            )
        )

        with (
            patch.object(dispatch, "_get_gemini_client", return_value=client),
            patch("app.core.llm_usage.emitter.emit_llm_usage_blocking") as emit,
        ):
            parsed, meta = await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="google",  # alias, on purpose
                model="gemini-2.5-flash",
                task="whatsapp_rag_answer",
            )

        assert parsed == Answer(answer="From the passages.", confidence=0.5)
        assert meta["provider"] == "gemini"
        assert meta["model"] == "gemini-2.5-flash"

        config = client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.response_schema is Answer
        assert config.response_mime_type == "application/json"
        # Gemini DOES report a total, so it is used rather than recomputed.
        assert emit.call_args.kwargs["usage"] == (30, 10, 40)

    @pytest.mark.asyncio
    async def test_uses_the_async_client(self, both_keys):
        """The call must go through client.aio, not the blocking client.

        This path serves concurrent WhatsApp messages against a 25s budget; the blocking
        Gemini client would stall the event loop and serialise them behind one another.
        """
        response = SimpleNamespace(parsed=Answer(answer="a"), usage_metadata=None)
        generate = AsyncMock(return_value=response)
        client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        )

        with patch.object(dispatch, "_get_gemini_client", return_value=client):
            await dispatch.generate_structured(
                schema=Answer, prompt="q", provider="gemini"
            )

        generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_nothing_parsable(self, both_keys):
        response = SimpleNamespace(parsed=None, usage_metadata=None)
        client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=AsyncMock(return_value=response)
                )
            )
        )

        with patch.object(dispatch, "_get_gemini_client", return_value=client):
            with pytest.raises(LLMInvocationFailedException):
                await dispatch.generate_structured(
                    schema=Answer, prompt="q", provider="gemini"
                )


class TestFallbackIsVisible:
    @pytest.mark.asyncio
    async def test_metadata_names_the_model_that_actually_ran(self):
        """The returned meta must describe reality, not the request.

        ally-be stores this on wa_messages.retrieval_meta and the admin log displays it,
        so an answer that changed because a fallback kicked in stays explainable.
        """
        response = SimpleNamespace(parsed=Answer(answer="a"), usage_metadata=None)
        client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=AsyncMock(return_value=response)
                )
            )
        )

        with (
            patch.object(dispatch.settings.ANTHROPIC, "API_KEY", None),
            patch.object(dispatch.settings.GEMINI, "API_KEY", "gemini-key"),
            patch.object(dispatch, "_get_gemini_client", return_value=client),
        ):
            _, meta = await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="anthropic",
                model="claude-sonnet-4-6",
            )

        assert meta["provider"] == "gemini"
        assert meta["fell_back_from"] == "anthropic"
