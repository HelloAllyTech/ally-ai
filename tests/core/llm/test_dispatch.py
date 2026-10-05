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
    """Every provider configured, so resolution is about intent, not availability."""
    with (
        patch.object(dispatch.settings.ANTHROPIC, "API_KEY", "anthropic-key"),
        patch.object(dispatch.settings.GEMINI, "API_KEY", "gemini-key"),
        patch.object(dispatch.settings.OPENAI, "API_KEY", "openai-key"),
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
            ("openai", "openai"),
            ("gpt", "openai"),
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
            ("gpt-4o-mini", "openai"),
            ("gpt-5-mini", "openai"),
            # The reasoning family carries no gpt prefix; without this an
            # o-series id resolves to the DEFAULT provider rather than the one
            # whose name is on it.
            ("o3-mini", "openai"),
            ("llama-3-70b", None),
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
        """A missing key degrades rather than failing outright.

        For a worker waiting on WhatsApp a different model beats no answer — but the
        substitution must be reported, never silent.
        """
        with (
            patch.object(dispatch.settings.ANTHROPIC, "API_KEY", None),
            patch.object(dispatch.settings.GEMINI, "API_KEY", "gemini-key"),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "openai-key"),
        ):
            provider, model, fell_back = dispatch.resolve_target(
                "anthropic", "claude-sonnet-4-6"
            )

        # OpenAI specifically, not "whichever other provider has a key" — that
        # made the substitute depend on iteration order of a tuple, so which
        # model answered could change with an unrelated edit.
        assert provider == "openai"
        assert fell_back == "anthropic"
        # The requested model belonged to the unavailable provider, so it must not carry
        # over.
        assert model != "claude-sonnet-4-6"

    def test_falls_back_past_openai_when_openai_is_the_one_thats_down(self):
        with (
            patch.object(dispatch.settings.OPENAI, "API_KEY", None),
            patch.object(dispatch.settings.GEMINI, "API_KEY", "gemini-key"),
            patch.object(dispatch.settings.ANTHROPIC, "API_KEY", None),
        ):
            provider, _, fell_back = dispatch.resolve_target("openai")

        assert provider == "gemini"
        assert fell_back == "openai"

    def test_raises_when_no_provider_configured(self):
        with (
            patch.object(dispatch.settings.ANTHROPIC, "API_KEY", None),
            patch.object(dispatch.settings.GEMINI, "API_KEY", None),
            patch.object(dispatch.settings.OPENAI, "API_KEY", None),
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


def openai_client(content='{"answer": "a"}'):
    """A stand-in for the async OpenAI client's one call shape."""
    response = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason="stop"
            )
        ],
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        )
    )


def gemini_client():
    response = SimpleNamespace(parsed=Answer(answer="a"), usage_metadata=None)
    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=AsyncMock(return_value=response))
        )
    )


class TestOpenAiPath:
    """The third structured-output mechanism in this module.

    Gemini takes a response_schema, Anthropic needs a forced tool, OpenAI takes
    a JSON Schema on response_format. None translates to the others, which is
    the reason the dispatch module exists at all.
    """

    @pytest.mark.asyncio
    async def test_sends_a_json_schema_and_parses_the_content(self):
        client = openai_client('{"answer": "grounded"}')
        with (
            patch.object(dispatch.settings.OPENAI, "API_KEY", "openai-key"),
            patch.object(dispatch, "_get_openai_client", return_value=client),
        ):
            parsed, meta = await dispatch.generate_structured(
                schema=Answer, prompt="q", provider="openai", model="gpt-4o-mini"
            )

        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["response_format"]["type"] == "json_schema"
        assert parsed.answer == "grounded"
        assert meta["fell_back_from"] is None

    @pytest.mark.asyncio
    async def test_omits_temperature_for_the_reasoning_family(self):
        """gpt-5 and the o-series 400 on a custom temperature rather than
        ignoring it, so a judge passing temperature=0 must not break them."""
        client = openai_client()
        with (
            patch.object(dispatch.settings.OPENAI, "API_KEY", "openai-key"),
            patch.object(dispatch, "_get_openai_client", return_value=client),
        ):
            await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="openai",
                model="gpt-5-mini",
                temperature=0.7,
            )

        assert "temperature" not in client.chat.completions.create.await_args.kwargs

    @pytest.mark.asyncio
    async def test_omits_the_token_cap_when_uncapped(self):
        """max_tokens=None means "no explicit cap" — the judges rely on it,
        because their output length is one element per conversational turn."""
        client = openai_client()
        with (
            patch.object(dispatch.settings.OPENAI, "API_KEY", "openai-key"),
            patch.object(dispatch, "_get_openai_client", return_value=client),
        ):
            await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="openai",
                model="gpt-4o-mini",
                max_tokens=None,
            )

        kwargs = client.chat.completions.create.await_args.kwargs
        assert "max_completion_tokens" not in kwargs

    @pytest.mark.asyncio
    async def test_truncated_content_is_a_failure_not_a_partial_object(self):
        client = openai_client(content="")
        with (
            patch.object(dispatch.settings.OPENAI, "API_KEY", "openai-key"),
            patch.object(dispatch, "_get_openai_client", return_value=client),
        ):
            with pytest.raises(LLMInvocationFailedException):
                await dispatch.generate_structured(
                    schema=Answer, prompt="q", provider="openai", model="gpt-4o-mini"
                )


class TestRetryAfterFailure:
    """
    A fallback after a RUNTIME failure, not just a missing key.

    `resolve_target` only falls back when a key is absent, and an expired key is
    present — it passes that check and fails at call time instead. That is the
    exact shape of the incident this module was changed for, so a dead
    credential has to be survivable here and not only detectable.

    The classification is deliberately narrow, because these callers include the
    judges: a fallback changes the model a judgment row is filed under, so it
    must only happen when the PROVIDER is unusable, never because one response
    failed to parse.
    """

    @staticmethod
    def _err(status=None, cls_name="APIStatusError"):
        return (
            type(cls_name, (Exception,), {})(f"boom {status}")
            if status is None
            else type(cls_name, (Exception,), {"status_code": status})(f"boom {status}")
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 403, 404, 408, 429, 500, 503])
    async def test_retries_on_an_unusable_provider(self, status):
        failing = AsyncMock(side_effect=self._err(status))
        with (
            patch.object(dispatch.settings.GEMINI, "API_KEY", "g"),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "o"),
            patch.dict(dispatch._GENERATORS, {"gemini": failing}),
            patch.object(dispatch, "_get_openai_client", return_value=openai_client()),
        ):
            _, meta = await dispatch.generate_structured(
                schema=Answer, prompt="q", provider="gemini", model="gemini-2.5-pro"
            )

        assert meta["provider"] == "openai"
        assert meta["fell_back_from"] == "gemini"

    @staticmethod
    def _genai_err(code, status_name):
        """Shaped like google-genai's `ClientError`: the HTTP status is `code`,
        and `status` is the gRPC status name, not a number."""
        err = type("ClientError", (Exception,), {})(f"{code} {status_name}")
        err.code = code
        err.status = status_name
        return err

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code,status_name",
        [
            (401, "UNAUTHENTICATED"),
            (403, "PERMISSION_DENIED"),
            (429, "RESOURCE_EXHAUSTED"),
        ],
    )
    async def test_retries_a_gemini_client_error_on_a_dead_provider(
        self, code, status_name
    ):
        failing = AsyncMock(side_effect=self._genai_err(code, status_name))
        with (
            patch.object(dispatch.settings.GEMINI, "API_KEY", "g"),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "o"),
            patch.dict(dispatch._GENERATORS, {"gemini": failing}),
            patch.object(dispatch, "_get_openai_client", return_value=openai_client()),
        ):
            _, meta = await dispatch.generate_structured(
                schema=Answer, prompt="q", provider="gemini", model="gemini-2.5-pro"
            )

        assert meta["provider"] == "openai"
        assert meta["fell_back_from"] == "gemini"

    @pytest.mark.asyncio
    async def test_does_not_retry_a_gemini_client_error_for_a_rejected_request(self):
        failing = AsyncMock(side_effect=self._genai_err(400, "INVALID_ARGUMENT"))
        with (
            patch.object(dispatch.settings.GEMINI, "API_KEY", "g"),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "o"),
            patch.dict(dispatch._GENERATORS, {"gemini": failing}),
        ):
            with pytest.raises(LLMInvocationFailedException):
                await dispatch.generate_structured(
                    schema=Answer, prompt="q", provider="gemini", model="gemini-2.5-pro"
                )

    @pytest.mark.asyncio
    async def test_does_not_retry_a_rejected_request(self):
        # A 400 means the request was wrong. Retrying it elsewhere fails twice.
        failing = AsyncMock(side_effect=self._err(400))
        with (
            patch.object(dispatch.settings.GEMINI, "API_KEY", "g"),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "o"),
            patch.dict(dispatch._GENERATORS, {"gemini": failing}),
        ):
            with pytest.raises(LLMInvocationFailedException):
                await dispatch.generate_structured(
                    schema=Answer, prompt="q", provider="gemini", model="gemini-2.5-pro"
                )

    @pytest.mark.asyncio
    async def test_does_not_retry_an_unusable_RESPONSE(self):
        """The provider answered; the answer was junk. Switching models for that
        would mix a content failure into a judge's series as if it were an
        outage — and it carries an internal 500, so it has to be excluded by
        type rather than by status."""
        failing = AsyncMock(
            side_effect=LLMInvocationFailedException("returned no structured output")
        )
        with (
            patch.object(dispatch.settings.GEMINI, "API_KEY", "g"),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "o"),
            patch.dict(dispatch._GENERATORS, {"gemini": failing}),
        ):
            with pytest.raises(LLMInvocationFailedException):
                await dispatch.generate_structured(
                    schema=Answer, prompt="q", provider="gemini", model="gemini-2.5-pro"
                )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            *(
                type("APIStatusError", (Exception,), {"status_code": status})(
                    f"boom {status}"
                )
                for status in (401, 403, 404, 408, 429, 500, 503)
            ),
            type("ReadTimeout", (Exception,), {})("timed out"),
            type("APIConnectionError", (Exception,), {})("reset"),
        ],
        ids=lambda e: getattr(e, "status_code", type(e).__name__),
    )
    async def test_never_fallback_refuses_the_retry(self, error):
        """Every failure the retry would otherwise catch — dead key, retired
        model, capacity, timeout — fails the call instead, and the fallback
        provider is never called."""
        failing = AsyncMock(side_effect=error)
        fallback = AsyncMock()
        with (
            patch.object(dispatch.settings.GEMINI, "API_KEY", "g"),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "o"),
            patch.dict(dispatch._GENERATORS, {"gemini": failing, "openai": fallback}),
        ):
            with pytest.raises(LLMInvocationFailedException):
                await dispatch.generate_structured(
                    schema=Answer,
                    prompt="q",
                    provider="gemini",
                    model="gemini-2.5-pro",
                    never_fallback=True,
                )
        assert failing.await_count == 1
        fallback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_never_fallback_refuses_a_missing_key_substitute(self):
        """`resolve_target` substitutes OpenAI when the selected provider has
        no key at all. For a pinned caller that is every call until someone
        notices, each on a model whose result is discarded — so it fails
        before any provider is called."""
        gemini = AsyncMock()
        openai = AsyncMock()
        with (
            patch.object(dispatch.settings.GEMINI, "API_KEY", None),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "o"),
            patch.dict(dispatch._GENERATORS, {"gemini": gemini, "openai": openai}),
        ):
            with pytest.raises(LLMInvocationFailedException) as exc:
                await dispatch.generate_structured(
                    schema=Answer,
                    prompt="q",
                    provider="gemini",
                    model="gemini-2.5-pro",
                    never_fallback=True,
                )

        assert "gemini" in str(exc.value)
        gemini.assert_not_awaited()
        openai.assert_not_awaited()

    def test_resolve_target_refuses_a_substitute_only_when_asked(self):
        with (
            patch.object(dispatch.settings.GEMINI, "API_KEY", None),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "o"),
        ):
            # The default is unchanged: the WhatsApp bot still degrades rather
            # than leaving a worker without an answer.
            assert dispatch.resolve_target("gemini", "gemini-2.5-pro")[2] == "gemini"
            with pytest.raises(LLMInvocationFailedException):
                dispatch.resolve_target("gemini", "gemini-2.5-pro", never_fallback=True)

    @pytest.mark.asyncio
    async def test_never_fallback_does_not_change_a_healthy_call(self):
        with (
            patch.object(dispatch.settings.GEMINI, "API_KEY", "g"),
            patch.object(dispatch, "_get_gemini_client", return_value=gemini_client()),
        ):
            parsed, meta = await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="gemini",
                model="gemini-2.5-pro",
                never_fallback=True,
            )

        assert parsed.answer == "a"
        assert meta == {
            "provider": "gemini",
            "model": "gemini-2.5-pro",
            "fell_back_from": None,
        }

    @pytest.mark.asyncio
    async def test_does_not_retry_the_provider_that_just_failed(self):
        # OpenAI IS the fallback; retrying it would fail for the same reason.
        failing = AsyncMock(side_effect=self._err(500))
        with (
            patch.object(dispatch.settings.OPENAI, "API_KEY", "o"),
            patch.dict(dispatch._GENERATORS, {"openai": failing}),
        ):
            with pytest.raises(LLMInvocationFailedException):
                await dispatch.generate_structured(
                    schema=Answer, prompt="q", provider="openai", model="gpt-4o-mini"
                )
        assert failing.await_count == 1


class TestFallbackIsVisible:
    @pytest.mark.asyncio
    async def test_metadata_names_the_model_that_actually_ran(self):
        """The returned meta must describe reality, not the request.

        ally-be stores this on wa_messages.retrieval_meta and the admin log displays it,
        so an answer that changed because a fallback kicked in stays explainable.
        """
        with (
            patch.object(dispatch.settings.ANTHROPIC, "API_KEY", None),
            patch.object(dispatch.settings.GEMINI, "API_KEY", "gemini-key"),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "openai-key"),
            patch.object(dispatch, "_get_openai_client", return_value=openai_client()),
        ):
            _, meta = await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="anthropic",
                model="claude-sonnet-4-6",
            )

        # OpenAI, not Gemini: the fallback targets it deliberately because it is
        # the one provider whose key is a required setting in this service.
        assert meta["provider"] == "openai"
        assert meta["fell_back_from"] == "anthropic"

    @pytest.mark.asyncio
    async def test_falls_through_to_gemini_when_openai_is_also_down(self):
        """The fallback is a preference order, not a single hard-coded target."""
        with (
            patch.object(dispatch.settings.ANTHROPIC, "API_KEY", None),
            patch.object(dispatch.settings.OPENAI, "API_KEY", None),
            patch.object(dispatch.settings.GEMINI, "API_KEY", "gemini-key"),
            patch.object(dispatch, "_get_gemini_client", return_value=gemini_client()),
        ):
            _, meta = await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="anthropic",
                model="claude-sonnet-4-6",
            )

        assert meta["provider"] == "gemini"
        assert meta["fell_back_from"] == "anthropic"


def _gemini_client_with_usage(usage_metadata):
    response = SimpleNamespace(parsed=Answer(answer="a"), usage_metadata=usage_metadata)
    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=AsyncMock(return_value=response))
        )
    )


def _openai_client_with_usage(usage):
    response = SimpleNamespace(
        usage=usage,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"answer": "a"}'),
                finish_reason="stop",
            )
        ],
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        )
    )


class TestUsageMeasurement:
    """What the cost dashboard is told a call consumed.

    The counts must describe what was BILLED, not what was answered: a token
    priced at the output rate that never reaches `completion_tokens` is
    invisible to every cost figure built on this row.
    """

    @pytest.mark.asyncio
    async def test_gemini_thinking_is_counted_as_completion(self, both_keys):
        """Gemini reports thought tokens apart from the answer but bills them
        at the output rate, so completion is candidates + thoughts. The split
        rides in metadata so the thinking share stays visible."""
        client = _gemini_client_with_usage(
            SimpleNamespace(
                prompt_token_count=1000,
                candidates_token_count=200,
                thoughts_token_count=1500,
                # Gemini's own total already includes the thoughts.
                total_token_count=2700,
                cached_content_token_count=None,
            )
        )
        with (
            patch.object(dispatch, "_get_gemini_client", return_value=client),
            patch("app.core.llm_usage.emitter.emit_llm_usage_blocking") as emit,
        ):
            await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="gemini",
                model="gemini-2.5-pro",
                task="drift_judge",
            )

        kwargs = emit.call_args.kwargs
        assert kwargs["usage"] == (1000, 1700, 2700)
        assert kwargs["metadata"] == {
            "thoughts_tokens": 1500,
            "candidates_tokens": 200,
        }
        # Absent is reported as absent, not as a cache miss of zero.
        assert kwargs["cached_tokens"] is None

    @pytest.mark.asyncio
    async def test_gemini_cached_tokens_are_reported_as_part_of_the_prompt(
        self, both_keys
    ):
        client = _gemini_client_with_usage(
            SimpleNamespace(
                prompt_token_count=1000,
                candidates_token_count=100,
                thoughts_token_count=0,
                total_token_count=1100,
                cached_content_token_count=600,
            )
        )
        with (
            patch.object(dispatch, "_get_gemini_client", return_value=client),
            patch("app.core.llm_usage.emitter.emit_llm_usage_blocking") as emit,
        ):
            await dispatch.generate_structured(
                schema=Answer, prompt="q", provider="gemini", task="drift_judge"
            )

        kwargs = emit.call_args.kwargs
        # The prompt count is left inclusive of the cache hit: cached_tokens is
        # a subset of it, never an addition to it.
        assert kwargs["usage"] == (1000, 100, 1100)
        assert kwargs["cached_tokens"] == 600

    @pytest.mark.asyncio
    async def test_gemini_without_thinking_fields_is_unchanged(self, both_keys):
        """An SDK response with no thoughts field (a model that does not
        think) still emits exactly what it did before."""
        client = _gemini_client_with_usage(
            SimpleNamespace(
                prompt_token_count=30,
                candidates_token_count=10,
                total_token_count=40,
            )
        )
        with (
            patch.object(dispatch, "_get_gemini_client", return_value=client),
            patch("app.core.llm_usage.emitter.emit_llm_usage_blocking") as emit,
        ):
            await dispatch.generate_structured(
                schema=Answer, prompt="q", provider="gemini", task="drift_judge"
            )

        kwargs = emit.call_args.kwargs
        assert kwargs["usage"] == (30, 10, 40)
        assert kwargs["metadata"] == {"thoughts_tokens": 0, "candidates_tokens": 10}
        assert kwargs["cached_tokens"] is None

    @pytest.mark.asyncio
    async def test_openai_reasoning_is_not_counted_twice(self):
        """OpenAI's completion_tokens ALREADY include reasoning. Adding
        reasoning_tokens again would double-bill every reasoning-model call."""
        client = _openai_client_with_usage(
            SimpleNamespace(
                prompt_tokens=800,
                completion_tokens=500,
                total_tokens=1300,
                prompt_tokens_details=SimpleNamespace(cached_tokens=512),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=320),
            )
        )
        with (
            patch.object(dispatch.settings.OPENAI, "API_KEY", "openai-key"),
            patch.object(dispatch, "_get_openai_client", return_value=client),
            patch("app.core.llm_usage.emitter.emit_llm_usage_blocking") as emit,
        ):
            await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="openai",
                model="gpt-5-mini",
                task="drift_judge",
            )

        kwargs = emit.call_args.kwargs
        assert kwargs["usage"] == (800, 500, 1300)
        assert kwargs["cached_tokens"] == 512
        assert kwargs["metadata"] == {"reasoning_tokens": 320}

    @pytest.mark.asyncio
    async def test_openai_without_details_reports_nothing_extra(self):
        client = _openai_client_with_usage(
            SimpleNamespace(prompt_tokens=8, completion_tokens=5, total_tokens=13)
        )
        with (
            patch.object(dispatch.settings.OPENAI, "API_KEY", "openai-key"),
            patch.object(dispatch, "_get_openai_client", return_value=client),
            patch("app.core.llm_usage.emitter.emit_llm_usage_blocking") as emit,
        ):
            await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="openai",
                model="gpt-4o-mini",
                task="drift_judge",
            )

        kwargs = emit.call_args.kwargs
        assert kwargs["usage"] == (8, 5, 13)
        assert kwargs["cached_tokens"] is None
        assert kwargs["metadata"] is None

    @pytest.mark.asyncio
    async def test_openai_zero_reasoning_carries_no_metadata(self):
        """A non-reasoning model reports reasoning_tokens=0; a metadata blob on
        every such row would be noise."""
        client = _openai_client_with_usage(
            SimpleNamespace(
                prompt_tokens=8,
                completion_tokens=5,
                total_tokens=13,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            )
        )
        with (
            patch.object(dispatch.settings.OPENAI, "API_KEY", "openai-key"),
            patch.object(dispatch, "_get_openai_client", return_value=client),
            patch("app.core.llm_usage.emitter.emit_llm_usage_blocking") as emit,
        ):
            await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="openai",
                model="gpt-4o-mini",
                task="drift_judge",
            )

        kwargs = emit.call_args.kwargs
        # Reported and zero is a measured miss — kept as 0, not dropped.
        assert kwargs["cached_tokens"] == 0
        assert kwargs["metadata"] is None


class TestSessionAttribution:
    """`scenario_session_id` must reach the usage row, whichever provider ran.

    The judges have no room, so without it their cost cannot be tied to the
    session they judged.
    """

    @pytest.mark.asyncio
    async def test_reaches_the_emitter_on_gemini(self, both_keys):
        client = _gemini_client_with_usage(
            SimpleNamespace(
                prompt_token_count=3, candidates_token_count=1, total_token_count=4
            )
        )
        with (
            patch.object(dispatch, "_get_gemini_client", return_value=client),
            patch("app.core.llm_usage.emitter.emit_llm_usage_blocking") as emit,
        ):
            await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="gemini",
                task="language_judge",
                scenario_session_id="sess-123",
            )

        assert emit.call_args.kwargs["scenario_session_id"] == "sess-123"

    @pytest.mark.asyncio
    async def test_reaches_the_emitter_on_anthropic(self, both_keys):
        client = SimpleNamespace(
            messages=SimpleNamespace(
                create=AsyncMock(return_value=anthropic_response(tool_input={}))
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
                task="language_judge",
                scenario_session_id="sess-123",
            )

        assert emit.call_args.kwargs["scenario_session_id"] == "sess-123"
        # Anthropic semantics are untouched by this change.
        assert emit.call_args.kwargs["cached_tokens"] is None
        assert emit.call_args.kwargs["metadata"] is None

    @pytest.mark.asyncio
    async def test_survives_a_fallback_retry(self):
        """The retried call is still the same session's cost."""
        failing = AsyncMock(
            side_effect=type("APIStatusError", (Exception,), {"status_code": 503})(
                "boom"
            )
        )
        client = _openai_client_with_usage(
            SimpleNamespace(prompt_tokens=8, completion_tokens=5, total_tokens=13)
        )
        with (
            patch.object(dispatch.settings.GEMINI, "API_KEY", "g"),
            patch.object(dispatch.settings.OPENAI, "API_KEY", "o"),
            patch.dict(dispatch._GENERATORS, {"gemini": failing}),
            patch.object(dispatch, "_get_openai_client", return_value=client),
            patch("app.core.llm_usage.emitter.emit_llm_usage_blocking") as emit,
        ):
            await dispatch.generate_structured(
                schema=Answer,
                prompt="q",
                provider="gemini",
                task="language_judge",
                scenario_session_id="sess-123",
            )

        assert failing.await_args.kwargs["scenario_session_id"] == "sess-123"
        assert emit.call_args.kwargs["scenario_session_id"] == "sess-123"

    @pytest.mark.asyncio
    async def test_defaults_to_none(self, both_keys):
        """Every existing caller (the WhatsApp bot, the analytics agent) omits
        it and emits exactly as before."""
        client = _gemini_client_with_usage(
            SimpleNamespace(
                prompt_token_count=3, candidates_token_count=1, total_token_count=4
            )
        )
        with (
            patch.object(dispatch, "_get_gemini_client", return_value=client),
            patch("app.core.llm_usage.emitter.emit_llm_usage_blocking") as emit,
        ):
            await dispatch.generate_structured(
                schema=Answer, prompt="q", provider="gemini", task="whatsapp_rag_answer"
            )

        assert emit.call_args.kwargs["scenario_session_id"] is None
