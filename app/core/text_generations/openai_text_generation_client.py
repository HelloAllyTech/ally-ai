from typing import Optional

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global OpenAI chat client
_openai_chat_client = None

# Cache of per-override chat clients (OpenAI or Gemini) so we don't rebuild one
# on every call. Keyed by (provider, model, temperature).
_override_client_cache: dict = {}


def _is_openai_model(model: Optional[str]) -> bool:
    """Recognize OpenAI model names so the provider can be inferred from the
    model when an explicit provider isn't supplied (legacy fallback)."""
    if not model:
        return False
    name = model.strip().lower()
    return name.startswith(("gpt", "o1", "o3", "o4", "chatgpt", "text-", "davinci"))


def _is_gemini_model(model: Optional[str]) -> bool:
    """Recognize Gemini model names for provider inference."""
    return bool(model) and model.strip().lower().startswith("gemini")


def _resolve_provider(model: Optional[str], provider: Optional[str]) -> Optional[str]:
    """Resolve the runnable provider for a prompt override. Prefers the explicit
    `provider` (only 'openai'/'gemini' run here); else infers from the model
    name. Returns None when neither resolves to a provider ally-ai can run."""
    if provider:
        p = provider.strip().lower()
        return p if p in ("openai", "gemini") else None
    if _is_openai_model(model):
        return "openai"
    if _is_gemini_model(model):
        return "gemini"
    return None


def _build_gemini_client(model: str, temperature: Optional[float]):
    """Build a langchain Gemini chat client for a prompt override. Returns None
    when the key or the optional dependency is unavailable so the caller can
    degrade to the default OpenAI client instead of breaking the call."""
    if not settings.GEMINI.API_KEY:
        logger.warning(
            "GEMINI__API_KEY not configured; ignoring Gemini override %r.", model
        )
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception:  # pragma: no cover - optional dependency
        logger.warning(
            "langchain-google-genai not installed; ignoring Gemini override %r.",
            model,
        )
        return None
    kwargs = {"model": model, "google_api_key": settings.GEMINI.API_KEY}
    if temperature is not None:
        kwargs["temperature"] = temperature
    logger.info(
        "Building override Gemini client (model=%s, temperature=%s)",
        model,
        temperature,
    )
    return ChatGoogleGenerativeAI(**kwargs)


def _model_supports_temperature(model: Optional[str]) -> bool:
    """Whether a model accepts a custom temperature. OpenAI reasoning models
    (o-series, gpt-5 family) only allow the default and 400 on others."""
    if not model:
        return True
    name = model.strip().lower()
    return not name.startswith(("o1", "o3", "o4", "gpt-5"))


class OpenAITextGenerationClient:
    @staticmethod
    def get_client() -> ChatOpenAI:
        """
        Get the singleton instance of the OpenAI chat client.

        Returns:
            ChatOpenAI: The OpenAI chat client.
        """
        if not _openai_chat_client:
            logger.error(
                "OpenAI chat client has not been created. Please create a client first."
            )
            raise Exception(
                "OpenAI chat client has not been created. Please create a client first."
            )

        return _openai_chat_client

    @staticmethod
    def create_client(model_name: str) -> None:
        """
        Create a singleton instance of the OpenAI chat client.

        Parameters:
            model_name (str): The name of the model to use.
        """
        global _openai_chat_client

        if not _openai_chat_client:
            logger.info(f"Creating a new OpenAI chat client with model {model_name}...")
            _openai_chat_client = ChatOpenAI(
                model=model_name,
                api_key=settings.OPENAI.API_KEY,
                organization=settings.OPENAI.ORGANIZATION_ID,
            )
        else:
            logger.warning(
                "OpenAI chat client already exists. Reusing the existing client."
            )

    @staticmethod
    def get_or_create_client(
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        provider: Optional[str] = None,
    ):
        """Return a chat client for a prompt-level (model, temperature) override,
        cached by (provider, model, temperature). Falls back to the process
        default OpenAI client when no override applies.

        Runs OpenAI + Gemini. `provider` (explicit, from the prompt) decides the
        provider when set; otherwise it's inferred from the model name (legacy
        fallback). A model on a provider we can't run — or a Gemini pick when the
        key/dependency is missing — is ignored, but its temperature still applies
        to the default OpenAI model.
        """
        resolved = _resolve_provider(model, provider)

        # Gemini override: build/cache a langchain Gemini client. If the key or
        # dependency is missing, degrade to the default OpenAI model below.
        if model and resolved == "gemini":
            key = ("gemini", model, temperature)
            client = _override_client_cache.get(key)
            if client is None:
                client = _build_gemini_client(model, temperature)
                if client is not None:
                    _override_client_cache[key] = client
            if client is not None:
                return client
            model = None  # Gemini unavailable; fall through to default OpenAI.

        if model and resolved != "openai":
            logger.warning(
                "Ignoring non-OpenAI/Gemini prompt model override %r for ally-ai "
                "text-gen; applying temperature=%s only.",
                model,
                temperature,
            )
            model = None

        effective_model = model or OpenAITextGenerationClient.get_client().model_name

        # Reasoning models (o-series, gpt-5) reject a custom temperature; drop it
        # so an override on such a model degrades to its default, never 400s.
        if temperature is not None and not _model_supports_temperature(effective_model):
            logger.info(
                "Model %s does not support a custom temperature; omitting it.",
                effective_model,
            )
            temperature = None

        if not model and temperature is None:
            return OpenAITextGenerationClient.get_client()

        key = ("openai", effective_model, temperature)
        client = _override_client_cache.get(key)
        if client is None:
            kwargs = {
                "model": effective_model,
                "api_key": settings.OPENAI.API_KEY,
                "organization": settings.OPENAI.ORGANIZATION_ID,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            logger.info(
                "Building override OpenAI client (model=%s, temperature=%s)",
                effective_model,
                temperature,
            )
            client = ChatOpenAI(**kwargs)
            _override_client_cache[key] = client
        return client
