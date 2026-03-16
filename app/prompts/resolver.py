"""
Resolve prompt templates from backend metadata or local .txt files.
"""

from typing import Any, Dict, Optional

from app.utils.logger import get_logger
from app.prompts.manager import prompt_manager

logger = get_logger(__name__)

PROMPT_CODE_PREFIX = "ally_ai_"


def _prompt_code_to_path(prompt_code: str) -> tuple[str, str]:
    """
    Parse prompt_code into (file_key, prompt_key) for local lookup.
    E.g. ally_ai_nudge_nudge -> (nudge, nudge)
    """
    code = prompt_code
    if code.startswith(PROMPT_CODE_PREFIX):
        code = code[len(PROMPT_CODE_PREFIX) :]
    if "_" not in code:
        logger.warning(f"Invalid prompt code (no subdir): {prompt_code}")
        return ("", "")
    file_key, prompt_key = code.split("_", 1)
    return (file_key, prompt_key)


def _get_local_template(prompt_code: str) -> str:
    """Get template from local prompts folder using path-based prompt_code."""
    file_key, prompt_key = _prompt_code_to_path(prompt_code)
    if not file_key or not prompt_key:
        return ""
    return prompt_manager.get_template(file_key, prompt_key) or ""


def resolve_template(
    prompt_code: str,
    backend_prompts: Optional[Dict[str, str]] = None,
) -> str:
    """
    Resolve a prompt template.
    Use metadata prompt when present and non-empty.
    Otherwise use local .txt.
    """
    backend_template = backend_prompts.get(prompt_code) if backend_prompts else None
    if backend_template and backend_template.strip():
        return backend_template.strip()

    local_template = _get_local_template(prompt_code)
    return local_template


def load_template(
    internal_path: str,
    prompt_data: Any = None,
    prompts: Optional[Dict[str, str]] = None,
) -> str:
    """
    Resolve template with fallback to local.
    """
    prompt_code = f"ally_ai_{internal_path.replace('/', '_')}"

    # Use 'prompts' if passed directly as keyword, otherwise look into prompt_data
    backend_prompts = prompts
    if backend_prompts is None:
        if isinstance(prompt_data, dict):
            backend_prompts = prompt_data
        else:
            backend_prompts = getattr(prompt_data, "prompts", None) if prompt_data else None

    return resolve_template(prompt_code, backend_prompts) or ""


def load_and_format(
    internal_path: str,
    prompt_data: Any = None,
    prompts: Optional[Dict[str, str]] = None,
    **kwargs,
) -> str:
    """
    Load template and optionally format with Python str.format().
    """
    template = load_template(internal_path, prompt_data=prompt_data, prompts=prompts)
    if not template:
        return ""

    if kwargs:
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing format key {e} for prompt {internal_path}")
            return template

    return template
