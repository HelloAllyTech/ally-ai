"""Prompt assembly for the text-helpline copilot.

Templates live in app/prompts/helpline/*.txt and are resolved through the shared
resolver, so ally-be's prompt management can override the text, model and temperature at
runtime. Everything here is the part that must NOT be overridable: how the conversation
is laid out for the model, which is what the service's post-processing (the verbatim
signal check, the closed vocabularies) relies on.

The talker is anonymous and unvetted, so everything they type is untrusted text that
ends up inside a prompt. Two defences live here rather than in the template: each prior
turn is collapsed onto ONE line (so a message cannot forge a "Listener:" line of its
own), and the latest message sits between explicit markers the template tells the model
to treat as data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.prompts.resolver import load_and_format

RISK_PROMPT_PATH = "helpline/risk_classify"
TURN_PROMPT_PATH = "helpline/copilot_turn"

# How much history each call sees. The contract caps what ally-be sends (4 for the risk
# check, 12 for the turn); these are the same numbers applied again here so an over-long
# list cannot inflate the prompt.
MAX_RISK_CONTEXT_TURNS = 4
MAX_TURN_MESSAGES = 12

# Per-turn character caps. ally-be limits a message to 2,000 characters; a turn in the
# context is trimmed harder because it only has to explain the latest one.
MAX_CONTEXT_TURN_CHARS = 600
MAX_CONVERSATION_TURN_CHARS = 1200
MAX_LATEST_MESSAGE_CHARS = 4000
MAX_SUMMARY_CHARS = 2000

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "kn": "Kannada",
}

_NUDGE_REQUESTED = (
    "Write one `nudge`: a single concrete coaching tip for the LISTENER about their "
    "next move, at most 240 characters, in English. It is advice to the listener, NOT "
    "something to send to the talker. Say what you noticed, then the tip — for example "
    '"They have said they feel alone twice; try reflecting that before asking about '
    'the exam." Keep it specific and kind, never a criticism. Never suggest promising '
    "confidentiality, diagnosing, or discussing medication. If there is nothing useful "
    "to say, leave `nudge` empty."
)
_NUDGE_NOT_REQUESTED = "Leave `nudge` as an empty string."


def language_name(language: Optional[str]) -> str:
    """A human name for a language code, tolerating region tags ('hi-IN') and case."""
    code = (language or "en").strip().lower().replace("_", "-").split("-")[0] or "en"
    return LANGUAGE_NAMES.get(code, code)


def _speaker(role: str) -> str:
    return "Talker" if (role or "").strip().lower() == "talker" else "Listener"


def _one_line(text: str, limit: int) -> str:
    """Collapse whitespace to single spaces and trim to `limit`, keeping the head."""
    collapsed = " ".join((text or "").split())
    if len(collapsed) > limit:
        collapsed = collapsed[:limit].rstrip() + "…"
    return collapsed


def format_conversation(
    messages: Optional[Sequence[Dict[str, str]]],
    *,
    max_turns: int,
    turn_chars: int,
    empty: str,
) -> str:
    """Render turns oldest-first as 'Talker:' / 'Listener:' lines.

    Trimmed to the most recent turns rather than the earliest: what was just said is
    what the latest message refers to.
    """
    if not messages:
        return empty

    lines: List[str] = []
    for turn in list(messages)[-max_turns:]:
        content = _one_line(turn.get("content", ""), turn_chars)
        if content:
            lines.append(f"{_speaker(turn.get('role', ''))}: {content}")
    return "\n".join(lines) or empty


def build_risk_prompt(
    message: str,
    *,
    recent: Optional[Sequence[Dict[str, str]]] = None,
    language: Optional[str] = "en",
    prompts: Optional[Dict[str, Any]] = None,
) -> str:
    """Assemble the risk-classifier prompt, honouring any ally-be override."""
    return load_and_format(
        RISK_PROMPT_PATH,
        prompts=prompts,
        message=(message or "").strip()[:MAX_LATEST_MESSAGE_CHARS],
        recent=format_conversation(
            recent,
            max_turns=MAX_RISK_CONTEXT_TURNS,
            turn_chars=MAX_CONTEXT_TURN_CHARS,
            empty="(no earlier messages)",
        ),
        language=(language or "en"),
        language_name=language_name(language),
    )


def build_turn_prompt(
    messages: Sequence[Dict[str, str]],
    *,
    rolling_summary: str = "",
    language: Optional[str] = "en",
    include_nudge: bool = False,
    prompts: Optional[Dict[str, Any]] = None,
) -> str:
    """Assemble the copilot-turn prompt, honouring any ally-be override."""
    summary = _one_line(rolling_summary, MAX_SUMMARY_CHARS)
    return load_and_format(
        TURN_PROMPT_PATH,
        prompts=prompts,
        conversation=format_conversation(
            messages,
            max_turns=MAX_TURN_MESSAGES,
            turn_chars=MAX_CONVERSATION_TURN_CHARS,
            empty="(no messages yet)",
        ),
        rolling_summary=summary or "(none yet)",
        language=(language or "en"),
        language_name=language_name(language),
        nudge_instruction=_NUDGE_REQUESTED if include_nudge else _NUDGE_NOT_REQUESTED,
    )
