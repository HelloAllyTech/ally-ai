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

# gpt-4o-mini skipped the direct safety question on "everyone would be better off
# without me" in live checks even with the rule above it, so when ally-be's own screen
# has already flagged the chat the prompt says so outright.
_RISK_NOT_FLAGGED = (
    "The platform's risk screen has not flagged this conversation. Still apply the "
    "safety rule above if you notice risk yourself."
)
_RISK_FLAGGED = (
    "The platform's risk screen HAS flagged this conversation ({level}{about}). Your "
    "FIRST suggestion must be tagged harm and ask directly about safety as described "
    "above. If the listener has already asked and the talker has answered in the last "
    "two messages, ask the next safety question instead (a plan, the means, when, "
    "whether they have tried before, who could be with them right now). If someone "
    "is harming them, ask whether they will be safe and what they could do if things "
    "get worse."
)


def talker_script_language(
    language: Optional[str], messages: Sequence[Dict[str, str]]
) -> str:
    """The language name the turn prompt asks for, pinned to the talker's script.

    Telling the model "follow the talker's script" was not enough: in live checks
    gpt-4.1-mini answered romanised Hindi in Devanagari. So the script is measured
    here from the talker's own recent messages and named outright.
    """
    name = language_name(language)
    code = (language or "en").strip().lower().split("-")[0]
    if code == "en":
        return _english_chat_language(messages)
    latin = native = 0
    for turn in list(messages)[-6:]:
        if turn.get("role") != "talker":
            continue
        for ch in turn.get("content") or "":
            if not ch.isalpha():
                continue
            if "a" <= ch.lower() <= "z":
                latin += 1
            else:
                native += 1
    if latin and latin > native:
        return (
            f"{name}, written in Latin letters (romanised {name}) exactly as the "
            "talker writes it, not in the native script"
        )
    return f"{name}, in its native script"


# Common romanised-Hindi words. A talker who picked English on the consent screen often
# types Hinglish anyway ("haan, bas thak gayi hoon"); live testing on 2026-10-06 got an
# English-only draft back for exactly that. Two distinct hits in the talker's recent
# messages is the signal — one could be a name or a loan word.
_HINGLISH_MARKERS = frozenset(
    "hai hain hoon hu nahi nahin mein mujhe mujhko kya kyun kyon bahut bohot "
    "haan nahi bas sab kuch koi raha rahi rahe tha thi kar karna kiya gaya gayi "
    "aap tum main mera meri mere apna apni yaar ghar pe par lekin aur bhi".split()
)
_SCRIPT_RANGES = (
    ("\u0900", "\u097f", "Hindi or Marathi, in Devanagari"),
    ("\u0b80", "\u0bff", "Tamil, in Tamil script"),
    ("\u0c80", "\u0cff", "Kannada, in Kannada script"),
)


def _english_chat_language(messages: Sequence[Dict[str, str]]) -> str:
    """For an English chat: English, unless the talker clearly writes otherwise."""
    script_counts = [0] * len(_SCRIPT_RANGES)
    markers: set[str] = set()
    for turn in list(messages)[-6:]:
        if turn.get("role") != "talker":
            continue
        content = turn.get("content") or ""
        for ch in content:
            for index, (low, high, _) in enumerate(_SCRIPT_RANGES):
                if low <= ch <= high:
                    script_counts[index] += 1
        for word in content.lower().replace(",", " ").replace(".", " ").split():
            if word.strip("!?'\"") in _HINGLISH_MARKERS:
                markers.add(word.strip("!?'\""))
    best = max(range(len(script_counts)), key=lambda i: script_counts[i])
    if script_counts[best] >= 4:
        return (
            f"{_SCRIPT_RANGES[best][2]} — the same language and script the talker is "
            "writing in (the chat was opened in English)"
        )
    if len(markers) >= 2:
        return (
            "Hinglish — romanised Hindi mixed with English, in Latin letters, the way "
            "the talker is writing (the chat was opened in English)"
        )
    return "English"


def risk_instruction(risk_level: Optional[str], risk_subject: Optional[str]) -> str:
    """The sentence that tells the turn prompt whether ally-be flagged risk."""
    level = (risk_level or "NONE").upper()
    if level not in ("ELEVATED", "HIGH"):
        return _RISK_NOT_FLAGGED
    about = ", about someone else the talker knows" if risk_subject == "OTHER" else ""
    return _RISK_FLAGGED.format(level=level.lower(), about=about)


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
    risk_level: Optional[str] = "NONE",
    risk_subject: Optional[str] = "",
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
        talker_language=talker_script_language(language, messages),
        nudge_instruction=_NUDGE_REQUESTED if include_nudge else _NUDGE_NOT_REQUESTED,
        risk_instruction=risk_instruction(risk_level, risk_subject),
    )
