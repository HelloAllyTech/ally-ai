"""Prompt assembly for the knowledge agent.

Templates live in app/prompts/knowledge/*.txt and are resolved through the shared
resolver, so ally-be's prompt management can override the text, model and temperature at
runtime. Everything here is the part that must NOT be overridable: how retrieved
passages are numbered and laid out, because the citation contract depends on those
numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.prompts.resolver import load_and_format

ANSWER_PROMPT_PATH = "knowledge/whatsapp_answer"
TRANSLATE_PROMPT_PATH = "knowledge/translate_query"
CRISIS_PROMPT_PATH = "knowledge/crisis_classify"

# Cap per passage when rendering. A single pathological chunk cannot be allowed to crowd
# the others out of the prompt; ally-be's chunker targets ~400 tokens, so this only ever
# bites on malformed input.
MAX_PASSAGE_CHARS = 4000

# Cap on rendered history. Two or three turns of context is enough to resolve "what
# about for children?"; more just costs tokens on every question.
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 1500


def format_passages(passages: List[Dict[str, Any]]) -> str:
    """
    Render retrieved passages as a numbered list.

    The numbers ARE the citation contract: the model returns integers, and those
    integers are mapped back to chunk ids here rather than trusted from the model. So
    numbering starts at 1, is contiguous, and matches the order of `passages` exactly —
    the caller resolves `citations` positionally against the same list it passed in.

    Source metadata is included per passage because the model needs to be able to tell
    two passages apart when deciding which it used, but it is told not to reproduce it:
    the source lines are rendered by ally-be from the resolved citations, not copied out
    of the answer text.
    """
    if not passages:
        return "(no passages were retrieved)"

    blocks: List[str] = []
    for index, passage in enumerate(passages, start=1):
        text = (passage.get("text") or "").strip()
        if len(text) > MAX_PASSAGE_CHARS:
            text = text[:MAX_PASSAGE_CHARS].rstrip() + "…"

        source_bits: List[str] = []
        title = (passage.get("document_title") or "").strip()
        if title:
            source_bits.append(title)
        page_from = int(passage.get("page_from") or 0)
        page_to = int(passage.get("page_to") or 0)
        if page_from:
            if page_to and page_to != page_from:
                source_bits.append(f"pp. {page_from}-{page_to}")
            else:
                source_bits.append(f"p. {page_from}")
        section = (passage.get("section_path") or "").strip()
        if section:
            source_bits.append(section)
        source = " · ".join(source_bits) if source_bits else "untitled source"

        blocks.append(f"[{index}] {source}\n{text}")

    return "\n\n".join(blocks)


def format_history(history: Optional[List[Dict[str, str]]]) -> str:
    """
    Render recent turns as plain labelled lines.

    Trimmed to the most recent turns rather than the earliest: a follow-up depends on
    what was just said, and an old turn that no longer applies is worse than no context,
    because the model may answer the previous question again.
    """
    if not history:
        return "(no previous messages)"

    recent = history[-MAX_HISTORY_TURNS:]
    lines: List[str] = []
    for turn in recent:
        role = (turn.get("role") or "").strip().lower()
        content = " ".join((turn.get("content") or "").split())
        if not content:
            continue
        speaker = "Worker" if role in ("user", "worker", "human") else "Assistant"
        lines.append(f"{speaker}: {content}")

    rendered = "\n".join(lines)
    if len(rendered) > MAX_HISTORY_CHARS:
        # Keep the TAIL when trimming — the newest turn is the one the question refers
        # to.
        rendered = "…\n" + rendered[-MAX_HISTORY_CHARS:]
    return rendered or "(no previous messages)"


def build_answer_prompt(
    question: str,
    *,
    passages: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
    max_chars: int = 1400,
    prompts: Optional[Dict[str, Any]] = None,
) -> str:
    """Assemble the answer prompt, honouring any ally-be override of the template."""
    return load_and_format(
        ANSWER_PROMPT_PATH,
        prompts=prompts,
        question=question.strip(),
        passages=format_passages(passages),
        history=format_history(history),
        max_chars=max_chars,
    )


def build_translate_prompt(
    question: str, *, prompts: Optional[Dict[str, Any]] = None
) -> str:
    """Assemble the query-preparation prompt."""
    return load_and_format(
        TRANSLATE_PROMPT_PATH,
        prompts=prompts,
        question=question.strip(),
    )


def build_crisis_prompt(
    message: str, *, prompts: Optional[Dict[str, Any]] = None
) -> str:
    """Assemble the crisis-classification prompt."""
    return load_and_format(
        CRISIS_PROMPT_PATH,
        prompts=prompts,
        message=message.strip(),
    )
