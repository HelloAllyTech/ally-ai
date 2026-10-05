"""The text helpline's copilot: a risk classifier and a suggestion/nudge turn.

Stateless, like the knowledge agent: the conversation arrives on the request and this
service owns no database. ally-be owns every row, every timeout and every decision about
what to do with a result — it enforces 3 s on the risk call and 6 s on the turn, and
persists, flags and emits. Nothing returned from here is ever sent to a talker; the
suggestions are drafts a human listener edits.

Both calls fail OPEN into a report rather than raising: `failed: true` with empty
content. Raising would push ally-be onto its generic error path, and defaulting a failed
risk check to "crisis" would flag every chat the moment an API key expired, which is how
a listener learns to ignore flags. ally-be's keyword screen is what still holds while
the classifier is down, and it records the failure.

Message bodies are PHI. Nothing here puts a talker's words in the application log: the
verbatim `signal` goes to `phi_logger` only, and every other log line carries counts,
confidences and exception CLASS names, never exception text (a validation error quotes
its input).
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import settings
from app.core.helpline import safety
from app.core.helpline.prompt import (
    RISK_PROMPT_PATH,
    TURN_PROMPT_PATH,
    build_risk_prompt,
    build_turn_prompt,
)
from app.core.helpline.schemas import (
    SKILL_KEYS,
    STAGES,
    SUBJECTS,
    HelplineCopilotTurn,
    HelplineRiskVerdict,
)
from app.core.llm.dispatch import generate_structured
from app.core.llm_usage.tasks import LLMTask
from app.core.phi_events import PHIEvents
from app.core.phi_logger import PHILogEvent, phi_logger
from app.prompts.resolver import get_backend_llm_overrides
from app.utils.logger import get_logger

logger = get_logger(__name__)

#: Limits from the contract (ally-be docs/text-helpline.md §9.1). Applied here as well
#: as asked of the model, because a model asked for 300 characters writes 340 often
#: enough that the UI must not be the thing that finds out.
MAX_SIGNAL_CHARS = 120
MAX_SUGGESTION_CHARS = 300
MAX_NUDGE_CHARS = 240
MAX_SUGGESTIONS = 3

#: A salvaged partial signal must be at least this long, and at least this share of what
#: the model wrote, to be worth showing a listener as "why this was flagged". Below
#: that it is a stray fragment ("die") that reads as a different message than the one
#: that was sent, and blank is more honest.
_MIN_SALVAGED_SIGNAL_CHARS = 4
_MIN_SALVAGED_SIGNAL_SHARE = 0.4

#: Default sampling for the risk classifier. NOT 0.0: `generate_structured`'s OpenAI
#: path sends temperature only when it is truthy, so 0.0 is silently dropped and the
#: call runs at the model's default (1.0 for gpt-4o-mini) — the opposite of the
#: determinism a classifier wants. A small non-zero value survives that check.
_RISK_TEMPERATURE = 0.01

#: Default sampling for suggestions. Some variety is the point: the listener is shown
#: two or three genuinely different ways to respond.
_TURN_TEMPERATURE = 0.6

_SENTENCE_END = ".?!।"


def _meta_str(meta: Any, key: str) -> str:
    return str(meta.get(key, "") or "") if isinstance(meta, dict) else ""


def _clamp_confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalise_subject(value: Any) -> str:
    candidate = str(value or "").strip().upper()
    return candidate if candidate in SUBJECTS else "UNCLEAR"


def resolve_signal(signal: str, message: str) -> str:
    """Make `signal` a verbatim substring of `message` (at most 120 chars), or blank.

    ally-be locates the signal in the stored message and keeps only its OFFSETS, so a
    signal that is not literally in the message cannot be located and a listener would
    be shown words the talker never wrote. The model is told to copy exactly, and
    usually does; this is for when it does not. In order of preference:

      1. the signal as given;
      2. the same words with different case or whitespace (a model reading a message
         with a line break often returns it on one line) — returned as the MESSAGE's own
         text, not the model's;
      3. the longest stretch the two share, when that is long enough to still mean
         something;
      4. blank.
    """
    signal = (signal or "").strip()
    if not signal or not message:
        return ""

    if signal in message:
        return signal[:MAX_SIGNAL_CHARS]

    words = signal.split()
    if words:
        flexible = re.compile(r"\s+".join(re.escape(w) for w in words), re.IGNORECASE)
        match = flexible.search(message)
        if match:
            return match.group(0)[:MAX_SIGNAL_CHARS]

    matcher = difflib.SequenceMatcher(None, message, signal, autojunk=False)
    block = matcher.find_longest_match(0, len(message), 0, len(signal))
    fragment = message[block.a : block.a + block.size].strip()
    if len(fragment) >= _MIN_SALVAGED_SIGNAL_CHARS and len(fragment) >= (
        _MIN_SALVAGED_SIGNAL_SHARE * len(signal)
    ):
        return fragment[:MAX_SIGNAL_CHARS]
    return ""


def truncate(text: str, limit: int) -> str:
    """Trim to at most `limit` characters, ending on a sentence or word boundary.

    Cuts at the last sentence end in the kept text when there is one past the halfway
    mark, otherwise at the last space with an ellipsis — never mid-word, so a clipped
    suggestion still reads as a sentence the listener can finish.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text

    head = text[:limit]
    cut = max(head.rfind(mark) for mark in _SENTENCE_END)
    if cut >= limit // 2:
        return head[: cut + 1].rstrip()

    room = text[: limit - 1]
    space = room.rfind(" ")
    if space >= limit // 2:
        room = room[:space]
    return room.rstrip(" ,;:-") + "…"


def _canonical_skill(value: Any) -> str:
    candidate = str(value or "").strip().lower().replace(" ", "").replace("_", "")
    for key in SKILL_KEYS:
        if candidate == key:
            return key
    return ""


def _canonical_stage(value: Any) -> str:
    candidate = str(value or "").strip().lower()
    for stage in STAGES:
        if candidate == stage.lower():
            return stage
    return ""


class HelplineCopilotService:
    """Classifies talker messages for risk and drafts replies for the listener."""

    # ------------------------------------------------------------------ risk

    async def classify_risk(
        self,
        message: str,
        *,
        recent: Optional[Sequence[Dict[str, str]]] = None,
        language: str = "en",
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Decide whether a talker's latest message signals risk.

        The verdict comes back untouched except for tidying: no confidence threshold is
        applied here, because the prompt tells the model to choose `true` when unsure
        and re-gating on confidence would silently undo that. Turning (is_crisis,
        confidence) into HIGH or ELEVATED is ally-be's job — it holds the per-org
        threshold.

        A failure returns ``is_crisis: false`` with ``failed: true``. See the module
        docstring for why it is not the other way round.
        """
        failure: Dict[str, Any] = {
            "is_crisis": False,
            "confidence": 0.0,
            "signal": "",
            "subject": "UNCLEAR",
            "failed": True,
            "provider": "",
            "model": "",
        }

        text = message or ""
        if not text.strip():
            # Nothing to assess is a successful, empty result — not a failed
            # classifier, which would be logged as a degraded safety net.
            return {**failure, "failed": False}

        cfg = settings.HELPLINE
        provider, model, temperature = get_backend_llm_overrides(
            RISK_PROMPT_PATH, prompts
        )
        prompt = build_risk_prompt(
            text, recent=recent, language=language, prompts=prompts
        )
        if not prompt:
            logger.error("Helpline risk prompt resolved empty; skipping classification")
            return failure

        try:
            parsed, meta = await generate_structured(
                schema=HelplineRiskVerdict,
                prompt=prompt,
                task=LLMTask.HELPLINE_RISK_CLASSIFY.value,
                provider=provider,
                model=model or cfg.RISK_MODEL,
                # An override of exactly 0.0 is floored for the same reason as the
                # default (see _RISK_TEMPERATURE): it would otherwise be dropped.
                temperature=(
                    _RISK_TEMPERATURE
                    if temperature is None
                    else max(temperature, _RISK_TEMPERATURE)
                ),
                max_tokens=cfg.RISK_MAX_TOKENS,
            )
        # noqa: BLE001 below — a classifier failure must never raise to the caller.
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Helpline risk classification failed (%s); keyword screen holds",
                type(e).__name__,
            )
            return failure

        is_crisis = bool(parsed.is_crisis)
        confidence = _clamp_confidence(parsed.confidence)
        subject = _normalise_subject(parsed.subject) if is_crisis else "UNCLEAR"
        signal = resolve_signal(parsed.signal, text) if is_crisis else ""

        if is_crisis:
            # The confidence and subject are not content; the signal is the talker's
            # own words, so it goes to the PHI audit log and nowhere else.
            logger.info(
                "Helpline risk classifier fired: confidence=%.2f subject=%s",
                confidence,
                subject,
            )
            await phi_logger.log(
                PHILogEvent(
                    event_type=PHIEvents.DATA_ACCESSED,
                    chat_id=None,
                    audit_id=None,
                    details={
                        "message": "Helpline risk classifier fired",
                        "component": "HelplineCopilotService.classify_risk",
                        "confidence": confidence,
                        "subject": subject,
                        "signal": signal[:MAX_SIGNAL_CHARS],
                    },
                )
            )

        return {
            "is_crisis": is_crisis,
            "confidence": confidence,
            "signal": signal,
            "subject": subject,
            "failed": False,
            "provider": _meta_str(meta, "provider"),
            "model": _meta_str(meta, "model"),
        }

    # ------------------------------------------------------------------ turn

    async def copilot_turn(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        rolling_summary: str = "",
        language: str = "en",
        include_nudge: bool = False,
        risk_level: str = "NONE",
        risk_subject: str = "",
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Draft suggestions (and optionally a nudge) for the listener's next move.

        Generated text is never trusted on the way out. Each suggestion is checked
        against the safety filter and dropped, not repaired, if it fails; what is left
        is capped and trimmed. If nothing is left the list is empty — a copilot with
        nothing safe to say says nothing, and ally-be shows no suggestion card.
        """
        failure: Dict[str, Any] = {
            "stage": "",
            "nudge": "",
            "suggestions": [],
            "failed": True,
            "provider": "",
            "model": "",
        }

        cfg = settings.HELPLINE
        provider, model, temperature = get_backend_llm_overrides(
            TURN_PROMPT_PATH, prompts
        )
        prompt = build_turn_prompt(
            messages,
            rolling_summary=rolling_summary,
            language=language,
            include_nudge=include_nudge,
            risk_level=risk_level,
            risk_subject=risk_subject,
            prompts=prompts,
        )
        if not prompt:
            logger.error("Helpline turn prompt resolved empty; skipping the turn")
            return failure

        try:
            parsed, meta = await generate_structured(
                schema=HelplineCopilotTurn,
                prompt=prompt,
                task=LLMTask.HELPLINE_COPILOT_TURN.value,
                provider=provider,
                model=model or cfg.TURN_MODEL,
                temperature=(_TURN_TEMPERATURE if temperature is None else temperature),
                max_tokens=cfg.TURN_MAX_TOKENS,
            )
        except Exception as e:  # noqa: BLE001 — a copilot failure never raises
            logger.warning("Helpline copilot turn failed (%s)", type(e).__name__)
            return failure

        suggestions = self._clean_suggestions(parsed)
        nudge = self._clean_nudge(parsed.nudge) if include_nudge else ""

        return {
            "stage": _canonical_stage(parsed.stage),
            "nudge": nudge,
            "suggestions": suggestions,
            "failed": False,
            "provider": _meta_str(meta, "provider"),
            "model": _meta_str(meta, "model"),
        }

    # ------------------------------------------------------------------ cleaning

    @staticmethod
    def _clean_suggestions(parsed: HelplineCopilotTurn) -> List[Dict[str, str]]:
        """Normalise, safety-filter, de-duplicate, cap and trim the model's drafts."""
        candidates: List[Dict[str, str]] = []
        malformed = 0
        for item in parsed.suggestions or []:
            text = (item.text or "").strip()
            skill_key = _canonical_skill(item.skill_key)
            # No text, or a tag outside the contract's set, is not a suggestion ally-be
            # can store. Dropped rather than patched: a guessed tag would put a wrong
            # label in front of the listener.
            if not text or not skill_key:
                malformed += 1
                continue
            candidates.append({"text": text, "skill_key": skill_key})

        kept, dropped = safety.filter_suggestions(candidates)

        seen: set[str] = set()
        unique: List[Dict[str, str]] = []
        for suggestion in kept:
            fingerprint = " ".join(suggestion["text"].lower().split())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(suggestion)

        if dropped or malformed:
            # Counts and reason codes only — never the text.
            logger.info(
                "Helpline suggestions dropped: safety=%s malformed=%d kept=%d",
                dropped,
                malformed,
                len(unique),
            )

        return [
            {
                "text": truncate(s["text"], MAX_SUGGESTION_CHARS),
                "skill_key": s["skill_key"],
            }
            for s in unique[:MAX_SUGGESTIONS]
        ]

    @staticmethod
    def _clean_nudge(nudge: str) -> str:
        """The nudge is coaching for the listener, so most of the suggestion filter does
        not apply (it may rightly say "don't promise confidentiality"). A dosage or a
        drug name still never belongs in one."""
        nudge = (nudge or "").strip()
        if not nudge or safety.mentions_dosage_or_drug(nudge):
            return ""
        return truncate(nudge, MAX_NUDGE_CHARS)
