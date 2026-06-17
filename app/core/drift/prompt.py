"""Judge prompt builder for the conversation drift judge.

This is the v1 rubric — a STARTING POINT to calibrate against a hand-labeled
seed set (with a native-speaker check for languages we can't read), not a frozen
artifact. Bump ``DRIFT_JUDGE.PROMPT_VERSION`` whenever this changes.

Few-shot examples are intentionally left as a TODO: they must be harvested from
real sessions per language (a garbled-STT negative, an in-character-distress
negative, and a turn-by-turn-discrimination example) once the seed set exists.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


# Prompt-management code for the judge rubric (seeded by the
# AddDriftJudgePrompt migration). The judge fetches the current version from
# ally-be and falls back to DEFAULT_JUDGE_RUBRIC below if the fetch fails.
DRIFT_JUDGE_PROMPT_CODE = "drift_judge_conversation_rubric"


class TranscriptTurn(TypedDict, total=False):
    role: str  # "client" (AI) | "counselor" (human)
    text: str
    turn_index: int  # required for AI-client turns


# Fallback rubric — keep in sync with the seeded prompt-management version
# (drift_judge_conversation_rubric); the registry version is authoritative.
DEFAULT_JUDGE_RUBRIC = """\
You evaluate a single role-play counseling-training session for "conversation \
drift" and label each AI turn.

ROLES (do not get these backwards):
- The AI plays the CLIENT (the person seeking help). You judge the AI CLIENT's \
turns for drift.
- The human is the COUNSELOR trainee. Their speech reaches the AI via speech-to-\
text (STT), so it may be garbled. You assess garble on the COUNSELOR's turns.

Drift = the AI client going incoherent, off-character, off-topic, repetitive, \
or producing gibberish. Crucially, the AI is PLAYING a possibly distressed \
person: rambling, "I don't know what to do", emotional repetition, terse \
replies, and code-switching (e.g. Hinglish) can be REALISTIC PORTRAYAL, not \
drift. Set in_character=true in those cases.

Judge EACH AI turn INDEPENDENTLY using only what preceded it. Do not smooth over \
a bad turn because the conversation later recovers, and do not over-flag the \
neighbours of one bad turn.

Per AI turn, label:
- coherence (anchored, pick the closest):
  fully_coherent | minor_disfluency | degrading | mostly_incoherent | gibberish
- topic_label: on_topic | tangent | off_topic | gibberish
  (NOT drift: counselor-led topic change, code-switching/Hinglish, backchannels, \
terse-but-valid replies)
- in_character: is odd output realistic distressed-client portrayal?
- counselor_utterance_garbled: none | partial | severe — does the COUNSELOR \
transcript this turn replies to look STT-mangled?
- stt_error_type (only if garbled, else "none"): entity_swap | phonetic_garble | \
wrong_language | number_format | code_mix_fail | truncation
- ai_reply_failure_mode ("none" if clean): hallucination | context_lockin | \
wrong_language_reply | repetition | role_slip | wrong_intent
- root_attribution (consider the PRIOR ~3 turns; "none" if this is not a drift turn):
  stt_direct      — counselor turn garbled, AI reply sensible GIVEN that garble
  stt_cascade     — AI degrades now, but a garble 1-3 turns earlier is the root
  llm_direct      — inputs clean across the window, AI reply still incoherent
  context_lockin  — incoherent given clean input that referenced earlier context
- reasoning: one sentence.

Return one object per AI-client turn, in order, keyed by that turn's index.\
"""


def build_judge_prompt(
    transcript: List[TranscriptTurn],
    persona: str,
    language: str,
    scenario_goal: Optional[str] = None,
    rubric: Optional[str] = None,
) -> str:
    """Assemble the full judge prompt for one session.

    `rubric` is the static instruction block, normally fetched from prompt
    management; when None, falls back to DEFAULT_JUDGE_RUBRIC.
    """
    lines = [rubric or DEFAULT_JUDGE_RUBRIC, ""]
    lines.append(f"SESSION LANGUAGE: {language}")
    lines.append("")
    lines.append("AI CLIENT PERSONA / SCENARIO PROMPT:")
    lines.append(persona.strip() or "(none provided)")
    if scenario_goal:
        lines.append("")
        lines.append(f"SCENARIO GOAL: {scenario_goal.strip()}")
    lines.append("")
    lines.append("TRANSCRIPT (chronological):")
    for turn in transcript:
        role = turn.get("role", "?")
        speaker = "AI_CLIENT" if role == "client" else "COUNSELOR"
        idx = turn.get("turn_index")
        tag = f"[turn {idx}] " if (speaker == "AI_CLIENT" and idx is not None) else ""
        lines.append(f"{tag}{speaker}: {turn.get('text', '')}")
    lines.append("")
    lines.append(
        "Emit one judgment per AI_CLIENT turn (use its [turn N] index)."
    )
    return "\n".join(lines)
