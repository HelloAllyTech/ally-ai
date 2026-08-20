"""Judge prompt builder for the conversation drift judge.

This is the v2 rubric — a STARTING POINT to calibrate against a hand-labeled
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

CLIENTHOOD — is the AI still a client seeking help, or has it turned helpful?

ANSWER role_inversion, offered_solution, solutions_offered, resistance_briefed \
AND introduced_new_information ON EVERY AI-CLIENT TURN. "No" is false or 0 — \
never a missing field. A turn where nothing happened still needs its answers, \
because an omitted label removes that turn from the measurement instead of \
counting as a clean one.

- role_inversion (true/false): did the AI ask the COUNSELOR about the counselor \
(their views, feelings, experience) or give the counselor advice? A client \
asking for help — "what should I do?", "is that normal?" — is NOT inversion. \
Inversion is the AI taking the counselor's chair.
- offered_solution (true/false): did the AI propose a solution or coping plan \
for its OWN problem, unprompted, instead of letting the counselor get there?
- solutions_offered (integer): how many DISTINCT such solutions this turn. 0 if \
none. Count them; do not judge whether that is too many.
- resistance_briefed (true/false): does the persona/scenario brief call for \
resistance, denial or reluctance? Read this from the BRIEF, not from this turn \
— your answer will be the same for every turn in the session.

PROGRESSION — did the conversation move?
- introduced_new_information (true/false): did this turn add anything the client \
had not already said — a new detail, feeling, event or objection? Restating \
earlier content in different words is FALSE.
- stuck_is_appropriate (true/false/null): only when introduced_new_information \
is false. TRUE if holding the same position was correct portrayal given the \
brief and what the counselor just did — a resistant client should NOT yield to \
a weak or premature intervention. FALSE if the client should have moved and did \
not. NULL when the turn did advance.

  Being "stuck" is not automatically a failure. A client who holds their ground \
against a poor intervention is behaving correctly; one who repeats because the \
generation lost the thread is not. That distinction is the whole point of this \
label — do not collapse it.

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


LEAN_LABELS_INSTRUCTION = """\

BACKFILL MODE — LABELS ONLY.

The rubric above is the SAME rubric, unchanged, and the definitions in it are \
the ones to apply. What changes is only which FIELDS you return.

These turns were already judged under the earlier rubric and those judgments \
are kept, so do not re-emit them. Omit coherence, topic_label, in_character, \
counselor_utterance_garbled, stt_error_type, ai_reply_failure_mode and \
root_attribution entirely.

ANSWER ALL FIVE OF THESE ON EVERY AI-CLIENT TURN. "No" is false or 0 — never a \
missing field. A turn where nothing happened still needs its answers, because \
an omitted label removes that turn from the measurement instead of counting as \
a clean one:

  role_inversion              true / false, every turn
  offered_solution            true / false, every turn
  solutions_offered           an integer, 0 when none, every turn
  resistance_briefed          true / false, every turn (same answer all session)
  introduced_new_information  true / false, every turn

Then, conditionally:

  stuck_is_appropriate   ONLY when introduced_new_information is false. Null \
when the turn advanced — the rubric defines no answer there.

  reasoning              one short sentence ONLY where a label actually fires. \
Null on clean turns. This is the ONLY field you may leave out: justifying \
"nothing happened" on every turn is the largest avoidable cost in this job and \
is read by nobody.\
"""


def build_lean_labels_prompt(
    transcript: List[TranscriptTurn],
    persona: str,
    language: str,
    scenario_goal: Optional[str] = None,
    rubric: Optional[str] = None,
) -> str:
    """The same prompt as the full judge, asking for only the added labels.

    Deliberately built on top of ``build_judge_prompt`` with the SAME rubric
    text rather than a trimmed copy of the clienthood/progression sections. Two
    prompts that define the same labels in their own words will drift, and when
    they do, backfilled history and live data disagree — which shows up in a
    chart as a step change indistinguishable from a real regression.

    The rubric is a couple of thousand input tokens and input is an eighth the
    price of output; the entire saving here is in what comes BACK. Measured on
    production sessions, the full judge averages 2.4k prompt against 3.8k
    completion, so trimming the response is where the money is.
    """
    return (
        build_judge_prompt(transcript, persona, language, scenario_goal, rubric)
        + "\n"
        + LEAN_LABELS_INSTRUCTION
    )
