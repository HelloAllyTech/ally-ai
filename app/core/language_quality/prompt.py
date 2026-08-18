"""Judge prompt builder for the language-quality judge.

This is the v1 rubric per language-eval-judge-schema.md. There is NO
hand-labeled calibration set (PRD NFR1): the rubric is tuned via spot-checks
of judged sessions; bump ``LANGUAGE_JUDGE.PROMPT_VERSION`` whenever it changes
and re-judge affected slices so comparisons stay within one judge version.

Few-shot examples are intentionally left as a TODO: harvest per language from
real judged sessions (at minimum one garbled-STT negative, one
in-character-distress negative, and one diglossia positive).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

# Reuse the drift judge's transcript turn shape — ally-be builds transcripts
# identically for both judges.
from app.core.drift.prompt import TranscriptTurn  # noqa: F401  (re-exported)

# Prompt-management code for the judge rubric. The judge fetches the current
# version from ally-be and falls back to DEFAULT_JUDGE_RUBRIC if unavailable.
LANGUAGE_JUDGE_PROMPT_CODE = "language_quality_judge_rubric"


class LanguageEvalParams(BaseModel):
    """Per-language eval config (from ally-be languages.evalConfig; all
    optional — absent values render as 'unknown' so the judge stays usable
    before per-language config is populated)."""

    language_label: Optional[str] = None  # e.g. "Tamil (India)"
    target_variety: Optional[str] = None  # e.g. "colloquial spoken Tamil"
    diglossia: Optional[bool] = None
    code_switch_partners: List[str] = Field(default_factory=list)


class ScenarioStyleParams(BaseModel):
    """Presence flags + fillers from the scenario's per-language style config.
    These let the judge label the prompt-vs-model side of each appropriateness
    error (isolation_basis = persona_specified / persona_unspecified)."""

    register_directive_configured: Optional[bool] = None
    style_exemplars_configured: Optional[bool] = None
    allowed_fillers: List[str] = Field(default_factory=list)
    engine: Optional[str] = None  # SIMULATION | ROLEPLAY_V2
    locked_content_exists: Optional[bool] = None


# Fallback rubric — keep in sync with the seeded prompt-management version
# (language_quality_judge_rubric); the registry version is authoritative.
DEFAULT_JUDGE_RUBRIC = """\
You annotate LANGUAGE-QUALITY ERRORS in a role-play counseling-training \
session, turn by turn.

ROLES (do not get these backwards):
- The AI plays the CLIENT (the person seeking help). You annotate ONLY the AI \
CLIENT's turns.
- The human is the COUNSELOR trainee. Their speech reaches the AI via \
speech-to-text (STT) and may be garbled. For each AI turn you rate \
input_garbled from the COUNSELOR utterance it replies to; garble is never the \
AI's error.

TASK: for each AI CLIENT turn emit zero or more error annotations. MOST TURNS \
SHOULD HAVE ZERO ERRORS — do not manufacture findings. Never emit scores, \
grades, or overall judgments. Judge each turn independently using only what \
preceded it; do not smooth over a bad turn because the session later recovers, \
and do not over-flag the neighbours of one bad turn.

DIMENSIONS AND CATEGORIES (use exactly these; category must belong to its \
dimension):
- understanding: misinterpreted_intent (answers a different intent than the \
counselor expressed) | ignored_context (ignores information clearly \
established earlier)
- adequacy: off_topic (unrelated to the turn or scenario) | hallucination \
(asserts persona/backstory/world facts not in, and not reasonably implied by, \
the configured persona) | omission (fails to convey content the turn clearly \
required)
- fluency: grammar (an error a native speaker would not make) | script_error \
(wrong script, broken glyphs, transliteration where native script expected) | \
disfluency (unnatural repetition/fragmentation beyond configured fillers) | \
truncation (cut off mid-thought)
- coherence: contradiction (contradicts what the persona previously \
established) | non_sequitur (no discernible connection to the conversation)
- register: too_formal_diglossia (literary/textbook variety where the \
colloquial spoken variety is expected) | too_casual (below the socially \
expected register)
- dialect_lexicon — LEXICAL CORRECTNESS. Three distinct failures; do not \
collapse them:
  * nonexistent_word (a token that is not a word in the target language at \
all: invented, mis-transliterated, or a malformed compound. If a native \
speaker would say "that is not a word", this is the category)
  * wrong_sense (a real word of the target language used with a meaning it \
does not carry here — the sentence parses but means something else, or \
something odd)
  * wrong_regional_variety (a real word with the right meaning, but from \
outside the configured regional variety)

  Flag these on the MEANING, not on formality — a word that is merely bookish \
belongs in register, not here. This dimension has historically under-fired \
while partner organisations reported exactly these problems as blocking, so \
when a word looks wrong, prefer labelling it over letting it pass.
- colloquialness: literal_translation_stilt (calqued, translated-sounding \
phrasing no native speaker would produce)
- persona_social: too_blunt (socially inappropriate directness given the \
emotional context) | persona_break (voice/knowledge/attitude inconsistent \
with the configured character, including assistant-like behavior)
- codeswitch: foreign_token_leak (unintended other-language tokens where the \
target language was expected) | unnatural_switch (a switch at a boundary or \
of a kind a real bilingual speaker would not produce)

SEVERITY (pick the closest):
- minor: noticeable to a native speaker; meaning and training value intact.
- major: degrades believability or meaning; a trainee would notice something \
is off.
- critical: breaks the simulation for this turn — meaning lost, persona \
shattered, or output unusable.

NOT ERRORS (never annotate these):
1. In-character distress — rambling, hesitation, "I don't know", emotional \
repetition, terse replies are realistic portrayal of a distressed client.
2. Natural code-switching with the configured partner language(s) — that is \
CORRECT behavior. Only leakage into other languages or unnatural switch \
points are errors.
3. Configured filler words/backchannels (see ALLOWED FILLERS) — never \
disfluency.
4. Intentional withholding or deflection of locked/secret content — a vague \
or deflecting answer about a secret is CORRECT persona behavior, never \
omission or non_sequitur. When LOCKED CONTENT EXISTS is yes, prefer no \
annotation over guessing.
5. Counselor-led topic changes — following the trainee somewhere new is not \
off_topic.
6. Register mirroring — matching a casual counselor's register is not \
too_casual unless it breaks persona.

CONDITIONING ON STT: if the counselor's input was garbled, still annotate \
fluency/register/dialect/persona errors normally, set input_garbled \
accordingly, and use isolation_basis=input_garbled for any understanding or \
adequacy oddity plausibly caused by the garble.

ISOLATION BASIS (per annotation, use exactly one):
- input_clean: counselor input this turn (and recent turns) is clean — the \
error is attributable to generation, not mishearing.
- input_garbled: plausibly caused or excused by garbled input.
- persona_specified: the configuration explicitly asks for the violated \
expectation (see REGISTER DIRECTIVE / STYLE EXEMPLARS flags) — the model \
ignored an instruction it was given.
- persona_unspecified: the configuration never asked for it — likely a \
configuration gap, not a model failure.
- pattern_systemic: the same error class recurs across multiple turns.

EVIDENCE: every annotation quotes the shortest span (at most ~15 words) that \
exhibits the error, verbatim, in the original script. reasoning is one \
sentence.

Return one object per AI-client turn, in order, keyed by that turn's index, \
with its input_garbled level and its (usually empty) errors array.\
"""


def _yn(value: Optional[bool]) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def build_judge_prompt(
    transcript: List[TranscriptTurn],
    persona: str,
    language: str,
    language_params: Optional[LanguageEvalParams] = None,
    style_params: Optional[ScenarioStyleParams] = None,
    rubric: Optional[str] = None,
) -> str:
    """Assemble the full judge prompt for one session.

    `rubric` is the static instruction block, normally fetched from prompt
    management; when None, falls back to DEFAULT_JUDGE_RUBRIC.
    """
    lp = language_params or LanguageEvalParams()
    sp = style_params or ScenarioStyleParams()

    lines = [rubric or DEFAULT_JUDGE_RUBRIC, ""]
    label = f"{lp.language_label} ({language})" if lp.language_label else language
    lines.append(f"SESSION LANGUAGE: {label}")
    lines.append(f"TARGET VARIETY: {lp.target_variety or 'unknown'}")
    lines.append(f"DIGLOSSIA APPLIES: {_yn(lp.diglossia)}")
    partners = ", ".join(lp.code_switch_partners) if lp.code_switch_partners else "unknown"
    lines.append(f"CODE-SWITCH PARTNERS: {partners}")
    lines.append(
        f"REGISTER DIRECTIVE CONFIGURED: {_yn(sp.register_directive_configured)}"
    )
    lines.append(f"STYLE EXEMPLARS CONFIGURED: {_yn(sp.style_exemplars_configured)}")
    fillers = ", ".join(sp.allowed_fillers) if sp.allowed_fillers else "none configured"
    lines.append(f"ALLOWED FILLERS: {fillers}")
    if sp.engine:
        lines.append(f"ENGINE: {sp.engine}")
    lines.append(f"LOCKED CONTENT EXISTS: {_yn(sp.locked_content_exists)}")
    lines.append("")
    lines.append("AI CLIENT PERSONA / SCENARIO PROMPT:")
    lines.append(persona.strip() or "(none provided)")
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
        "Emit one judgment per AI_CLIENT turn (use its [turn N] index), with "
        "input_garbled and the errors array (usually empty)."
    )
    return "\n".join(lines)
