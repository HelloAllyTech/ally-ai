"""Judge prompt builder for the thinking-filler judge.

The rubric's hard problem is that most of what a filler does is legitimately
unremarkable. "Hmm" is a perfectly good filler and will be the right answer many
times in a session, so a rubric that rewards finding fault will mark a healthy
session down and a rubric that rewards leniency will never catch the failure
this eval exists for. The instructions below therefore spend most of their space
on what is NOT a fault — the same shape as the language-quality rubric, and for
the same reason.

There is no hand-labeled calibration set. Tune by spot-checking judged sessions,
bump ``FILLER_JUDGE.PROMPT_VERSION`` whenever the rubric changes, and re-judge
affected slices so comparisons stay inside one judge version.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.core.filler_quality.schemas import FillerObservation

# Prompt-management code for the judge rubric. The judge fetches the current
# version from ally-be and falls back to DEFAULT_FILLER_RUBRIC if unavailable.
FILLER_JUDGE_PROMPT_CODE = "filler_quality_judge_rubric"


class FillerStyleParams(BaseModel):
    """What the character was configured to sound like.

    Without these the judge has nothing to score register against and will fall
    back to scoring "is this a plausible English filler", which is not the
    question. Presence flags matter as much as the values: a low register score
    on a character with no configured style is a configuration gap, not a model
    failure, and the two need telling apart.
    """

    language_label: Optional[str] = None
    #: Authored examples of how this character speaks.
    style_exemplars: List[str] = Field(default_factory=list)
    #: Discourse particles this language/character actually uses.
    allowed_fillers: List[str] = Field(default_factory=list)


DEFAULT_FILLER_RUBRIC = """\
You annotate problems with THINKING FILLERS in a role-play counseling-training \
session, one filler at a time.

WHAT A THINKING FILLER IS: the AI plays the CLIENT. When the learner (the \
COUNSELOR trainee) stops speaking, the client's real reply takes a moment to \
form, so the client first utters something very short — a hesitation, an \
acknowledgement, a brief reflection — to hold the moment. That short utterance \
is the filler. The real reply follows immediately after it, and was generated \
SEPARATELY: the filler did not know what the reply would say.

You are given, for each filler: what the counselor had just said, the filler \
itself, and the real reply that followed.

TASK: for each filler emit zero or more findings. MOST FILLERS SHOULD HAVE ZERO \
FINDINGS — do not manufacture them. Never emit scores, ratings or overall \
judgments. Judge each filler on its own, using only what preceded it.

DIMENSIONS AND CATEGORIES (use exactly these; category must belong to its \
dimension):
- character_fit:
  * generic_for_character (could belong to any character in any scenario). \
ONLY emit this when the STYLE block below shows a configured style — on a \
character who was never given one, a generic filler is a configuration gap, \
not a fault.
  * wrong_register (formality or warmth wrong for this character in this \
moment — e.g. brisk cheer from a character who has just been distressed)
  * persona_break (breaks the character's voice or knowledge, or is in the \
wrong language or script)
- context_fit:
  * answers_earlier_turn (responds to a previous moment in the conversation \
rather than the turn it followed)
  * incongruous_reaction (a reaction this turn does not support — e.g. warm \
agreement after a hostile turn)
- safety:
  * committed (answers a question, states a fact, agrees or disagrees, takes a \
side, names a specific, or presupposes an outcome). A filler is spoken BEFORE \
the reply exists, so anything it commits to is a guess the reply may contradict \
a second later. Read the reply and ask whether the two sit oddly together.
  * echoes_specific (repeats a name, number or place back at the learner)

SEVERITY (pick the closest):
- minor: a listener might notice; the moment still works.
- major: the filler reads as wrong for the character or the moment.
- critical: breaks the illusion outright, or the real reply contradicts it.

NOT FINDINGS — never annotate these:
1. Brevity and emptiness. A filler is SUPPOSED to carry no information. "Hmm", \
"I see", "Right" are correct, complete fillers. Being short or unremarkable is \
not a fault of any kind.
2. Repetition across the session. Whether a phrase recurs is computed \
separately from the sequence of plays; it is not your concern and you cannot \
see enough to judge it.
3. In-character hesitation, vagueness or distress. A struggling client \
hesitates; that is the point.
4. Configured discourse particles (see ALLOWED FILLERS) — always correct.
5. The reply going somewhere unexpected. Only emit `committed` if the FILLER \
ITSELF asserted something; a surprising reply is not the filler's fault.
6. A filler whose length or energy does not match the reply. They are separate \
utterances and the filler need not set the reply up.
7. A generic filler on a character with no configured style (see \
generic_for_character above).

EVIDENCE: every finding quotes the shortest verbatim span (at most ~15 words) \
that exhibits it, in the original script. reasoning is ONE sentence.

Return one object per filler, in order, keyed by that filler's turn index, \
with its (usually empty) findings array.\
"""


def build_judge_prompt(
    observations: List[FillerObservation],
    persona: str,
    language: str,
    style_params: Optional[FillerStyleParams] = None,
    rubric: Optional[str] = None,
) -> str:
    """Assemble the full judge prompt for one session's fillers.

    ``rubric`` is the static instruction block, normally fetched from prompt
    management; when None, falls back to DEFAULT_FILLER_RUBRIC.
    """
    sp = style_params or FillerStyleParams()

    lines = [rubric or DEFAULT_FILLER_RUBRIC, ""]
    label = f"{sp.language_label} ({language})" if sp.language_label else language
    lines.append(f"SESSION LANGUAGE: {label}")

    lines.append("")
    lines.append("STYLE — how this character was configured to speak:")
    if sp.style_exemplars:
        for i, sample in enumerate(sp.style_exemplars, 1):
            text = str(sample).strip()
            if text:
                lines.append(f"  Example {i}: {text}")
    else:
        # Said explicitly rather than omitted: a judge that cannot see whether
        # style was configured will read every generic filler as a model
        # failure, when half the time it is an unconfigured scenario.
        lines.append("  (none configured — do not mark register down for being")
        lines.append("   generic when the character was never given a style)")
    fillers = ", ".join(sp.allowed_fillers) if sp.allowed_fillers else "none configured"
    lines.append(f"ALLOWED FILLERS: {fillers}")

    lines.append("")
    lines.append("AI CLIENT PERSONA / SCENARIO PROMPT:")
    lines.append(persona.strip() or "(none provided)")

    lines.append("")
    lines.append("FILLERS TO JUDGE:")
    for obs in observations:
        lines.append("")
        lines.append(f"[turn {obs.turn_index}]")
        lines.append(
            f"  COUNSELOR SAID: {obs.learner_utterance.strip() or '(nothing)'}"
        )
        lines.append(f"  FILLER: {obs.filler_text.strip()}")
        lines.append(
            f"  REAL REPLY THAT FOLLOWED: {obs.reply_text.strip() or '(none)'}"
        )

    lines.append("")
    lines.append(
        "Emit one object per filler (use its [turn N] index) with its "
        "(usually empty) findings array."
    )
    return "\n".join(lines)
