"""Judge prompt builder for the feedback-groundedness judge.

v1 rubric. Bump ``FEEDBACK_GROUNDEDNESS_JUDGE.PROMPT_VERSION`` whenever this
changes, so a re-judge coexists with prior runs instead of overwriting them.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

# Prompt-management code for this rubric; the registry version is
# authoritative and the inline default below is the fallback.
GROUNDEDNESS_JUDGE_PROMPT_CODE = "feedback_groundedness_rubric"


class TranscriptTurn(TypedDict, total=False):
    role: str  # "client" (AI) | "counselor" (human)
    text: str
    turn_index: int


class FeedbackClaim(TypedDict):
    claim_index: int
    kind: str  # "positive" | "improvement"
    text: str


DEFAULT_JUDGE_RUBRIC = """\
You check whether written feedback about a counselling practice session is \
TRUE of what actually happened in that session.

WHO IS WHO (do not get this backwards):
- The COUNSELLOR is the human trainee. The feedback is about THEM.
- The CLIENT is played by an AI. Nothing the client did is the counsellor's \
behaviour.

You are given the transcript and a list of feedback claims. Judge each claim \
INDEPENDENTLY against the transcript. Do not reward a claim for sounding \
reasonable, and do not punish one for being harshly worded — the only question \
is whether the transcript bears it out.

For each claim emit:

- verdict:
  * supported     — the transcript shows what the claim says
  * unsupported   — nothing in the transcript corroborates it (it may be true \
of some other session, or invented)
  * contradicted  — the transcript shows the OPPOSITE of the claim
  * misattributed — the behaviour did occur, but the claim pins it to the \
wrong turn, or credits the counsellor with something the CLIENT said

- quotes_transcript (true/false): does the claim cite specific words as having \
been said — quoted, or closely paraphrased as speech — rather than only \
describing behaviour in general terms?

- quote_is_accurate (true/false/null): only when quotes_transcript is true. \
Does that wording actually appear in the transcript? Null otherwise.

- reasoning: one sentence.

TWO THINGS TO WEIGH CAREFULLY:

1. An "improvement" claim saying the counsellor FAILED to do something is \
CONTRADICTED if the transcript shows them doing it — even once, and even if \
they did it clumsily or in different words than the feedback expected. This is \
the failure counsellors report as most damaging: being marked down for work \
they visibly did. Do not require the counsellor's phrasing to match any \
particular script.

2. Judge the claim's SUBSTANCE, not its style. Feedback may be blunt, generic \
or repetitive and still be entirely accurate; that is a separate problem from \
being wrong, and it is not what you are measuring here.\
"""


def build_judge_prompt(
    transcript: List[TranscriptTurn],
    claims: List[FeedbackClaim],
    language: str,
    rubric: Optional[str] = None,
) -> str:
    """Assemble the groundedness prompt for one session's feedback."""
    lines = [rubric or DEFAULT_JUDGE_RUBRIC, ""]
    lines.append(f"SESSION LANGUAGE: {language}")
    lines.append("")
    lines.append("TRANSCRIPT (chronological):")
    for turn in transcript:
        speaker = "AI_CLIENT" if turn.get("role") == "client" else "COUNSELLOR"
        idx = turn.get("turn_index")
        tag = f"[turn {idx}] " if idx is not None else ""
        lines.append(f"{tag}{speaker}: {turn.get('text', '')}")
    lines.append("")
    lines.append("FEEDBACK CLAIMS TO CHECK:")
    for claim in claims:
        lines.append(
            f"[claim {claim['claim_index']}] ({claim['kind']}) {claim['text']}"
        )
    lines.append("")
    lines.append(
        "Emit one judgment per claim, in order, keyed by its claim_index."
    )
    return "\n".join(lines)
