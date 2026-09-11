"""Rubric for the recall-quality judge.

The question is deliberately narrow: given what the learner just said and what the client
said back, was the fact the turn called for among the five recalled, or sitting in the pool
below the cap?

That is NOT a question about whether the reply was good. The actor's reply quality is already
judged elsewhere (drift, language, filler). This judge exists because recall's ranking has five
weighted terms and a hard cap, and the weights have never been tunable from anything but
argument — the scores were computed and discarded every turn.
"""

from typing import List, Optional, Sequence

#: Prompt-management code, so the rubric can be corrected without a deploy. See the note in
#: ally-be's repository about the pinned version: an override changes what the labels mean
#: while every row still says v1, so bump PROMPT_VERSION alongside any change of definition.
RECALL_QUALITY_JUDGE_PROMPT_CODE = "recall_quality_rubric"

DEFAULT_JUDGE_RUBRIC = """You are auditing one turn of a counselling roleplay.

The CLIENT is played by an AI actor with a pool of backstory facts about itself. On each turn a
ranking picks at most five of those facts as "what the client has in mind". You are judging
THAT CHOICE — not the client's reply, not the counsellor's skill.

You are given the counsellor's turn, the client's reply, the facts that were RECALLED, and the
facts that were PASSED OVER (the next-highest scoring candidates that missed the cap).

Return exactly one verdict:

- `no_demand` — this turn called for no particular backstory. Acknowledgements, greetings,
  "mm-hmm", "go on", a question about something already fully answered. USE THIS FREELY: most
  turns of a real conversation are like this, and marking them as recall failures would make
  every other number meaningless. A turn only makes a demand when a specific piece of the
  client's history or circumstance would change what a truthful client could say.

- `well_chosen` — the turn made a demand and the recalled facts included what it called for.
  Extra recalled facts that went unused do not spoil this: five are always chosen and a turn
  rarely needs five.

- `missed_better` — the turn made a demand, and a PASSED-OVER fact answers it better than
  anything recalled did. Name it in `better_fact`, quoted exactly from the passed-over list.
  This is the finding that matters most: the material was there and the ranking buried it.

- `nothing_apt` — the turn made a demand and NEITHER list contains anything that answers it.
  This is a gap in the character's profile, not a ranking failure, and the fix is to write more
  backstory rather than to retune a weight.

IGNORE THE SCORES. They are shown so a score/substance mismatch can be reported, not so you can
defer to them. A high-scoring fact that has nothing to do with the turn is exactly what this
audit is for.

JUDGE THE TURN AS ASKED, not as you would have run the session. A fact is not "called for"
because it would have been interesting; it is called for when a client who had it in mind would
answer differently from one who did not.

Also list in `unused_selected` any recalled fact the turn gave no occasion for. One or two is
normal. A turn where all five went unused, repeatedly, says the cap is larger than the
conversation can use.
"""


def _format_facts(facts: Sequence[dict], label: str) -> str:
    if not facts:
        return f"{label}: none."
    lines = [f"{label}:"]
    for fact in facts:
        text = (fact.get("text") or "").strip()
        if not text:
            continue
        lines.append(
            f"- \"{text}\" (score {float(fact.get('score') or 0.0):.3f}, "
            f"cue hits {int(fact.get('cue_hits') or 0)}, "
            f"similarity {float(fact.get('similarity') or 0.0):.3f})"
        )
    return "\n".join(lines)


def build_judge_prompt(
    *,
    counsellor_turn: str,
    client_reply: str,
    selected: Sequence[dict],
    passed_over: Sequence[dict],
    stance: Optional[str] = None,
    cue_tier: Optional[str] = None,
    rubric: Optional[str] = None,
) -> str:
    """Assemble the judge prompt for one turn."""
    parts: List[str] = [rubric or DEFAULT_JUDGE_RUBRIC, ""]

    if stance:
        # The stance recall was scored for congruence with. A guarded client withholding a
        # fact is not a recall failure, and a judge that does not know the stance would read
        # it as one.
        parts.append(f"CLIENT'S STANCE ON THIS TURN: {stance}")
    if cue_tier:
        # Which source supplied the cues. `scenario` means nothing in the conversation
        # itself drove this selection, which is context for a weak choice.
        parts.append(f"CUES CAME FROM: {cue_tier}")

    parts += [
        "",
        f"COUNSELLOR SAID: {counsellor_turn.strip() or '(nothing recorded)'}",
        f"CLIENT REPLIED: {client_reply.strip() or '(nothing recorded)'}",
        "",
        _format_facts(selected, "RECALLED (what the client had in mind)"),
        "",
        _format_facts(passed_over, "PASSED OVER (scored below the cap)"),
    ]
    return "\n".join(parts)
