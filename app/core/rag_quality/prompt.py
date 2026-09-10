"""Judge prompt builder for the RAG-quality judge.

v1 rubric. Bump ``RAG_QUALITY_JUDGE.PROMPT_VERSION`` whenever this changes, so
a re-judge coexists with prior runs instead of overwriting them — a rate is
only comparable within one (MODEL, PROMPT_VERSION) pair.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

# Prompt-management code for this rubric; the registry version is
# authoritative and the inline default below is the fallback.
RAG_QUALITY_JUDGE_PROMPT_CODE = "rag_quality_rubric"


class JudgedPassage(TypedDict, total=False):
    chunk_id: str
    document_title: str
    section_path: str
    similarity: float
    text: str


# What each corpus is FOR. Without this the judge cannot label relevance
# honestly: a passage on "what makes a speech sample useful" answers a craft
# question and is tangential to a subject question, and the same text would be
# scored differently depending on which corpus asked.
CORPUS_PURPOSE = {
    "character_library": (
        "The CHARACTER LIBRARY corpus. Its reader is an interviewer agent "
        "building a fictional client for counsellor practice. It holds two "
        "kinds of material: CRAFT guidance on how to construct a believable "
        "person (what makes a speech sample useful, why a stance belongs in a "
        "scenario's states rather than in static backstory), and "
        "LIVED-EXPERIENCE or observational material about what a situation is "
        "actually like (how a condition presents, what a working life "
        "involves, how someone in that position really talks). Clinical "
        "guidelines and treatment protocols do NOT belong here and are "
        "tangential at best."
    ),
    "whatsapp_qa": (
        "The WHATSAPP Q&A corpus. Its reader is a bot answering a mental "
        "healthcare worker's question, with citations. It holds clinical "
        "guidance and reference material — guidelines, intervention manuals, "
        "protocols. A passage is relevant when it helps ANSWER the question "
        "that was asked."
    ),
}

DEFAULT_JUDGE_RUBRIC = """\
You are auditing one retrieval from a knowledge corpus. You are given the \
QUERY that was issued, the corpus's purpose, and the PASSAGES that came back \
with their similarity scores.

Judge each passage INDEPENDENTLY against the query, then judge the set.

IGNORE THE SIMILARITY SCORES when deciding relevance. They are shown only so \
you can flag a mismatch between score and substance; a high score is not \
evidence of relevance and a low one is not evidence against it. The whole \
point of this audit is that those scores are not trustworthy on their own.

For each passage emit:

- relevance:
  * relevant    — bears directly on the query; someone writing from this query \
would actually use it
  * tangential  — same subject area, but does not answer the question
  * irrelevant  — not about the question at all

- superficial_match (true/false): true when the passage looks retrievable only \
because it SHARES WORDS with the query rather than sharing meaning. A passage \
about a "speech" at a wedding retrieved for a query about a person's "speech" \
patterns is the clearest form. This is what embedding similarity is worst at, \
so name it when you see it even if you also called the passage tangential.

- reasoning: one sentence.

Then judge the set as a whole:

- sufficiency:
  * sufficient     — the set answers the query
  * partial        — something useful, but with a gap a writer would notice
  * nothing_useful — nothing here helps, whatever the scores said

- missing: when sufficiency is not `sufficient`, what a writer would still \
need, in one phrase. Empty otherwise.

TWO THINGS TO WEIGH CAREFULLY:

1. TANGENTIAL IS NOT A SOFTER IRRELEVANT. It is its own failure and the more \
dangerous one, because tangential material is plausible: it is on-topic enough \
to be absorbed into a draft and wrong enough to make that draft about someone \
else. Use it whenever a passage would survive a skim but not answer the \
question.

2. JUDGE THE QUERY THAT WAS ASKED, not the one you would have asked. A query \
may be clumsily phrased or narrower than the writer's real need; a passage that \
answers the real need but not the query as written is `tangential`, and the \
`missing` field is where you say what the query should have reached for.\
"""


def build_judge_prompt(
    query: str,
    corpus: str,
    passages: List[JudgedPassage],
    min_similarity: float,
    rubric: Optional[str] = None,
) -> str:
    """Assemble the RAG-quality prompt for one retrieval."""
    lines = [rubric or DEFAULT_JUDGE_RUBRIC, ""]
    lines.append("CORPUS:")
    lines.append(
        CORPUS_PURPOSE.get(corpus, f"An unrecognised corpus ({corpus}).")
    )
    lines.append("")
    lines.append(f"QUERY AS ISSUED: {query}")
    # Stated so the judge can see when everything returned sits just above the
    # cut — a set of barely-passing passages reads differently from a set that
    # cleared comfortably.
    lines.append(f"SIMILARITY FLOOR IN FORCE: {min_similarity}")
    lines.append("")
    if not passages:
        lines.append("PASSAGES RETURNED: none.")
        lines.append("")
        lines.append(
            "Emit an empty passages list and judge sufficiency as "
            "`nothing_useful`, with `missing` naming what the query needed."
        )
        return "\n".join(lines)

    lines.append("PASSAGES RETURNED:")
    for passage in passages:
        header = f"[{passage['chunk_id']}] {passage.get('document_title', '')}"
        section = passage.get("section_path")
        if section:
            header += f" · {section}"
        similarity = passage.get("similarity")
        if similarity is not None:
            header += f" (similarity {similarity:.4f})"
        lines.append(header)
        lines.append(passage.get("text", ""))
        lines.append("")
    lines.append(
        "Emit one judgment per passage, keyed by its chunk_id exactly as "
        "given, then the set-level judgment."
    )
    return "\n".join(lines)
