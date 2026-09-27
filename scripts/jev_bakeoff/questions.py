"""Jev questions for the Phase 0 bake-off, built from the Gemini judges' rubrics.

Wording is taken from `app/core/drift/prompt.py` and
`app/core/feedback_groundedness/prompt.py`, so agreement measures the MODEL,
not a difference in what was asked. These are deliberately untuned: DSPy tuning
of the wording is Phase 1, and it needs this untuned baseline to beat.

Each label maps to one typed question:
  drift        coherence → Score; topic, garble, failure mode → Choice;
               the booleans → Noul
  groundedness verdict → Choice; quotes_transcript, quote_is_accurate → Noul
"""

from __future__ import annotations

ROLES = (
    "This is a role-play counselling-training session. The AI plays the CLIENT "
    "(the person seeking help); the human is the COUNSELOR trainee, whose speech "
    "reaches the AI through speech-to-text and may be garbled."
)

# Low → high, as Score requires. Index = COHERENCE_RANK in drift/schemas.py.
COHERENCE_LEVELS = [
    "gibberish",
    "mostly_incoherent",
    "degrading",
    "minor_disfluency",
    "fully_coherent",
]

DRIFT_QUESTIONS = {
    "coherence": {
        "type": "score",
        "instructions": (
            f"{ROLES} How coherent is the AI_CLIENT reply in turn_being_judged? "
            "Emotional repetition, rambling, terse replies and code-switching "
            "(e.g. Hinglish) can be realistic portrayal of a distressed person."
        ),
        "criteria": [
            "gibberish: not language, or unreadable output",
            "mostly_incoherent: mostly fails to make sense",
            "degrading: noticeably losing coherence",
            "minor_disfluency: small slips, still clearly understandable",
            "fully_coherent: clear and sensible",
        ],
    },
    "topic_label": {
        "type": "choice",
        "instructions": (
            f"{ROLES} Is the AI_CLIENT reply in turn_being_judged on topic? "
            "A counselor-led topic change, code-switching, backchannels and "
            "terse-but-valid replies are NOT off topic."
        ),
        "criteria": {
            "on_topic": "Stays with the conversation and the client's situation",
            "tangent": "Wanders to a loosely related side topic",
            "off_topic": "Unrelated to the conversation",
            "gibberish": "Not meaningful language",
        },
    },
    "in_character": {
        "type": "noul",
        "instructions": (
            f"{ROLES} Is anything odd in the AI_CLIENT reply in turn_being_judged "
            "realistic in-character portrayal of a distressed client, rather than drift?"
        ),
    },
    "counselor_utterance_garbled": {
        "type": "choice",
        "instructions": (
            f"{ROLES} Does the COUNSELOR utterance in turn_being_judged look "
            "mangled by speech-to-text?"
        ),
        "criteria": {
            "none": "Reads as what a person would plausibly say",
            "partial": "Some words look mis-transcribed, meaning still recoverable",
            "severe": "Mostly mis-transcribed, meaning lost",
        },
    },
    "ai_reply_failure_mode": {
        "type": "choice",
        "instructions": f"{ROLES} How did the AI_CLIENT reply in turn_being_judged fail, if at all?",
        "criteria": {
            "none": "The reply is clean",
            "hallucination": "Invents facts that contradict the brief or conversation",
            "context_lockin": "Stuck on earlier context, ignoring what was just said",
            "wrong_language_reply": "Replies in the wrong language",
            "repetition": "Repeats earlier content without purpose",
            "role_slip": "Stops acting as the client",
            "wrong_intent": "Misreads what the counselor meant",
        },
    },
    "role_inversion": {
        "type": "noul",
        "instructions": (
            f"{ROLES} In turn_being_judged, did the AI_CLIENT ask the COUNSELOR about "
            "the counselor themselves (their views, feelings, experience) or give the "
            "counselor advice? A client asking for help ('what should I do?') is NOT this."
        ),
    },
    "offered_solution": {
        "type": "noul",
        "instructions": (
            f"{ROLES} In turn_being_judged, did the AI_CLIENT propose a solution or "
            "coping plan for its OWN problem, unprompted, instead of letting the "
            "counselor get there?"
        ),
    },
    "introduced_new_information": {
        "type": "noul",
        "instructions": (
            f"{ROLES} Does the AI_CLIENT reply in turn_being_judged add anything the "
            "client had not already said in recent_conversation: a new detail, feeling, "
            "event or objection? Restating earlier content in other words is NO."
        ),
    },
    "resistance_briefed": {
        "type": "noul",
        "instructions": (
            "Does the client_brief call for the client to show resistance, denial or "
            "reluctance?"
        ),
    },
}

GROUNDEDNESS_ROLES = (
    "This is a counselling practice session. The COUNSELLOR is the human trainee "
    "and the feedback is about THEM. The CLIENT is played by an AI; nothing the "
    "client did is the counsellor's behaviour."
)


def groundedness_questions(claim: dict) -> dict:
    """Three questions about one feedback claim, keyed for that claim."""
    key = f"{claim['kind']}_{claim['claim_index']}"
    text = claim["text"] or ""
    note = (
        " This is an 'improvement' claim: if it says the counsellor FAILED to do "
        "something and the transcript shows them doing it, even once or clumsily, "
        "it is contradicted."
        if claim["kind"] == "improvement"
        else ""
    )
    return {
        f"{key}__verdict": {
            "type": "choice",
            "instructions": (
                f"{GROUNDEDNESS_ROLES} How does this feedback claim stand up against "
                f"the transcript? Judge substance, not style.{note}\nCLAIM: {text}"
            ),
            "criteria": {
                "supported": "The transcript shows what the claim says",
                "unsupported": "Nothing in the transcript corroborates it",
                "contradicted": "The transcript shows the opposite of the claim",
                "misattributed": (
                    "The behaviour occurred, but the claim pins it to the wrong turn "
                    "or credits the counsellor with something the client said"
                ),
            },
        },
        f"{key}__quotes_transcript": {
            "type": "noul",
            "instructions": (
                "Does this claim cite specific words as having been said, quoted or "
                f"closely paraphrased as speech?\nCLAIM: {text}"
            ),
        },
        f"{key}__quote_is_accurate": {
            "type": "noul",
            "instructions": (
                "Do the words this claim cites as said actually appear in the "
                f"transcript?\nCLAIM: {text}"
            ),
        },
    }
