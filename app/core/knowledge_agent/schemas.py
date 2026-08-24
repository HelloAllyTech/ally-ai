"""Structured output shapes for the knowledge agent's LLM calls.

These are passed as the provider's response schema (Gemini `response_schema`, Anthropic
forced-tool `input_schema`), so keep them flat and avoid `Optional` — an absent value
comes back as the empty default rather than null, which keeps the parsed object total.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class AnswerIntent(str, Enum):
    """What the agent decided to do with the question.

    DECLINE and CLARIFY are first-class outcomes rather than a phrase to string-match in
    the answer text, for the same reason PlanIntent.CLARIFY is in the analytics agent: a
    confident invention is the one failure mode a clinical Q&A bot must not have, and
    "did the model refuse" has to be a field the code can branch on.

    The two are NOT interchangeable. DECLINE means the corpus does not cover the
    question — a knowledge gap, which becomes an unanswered-question row for an admin to
    fill. CLARIFY means the question was too vague to retrieve against at all ("help
    with a client"), which is not a gap and must not pollute that queue.
    """

    ANSWER = "answer"
    DECLINE = "decline"
    CLARIFY = "clarify"


class DeclineReason(str, Enum):
    """Why an answer was not given. NONE when one was.

    Separating NO_HITS / BELOW_THRESHOLD (decided in code, before any LLM call) from
    MODEL_DECLINED (decided by the model with the passages in front of it) is what makes
    threshold tuning possible: a pile of BELOW_THRESHOLD means the floor is too high, a
    pile of MODEL_DECLINED means retrieval is finding the wrong passages.

    TRANSLATION_FAILED is a THIRD, distinct kind of gap: query translation failed and
    retrieval fell back to searching the worker's original-language text (see
    KnowledgeAgentService.prepare_query). A NO_HITS/BELOW_THRESHOLD decline that follows
    a translation failure is not evidence the corpus lacks this topic — it is evidence
    only that this question was searched in the wrong language — so it must not be
    reported, or answered for, as the same thing. Without this, "we don't cover this"
    and "we couldn't understand your language" are indistinguishable to the caller and
    both silently become the same unanswered-question row.
    """

    NONE = "none"
    NO_HITS = "no_hits"
    BELOW_THRESHOLD = "below_threshold"
    MODEL_DECLINED = "model_declined"
    TRANSLATION_FAILED = "translation_failed"
    ERROR = "error"


class KnowledgeAnswer(BaseModel):
    """The answer LLM's structured output."""

    intent: AnswerIntent = AnswerIntent.DECLINE
    # The answer, the clarifying question, or empty when declining. Plain text: WhatsApp
    # renders only *bold* and _italic_, so markdown headings and links arrive as literal
    # punctuation.
    answer: str = ""
    # 1-based numbers of the passages the answer actually used, as presented in the
    # prompt. Numbers rather than ids because a model asked to echo a UUID will sooner
    # or later invent a plausible one, and a fabricated id cannot be detected — whereas
    # an out-of-range integer can be, and is dropped.
    citations: List[int] = Field(default_factory=list)
    # BCP-47 tag of `answer`. The worker asked in their own language and must be
    # answered in it; this is the model reporting what it actually did, which is what
    # gets stored.
    language: str = "en"
    confidence: float = 0.0


class TranslatedQuery(BaseModel):
    """The query-preparation call's structured output.

    One call does double duty: it detects the language AND produces an English query to
    embed. Splitting them would mean two round trips on the latency budget of a worker
    waiting on WhatsApp, and detection alone is useless here — the reason to detect is
    to decide whether to translate.
    """

    # False when the question is already English, in which case `english_query` is
    # ignored and the original text is embedded unchanged.
    is_english: bool = True
    # BCP-47 tag of the question as asked. Romanised Hinglish is reported as `hi`, not
    # `en`: script is not language, and a Latin-script Hindi question embedded as
    # English retrieves badly.
    language: str = "en"
    # The question restated in English, preserving clinical terms, drug names and proper
    # nouns verbatim. Empty when is_english.
    english_query: str = ""


class CrisisVerdict(BaseModel):
    """The crisis classifier's structured output.

    Deliberately biased towards the false positive, and the schema says so where a
    reader will see it: a low `confidence` alongside ``is_crisis=True`` is a correct and
    expected combination, not a signal to override the verdict with a threshold. Keyword
    matching catches the explicit disclosures; this exists for the indirect ones ("I
    can't keep doing this"), which are the norm rather than the exception among people
    whose job is to notice them in others.
    """

    is_crisis: bool = False
    # The phrase from the message that drove the verdict, copied verbatim. What an admin
    # reads when checking whether the classifier is calibrated — a paraphrase would hide
    # the miscalibration by restating the model's reasoning instead of the evidence.
    signal: str = ""
    confidence: float = 0.0
