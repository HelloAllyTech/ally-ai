"""The WhatsApp Q&A bot's retrieval-augmented answering agent.

Stateless, like the analytics agent: conversation history arrives on the request and
this service owns no database. It reads the KnowledgeChunk index and calls an LLM;
ally-be owns every row, every phone number and every decision about what to do with the
result.

The pipeline, and why it is in this order:

    detect + translate → embed → retrieve → deterministic decline gate → answer

Translation comes FIRST because retrieval happens in embedding space: a Hindi or Tamil
question embedded by text-embedding-3-small and matched against an English corpus
retrieves badly, and no amount of prompt quality recovers from being handed the wrong
passages. The failure is invisible from the answer alone — the model dutifully answers
from whatever it was given — which is exactly why it is worth an extra call.

The deterministic gate comes BEFORE the answer call so that "the corpus does not cover
this" is decided by a threshold the admin can see and tune, not by a model's mood, and
so that the common case of a genuinely uncovered question costs no generation tokens at
all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.knowledge_agent.prompt import (
    ANSWER_PROMPT_PATH,
    CRISIS_PROMPT_PATH,
    TRANSLATE_PROMPT_PATH,
    build_answer_prompt,
    build_crisis_prompt,
    build_translate_prompt,
)
from app.core.knowledge_agent.schemas import (
    AnswerIntent,
    CrisisVerdict,
    DeclineReason,
    KnowledgeAnswer,
    TranslatedQuery,
)
from app.core.knowledge_base.knowledge_chunk_service import KnowledgeChunkService
from app.core.llm.dispatch import generate_structured
from app.core.llm_usage.tasks import LLMTask
from app.exceptions.custom_exceptions import LLMInvocationFailedException
from app.prompts.resolver import get_backend_llm_overrides
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Answer length the reply is composed to. ally-be renders the source lines on top of
# this and hard-caps the whole message at 1600 characters (Twilio's ceiling, below
# Meta's 4096 — composing to the portable one is what keeps the provider seam honest).
DEFAULT_MAX_ANSWER_CHARS = 1400


class KnowledgeAgentService:
    """Answers a question from the knowledge corpus, or declines."""

    def __init__(self, chunk_service: KnowledgeChunkService) -> None:
        self.chunk_service = chunk_service

    # ------------------------------------------------------------------ query prep

    async def prepare_query(
        self, question: str, *, prompts: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, Optional[str]]:
        """
        Detect the question's language and, when it is not English, restate it in
        English.

        Returns ``(search_text, language, translated_query)``. `translated_query` is
        None when no translation happened, so the caller can show an admin exactly what
        was searched versus what was asked.

        A failure here degrades to searching the ORIGINAL text rather than failing the
        question. That is deliberate: a translation outage should cost retrieval quality
        for non-English questions, not take the bot down for everyone.
        """
        provider, model, temperature = get_backend_llm_overrides(
            TRANSLATE_PROMPT_PATH, prompts
        )
        prompt = build_translate_prompt(question, prompts=prompts)
        if not prompt:
            logger.error(
                "Translate prompt template resolved empty; skipping translation"
            )
            return question, "en", None

        try:
            parsed, _meta = await generate_structured(
                schema=TranslatedQuery,
                prompt=prompt,
                task=LLMTask.WHATSAPP_QUERY_TRANSLATE.value,
                provider=provider,
                model=model or settings.KNOWLEDGE_AGENT.TRANSLATE_MODEL,
                temperature=temperature if temperature is not None else 0.0,
            )
        except (
            Exception
        ) as e:  # noqa: BLE001 — never fail the question over translation
            logger.warning(
                "Query translation failed (%s); searching the original text",
                type(e).__name__,
            )
            return question, "en", None

        language = (parsed.language or "en").strip() or "en"
        english = (parsed.english_query or "").strip()

        if parsed.is_english or not english:
            return question, language, None
        return english, language, english

    # ------------------------------------------------------------------ crisis

    async def classify_crisis(
        self, message: str, *, prompts: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Decide whether a message is about a crisis happening now.

        A SEPARATE call from `answer`, not a stage inside it, for two reasons. It lets
        ally-be run the two concurrently, so the safety net costs no latency on a
        question that is not a crisis. And it keeps the precedence decision — crisis
        beats answer, always — in ally-be, which is the side that actually sends.

        The keyword rules in ally-be remain the first line and are still terminal: they
        are instant, free, and auditable. This exists for what they structurally cannot
        catch, which is indirect disclosure ("I can't keep doing this"). Neither
        replaces the other.

        A failure here returns ``is_crisis: false`` rather than raising, and `failed`
        says so. Raising would take the whole question down over the classifier, and
        defaulting to true would answer every question with a crisis message the moment
        an API key expired — a bot that only ever says "call a crisis line" is a broken
        bot, and workers would stop reading the message that matters. The keyword rules
        are what still hold in that window, which is why they were not replaced.
        """
        cfg = settings.KNOWLEDGE_AGENT
        provider, model, temperature = get_backend_llm_overrides(
            CRISIS_PROMPT_PATH, prompts
        )
        prompt = build_crisis_prompt(message, prompts=prompts)
        if not prompt:
            logger.error(
                "Crisis prompt template resolved empty; skipping classification"
            )
            return {
                "is_crisis": False,
                "signal": "",
                "confidence": 0.0,
                "failed": True,
            }

        try:
            parsed, meta = await generate_structured(
                schema=CrisisVerdict,
                prompt=prompt,
                task=LLMTask.WHATSAPP_CRISIS_CLASSIFY.value,
                provider=provider,
                model=model or cfg.CRISIS_MODEL,
                # Zero, not low: this is a classification, and any sampling variance
                # means the same message is a crisis on Tuesday and not on Wednesday.
                temperature=temperature if temperature is not None else 0.0,
            )
        # noqa: BLE001 below — never fail the question over the classifier.
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Crisis classification failed (%s); keyword rules still apply",
                type(e).__name__,
            )
            return {
                "is_crisis": False,
                "signal": "",
                "confidence": 0.0,
                "failed": True,
            }

        if parsed.is_crisis:
            # Info, not debug, and without the message body: a positive verdict is an
            # operationally significant event, and the signal phrase is enough to review
            # calibration without writing a worker's disclosure to the application log.
            logger.info(
                "Crisis classifier fired: confidence=%.2f signal=%r",
                parsed.confidence,
                parsed.signal[:120],
            )

        return {
            "is_crisis": bool(parsed.is_crisis),
            "signal": (parsed.signal or "").strip(),
            "confidence": float(parsed.confidence or 0.0),
            "failed": False,
            "provider": meta.get("provider", "") if isinstance(meta, dict) else "",
            "model": meta.get("model", "") if isinstance(meta, dict) else "",
        }

    # ------------------------------------------------------------------ retrieval

    @staticmethod
    def _select_passages(
        hits: List[Dict[str, Any]],
        *,
        band: float,
        max_passages: int,
        max_context_tokens: int,
    ) -> List[Dict[str, Any]]:
        """
        Narrow raw hits to the passages actually worth putting in the prompt.

        Three filters, in order:

        1. A similarity BAND relative to the best hit. One strong match plus seven weak
        ones
           is worse than the strong match alone — the weak ones give the model licence
           to blend unrelated material into a single confident-sounding answer.
        2. A hard count cap, because a 1600-character reply cannot honestly ground on
        more
           than a handful of passages.
        3. A context token budget, using the token_count ally-be already computed per
        chunk.
           No tokeniser is needed here, and none exists in this service.
        """
        if not hits:
            return []

        best = float(hits[0].get("similarity") or 0.0)
        banded = [h for h in hits if float(h.get("similarity") or 0.0) >= best - band]

        selected: List[Dict[str, Any]] = []
        budget = max_context_tokens
        for hit in banded[:max_passages]:
            # Fall back to a rough character estimate only when ally-be sent no count,
            # so a missing field cannot silently disable the budget.
            tokens = int(hit.get("token_count") or 0) or max(
                1, len(hit.get("text") or "") // 4
            )
            if selected and tokens > budget:
                break
            selected.append(hit)
            budget -= tokens
        return selected

    # ------------------------------------------------------------------ citations

    @staticmethod
    def _resolve_citations(
        numbers: List[int], passages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map the model's passage numbers back to real chunk metadata, dropping anything
        bogus.

        Validated in code rather than trusted, on the same principle as validate_chart()
        in the analytics agent: a citation pointing at passage 9 when 5 were supplied
        would render as a source line the reader cannot check, and an answer that cites
        something is read as more trustworthy than one that does not. Out-of-range
        numbers are dropped rather than clamped — clamping would silently attribute the
        claim to a real but wrong passage, which is worse than losing the citation.

        Order is preserved and duplicates collapsed, so the rendered source list matches
        the order the model reasoned in.
        """
        resolved: List[Dict[str, Any]] = []
        seen: set[int] = set()

        for number in numbers:
            try:
                index = int(number)
            except (TypeError, ValueError):
                continue
            if index < 1 or index > len(passages) or index in seen:
                continue
            seen.add(index)
            passage = passages[index - 1]
            resolved.append(
                {
                    "passage_number": index,
                    "chunk_id": passage.get("chunk_id"),
                    "document_id": passage.get("document_id"),
                    "document_title": passage.get("document_title") or "",
                    "page_from": int(passage.get("page_from") or 0),
                    "page_to": int(passage.get("page_to") or 0),
                    "section_path": passage.get("section_path") or "",
                    "source_url": passage.get("source_url") or "",
                    "similarity": float(passage.get("similarity") or 0.0),
                }
            )

        if len(resolved) < len(numbers):
            logger.warning(
                "Dropped %d invalid citation number(s) out of %d",
                len(numbers) - len(resolved),
                len(numbers),
            )
        return resolved

    # ------------------------------------------------------------------ main entry

    async def answer(
        self,
        question: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
        prompts: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        decline_similarity: Optional[float] = None,
        max_passages: Optional[int] = None,
        max_context_tokens: Optional[int] = None,
        similarity_band: Optional[float] = None,
        max_answer_chars: int = DEFAULT_MAX_ANSWER_CHARS,
        translate_query: bool = True,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Answer one question. Every threshold is overridable per request by ally-be,
        which reads them from the `whatsapp_bot` settings row, so retrieval can be tuned
        without a deploy.
        """
        cfg = settings.KNOWLEDGE_AGENT
        top_k = top_k or cfg.TOP_K
        min_similarity = (
            cfg.MIN_SIMILARITY if min_similarity is None else min_similarity
        )
        decline_similarity = (
            cfg.DECLINE_SIMILARITY if decline_similarity is None else decline_similarity
        )
        max_passages = max_passages or cfg.MAX_PASSAGES
        max_context_tokens = max_context_tokens or cfg.MAX_CONTEXT_TOKENS
        similarity_band = (
            cfg.SIMILARITY_BAND if similarity_band is None else similarity_band
        )

        search_text, language, translated = (question, "en", None)
        if translate_query:
            search_text, language, translated = await self.prepare_query(
                question, prompts=prompts
            )

        hits = await self.chunk_service.search(
            query=search_text,
            limit=top_k,
            min_similarity=min_similarity,
            document_ids=document_ids,
        )
        top_similarity = float(hits[0].get("similarity") or 0.0) if hits else 0.0

        retrieval: Dict[str, Any] = {
            "top_k": top_k,
            "min_similarity": min_similarity,
            "decline_similarity": decline_similarity,
            "hit_count": len(hits),
            "top_similarity": round(top_similarity, 4),
            "passages_used": 0,
            "query_language": language,
            "translated_query": translated,
            "unsupported": False,
        }

        # --- Gate 1: deterministic, no LLM call ---
        #
        # Answering this in code rather than asking the model is what makes the
        # threshold auditable and tunable, and it means the common "we simply don't have
        # this" case costs nothing in generation tokens.
        if not hits or top_similarity < decline_similarity:
            reason = (
                DeclineReason.NO_HITS if not hits else DeclineReason.BELOW_THRESHOLD
            )
            logger.info(
                "Declining before generation: reason=%s hits=%d top=%.4f",
                reason.value,
                len(hits),
                top_similarity,
            )
            return {
                "intent": AnswerIntent.DECLINE,
                "answer": "",
                "language": language,
                "confidence": 0.0,
                "citations": [],
                "decline_reason": reason,
                "retrieval": retrieval,
                "provider": "",
                "model": "",
                "prompt_version": cfg.PROMPT_VERSION,
            }

        passages = self._select_passages(
            hits,
            band=similarity_band,
            max_passages=max_passages,
            max_context_tokens=max_context_tokens,
        )
        retrieval["passages_used"] = len(passages)

        provider, model, temperature = get_backend_llm_overrides(
            ANSWER_PROMPT_PATH, prompts
        )
        prompt = build_answer_prompt(
            question,
            passages=passages,
            history=history,
            max_chars=max_answer_chars,
            prompts=prompts,
        )
        if not prompt:
            # A missing template must not become an unguided free-text answer.
            logger.error("Answer prompt template resolved empty")
            raise LLMInvocationFailedException(
                "The knowledge answer prompt template is missing or empty."
            )

        parsed, meta = await generate_structured(
            schema=KnowledgeAnswer,
            prompt=prompt,
            task=LLMTask.WHATSAPP_RAG_ANSWER.value,
            provider=provider,
            model=model,
            temperature=temperature if temperature is not None else 0.0,
        )

        intent = parsed.intent
        answer_text = (parsed.answer or "").strip()
        citations = self._resolve_citations(parsed.citations, passages)
        decline_reason = DeclineReason.NONE

        # --- Gate 2 post-validation ---
        if intent == AnswerIntent.ANSWER and not answer_text:
            # "I answered" with nothing in it reaches the worker as silence. Treat it as
            # the decline it actually is, so it also lands in the unanswered queue.
            logger.warning("Model returned intent=answer with an empty answer")
            intent = AnswerIntent.DECLINE

        if intent == AnswerIntent.DECLINE:
            decline_reason = DeclineReason.MODEL_DECLINED
            citations = []
        elif intent == AnswerIntent.CLARIFY:
            # A clarification is not grounded in anything, so it carries no citations —
            # and deliberately does NOT create an unanswered-question row: a vague
            # question is not evidence of a corpus gap.
            citations = []
        elif intent == AnswerIntent.ANSWER and not citations:
            # Kept, not discarded: a legitimate synthesis across passages sometimes
            # cites nothing. Flagged so the dashboard can count ungrounded answers,
            # which is the number that tells you whether the grounding instruction is
            # holding.
            retrieval["unsupported"] = True
            logger.info("Answer returned with no usable citations")

        return {
            "intent": intent,
            "answer": answer_text,
            "language": (parsed.language or language or "en").strip(),
            "confidence": float(parsed.confidence or 0.0),
            "citations": citations,
            "decline_reason": decline_reason,
            "retrieval": retrieval,
            "provider": meta.get("provider", ""),
            "model": meta.get("model", ""),
            "prompt_version": cfg.PROMPT_VERSION,
        }
