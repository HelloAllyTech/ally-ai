"""Knowledge agent endpoint — retrieve, then answer or decline.

Stateless. History and every threshold arrive on the request; nothing is stored here.
The raw retrieval half is a separate endpoint (/knowledge-chunks/search) so an admin can
tune retrieval without spending generation tokens, and so a retrieval problem can be
diagnosed without the prompt confounding it.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_knowledge_agent_service
from app.core.knowledge_agent.agent import KnowledgeAgentService
from app.exceptions.custom_exceptions import (
    EmbeddingFailedException,
    LLMInvocationFailedException,
    VectorDBSearchFailedException,
)
from app.schemas.knowledge_agent import (
    CrisisCheckRequest,
    CrisisCheckResponse,
    KnowledgeAnswerRequest,
    KnowledgeAnswerResponse,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/answer",
    response_model=KnowledgeAnswerResponse,
    status_code=status.HTTP_200_OK,
    tags=["knowledge_agent"],
)
async def answer_knowledge_question(
    payload: KnowledgeAnswerRequest,
    service: KnowledgeAgentService = Depends(get_knowledge_agent_service),
):
    """
    Answer one worker's question from the corpus, or decline.

    Three outcomes, all 200 — `intent` carries which: `answer` (grounded, with
    citations), `decline` (the corpus does not cover it), `clarify` (the question was
    too vague to retrieve against). A decline is a successful, correct result, not an
    error, so it must not be an error status: ally-be sends the worker a real reply in
    every one of these cases, and only a genuine failure should push it onto the
    fallback path.

    NOTE for the caller: this can take several seconds — up to two LLM calls plus an
    embedding. Pass an explicit timeout (ally-be uses 25s) rather than relying on the
    default, or a slow call will outlive the SQS visibility window and the question gets
    answered twice.
    """
    try:
        result = await service.answer(
            payload.question,
            history=[turn.model_dump() for turn in payload.history],
            prompts=payload.prompts,
            top_k=payload.top_k,
            min_similarity=payload.min_similarity,
            decline_similarity=payload.decline_similarity,
            max_passages=payload.max_passages,
            max_context_tokens=payload.max_context_tokens,
            similarity_band=payload.similarity_band,
            max_answer_chars=payload.max_answer_chars,
            translate_query=payload.translate_query,
            document_ids=(
                [str(d) for d in payload.document_ids] if payload.document_ids else None
            ),
        )
        return KnowledgeAnswerResponse(**result)

    except EmbeddingFailedException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to embed the question",
        )
    except VectorDBSearchFailedException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The knowledge index is unavailable",
        )
    except LLMInvocationFailedException as e:
        # Surfaced as 502 rather than swallowed into a decline: a model outage is NOT
        # the same as "the corpus does not cover this", and reporting it as one would
        # quietly fill the unanswered-question queue with questions the corpus answers
        # fine.
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    except Exception:
        logger.exception("Unexpected error answering a knowledge question")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to answer the question",
        )


@router.post(
    "/crisis-check",
    response_model=CrisisCheckResponse,
    status_code=status.HTTP_200_OK,
    tags=["knowledge_agent"],
)
async def crisis_check(
    payload: CrisisCheckRequest,
    service: KnowledgeAgentService = Depends(get_knowledge_agent_service),
):
    """
    Decide whether a message is about a crisis happening now.

    The second layer of the safety net. ally-be's keyword rules are the first and remain
    terminal; this catches what a keyword list structurally cannot, which is indirect
    disclosure — "I can't keep doing this" carries no keyword and is how someone whose
    job is spotting it in others tends to say it.

    Separate from `/answer` so ally-be can run both CONCURRENTLY: the safety check then
    costs no latency on the overwhelming majority of messages that are ordinary
    reference questions, and a positive verdict simply discards the answer that came
    back alongside it. The precedence rule — crisis always wins — lives in ally-be,
    which is the side that sends.

    Always 200, including when the classifier itself failed; `failed` carries that. A
    5xx here would push ally-be onto its generic error path and reply "something went
    wrong" to a message that may be a crisis, which is the worst available outcome. On
    failure the keyword rules are what still hold.
    """
    try:
        verdict = await service.classify_crisis(
            payload.message, prompts=payload.prompts
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Crisis check failed unexpectedly: %s", e)
        return CrisisCheckResponse(failed=True)

    return CrisisCheckResponse(**verdict)
