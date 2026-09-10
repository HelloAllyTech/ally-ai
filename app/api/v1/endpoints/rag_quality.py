"""RAG-quality judge — a stateless transform.

Same contract as the drift, language and feedback-groundedness judges: ally-be
owns the data, selects which retrievals to judge, resolves each passage's text
from its own `kb_document_chunks`, and persists the labels. This service only
judges.

The response echoes the model and rubric version actually used, so the caller
stamps the stored rows with what ran rather than hard-coding our config.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.rag_quality.judge import judge_retrieval
from app.core.rag_quality.schemas import PassageJudgment, RetrievalJudgment
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class RagQualityRequest(BaseModel):
    """One retrieval to audit."""

    query: str
    # Which corpus, so the rubric can state what the material is FOR. Relevance
    # is not corpus-independent: a passage on how to write a speech sample
    # answers a craft question and is tangential to a subject question.
    corpus: str
    # Each: {chunk_id, document_title, section_path, similarity, text}. The text
    # comes from ally-be's own chunk rows — this service does not look it up,
    # because a re-chunk since the retrieval would give the judge a passage the
    # agent never saw.
    passages: List[dict] = Field(default_factory=list)
    # The floor in force for that retrieval, shown to the judge so a set of
    # barely-passing passages reads differently from one that cleared easily.
    min_similarity: float = 0.0
    rubric: Optional[str] = None


class RagQualityResponse(BaseModel):
    judge_model: str
    judge_prompt_version: str
    passages: List[PassageJudgment]
    # None only when the judge failed to produce a set-level verdict. An empty
    # retrieval is NOT this — it comes back as sufficiency `nothing_useful`,
    # which is a real finding, and a caller that conflated the two would count
    # its own outages as corpus gaps.
    retrieval: Optional[RetrievalJudgment] = None


@router.post("/judge", response_model=RagQualityResponse)
async def judge(req: RagQualityRequest) -> RagQualityResponse:
    """Judge one retrieval: each returned passage, then the set."""
    if not req.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="empty query"
        )

    # A retrieval that returned NOTHING is judged, not skipped. It is the most
    # informative row in the log: either the corpus lacks the material or the
    # floor was too tight, and the judge's `missing` field is what turns a count
    # of empty retrievals into a list of documents to go and find.
    try:
        passages, retrieval, judge_model = await judge_retrieval(
            req.query,
            req.corpus,
            req.passages,  # type: ignore[arg-type]
            req.min_similarity,
            rubric=req.rubric,
        )
    except Exception as exc:  # noqa: BLE001 - surface as 500, keep caller decoupled
        logger.exception("[rag_quality] judge failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="rag quality judge failed",
        ) from exc

    return RagQualityResponse(
        judge_model=judge_model,
        judge_prompt_version=settings.RAG_QUALITY_JUDGE.PROMPT_VERSION,
        passages=passages,
        retrieval=retrieval,
    )
