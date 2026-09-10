"""RAG-quality judge — one Gemini call per retrieval.

Mirrors the drift, language and feedback-groundedness judges: the LLM returns
per-passage labels only, every rate is computed by the caller in SQL, and token
usage is emitted so a backfill's cost is visible rather than arriving as an
invoice.

Judging a retrieval that returned NOTHING is deliberately supported and is not
a no-op. An empty result is the most informative row in the log — it is either
a corpus gap or a floor set too tight, and those need opposite fixes. The judge
cannot tell them apart on its own (it never sees what the floor rejected), but
it can say what the query needed, which is what turns a count of empty
retrievals into a list of documents to go and find.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from app.core.config import settings
from app.core.rag_quality.prompt import JudgedPassage, build_judge_prompt
from app.core.rag_quality.schemas import (
    PassageJudgment,
    RagQualityOutput,
    RetrievalJudgment,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def judge_retrieval(
    query: str,
    corpus: str,
    passages: List[JudgedPassage],
    min_similarity: float,
    rubric: Optional[str] = None,
) -> Tuple[List[PassageJudgment], Optional[RetrievalJudgment], str]:
    """Judge one retrieval: each passage against the query, then the set.

    Returns (passage judgments, set judgment, model that actually ran).

    The set judgment is None only when the judge failed or returned nothing
    usable — never as a way of expressing "no passages", which is a real
    verdict (`nothing_useful`) and is stored as one. A caller that treats None
    and `nothing_useful` alike would count its own outages as corpus gaps.

    The third value is THE MODEL THAT ACTUALLY RAN, which the caller must store
    rather than its own setting: a fallback recorded under the configured model
    would pollute a pinned series.
    """
    if not query.strip():
        return [], None, settings.RAG_QUALITY_JUDGE.MODEL

    prompt = build_judge_prompt(
        query, corpus, passages, min_similarity, rubric=rubric
    )

    from app.core.llm.dispatch import PROVIDER_GEMINI, generate_structured
    from app.core.llm_usage.tasks import LLMTask

    output, meta = await generate_structured(
        schema=RagQualityOutput,
        prompt=prompt,
        task=LLMTask.RAG_QUALITY_JUDGE.value,
        provider=PROVIDER_GEMINI,
        model=settings.RAG_QUALITY_JUDGE.MODEL,
        temperature=0,
        # Uncapped, like its sibling judges: the output length is a property of
        # how many passages came back, not a choice.
        max_tokens=None,
    )

    if meta.get("fell_back_from"):
        # These rows will carry a different judgeModel, so a trend that looks
        # like a retrieval change may just be this.
        logger.warning(
            "rag quality judge fell back from %s to %s/%s",
            meta["fell_back_from"],
            meta["provider"],
            meta["model"],
        )

    if output is None:
        logger.warning(
            "[rag_quality] judge returned nothing for %d passage(s)", len(passages)
        )
        return [], None, meta["model"]

    # Drop anything pointing at a passage we did not send. A hallucinated
    # chunk_id would otherwise be stored against a real passage's row and
    # silently misreport it — the same guard the groundedness judge applies to
    # claim_index, for the same reason.
    sent = {p["chunk_id"] for p in passages}
    kept: List[PassageJudgment] = []
    seen: set[str] = set()
    for judged in output.passages:
        if judged.chunk_id not in sent:
            logger.warning(
                "[rag_quality] dropping judgment for unknown chunk_id=%s",
                judged.chunk_id,
            )
            continue
        # A duplicate would double-count one passage in every rate computed
        # downstream. First label wins; the judge was asked for one each.
        if judged.chunk_id in seen:
            logger.warning(
                "[rag_quality] dropping duplicate judgment for chunk_id=%s",
                judged.chunk_id,
            )
            continue
        seen.add(judged.chunk_id)
        kept.append(judged)

    missing = sent - seen
    if missing:
        # Reported rather than backfilled with a guess: an unjudged passage
        # must stay unjudged, or a rate silently includes invented labels.
        logger.warning(
            "[rag_quality] judge skipped %d of %d passage(s)",
            len(missing),
            len(sent),
        )

    retrieval = output.retrieval
    if retrieval is not None and not (retrieval.missing or "").strip():
        # Blank means "nothing to name", and it has to arrive as null: the gap
        # question is read as `missing IS NOT NULL`, and an empty string stored
        # there answers "yes, a gap, unnamed" to every one of those queries.
        retrieval = retrieval.model_copy(update={"missing": None})

    return kept, retrieval, meta["model"]
