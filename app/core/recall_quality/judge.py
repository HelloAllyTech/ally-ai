"""Recall-quality judge — one Gemini call per turn.

Mirrors the drift, language, groundedness and RAG judges: the LLM returns a label only, every
rate is computed by the caller in SQL, and token usage is emitted so a run's cost is visible
rather than arriving as an invoice.

WHAT IT IS FOR. The voice agent's client recalls at most five backstory facts a turn, chosen by
a ranking with five weighted terms. Those weights have never been tunable from anything but
argument, because the scores were computed and discarded on every turn. With the pool now
recorded, the question worth asking is the one a score cannot answer: was the fact the turn
called for among the five, or sitting just below the cap?
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from app.core.config import settings
from app.core.recall_quality.prompt import build_judge_prompt
from app.core.recall_quality.schemas import RecallJudgment, RecallQualityOutput
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def judge_recall(
    *,
    counsellor_turn: str,
    client_reply: str,
    selected: Sequence[dict],
    passed_over: Sequence[dict],
    stance: Optional[str] = None,
    cue_tier: Optional[str] = None,
    rubric: Optional[str] = None,
) -> Tuple[Optional[RecallJudgment], str]:
    """Judge one turn's recall. Returns (judgment, model that actually ran).

    None means the call failed or returned nothing usable — never "no verdict was warranted",
    which is a real label (`no_demand`) and is stored as one. A caller that conflated the two
    would count its own outages as turns that needed nothing.

    The second value is THE MODEL THAT ACTUALLY RAN, which the caller stores rather than its
    own setting: a fallback recorded under the configured model would pollute a pinned series.
    """
    model_setting = settings.RECALL_QUALITY_JUDGE.MODEL

    # A turn with no pool at all was never a ranking decision, so there is nothing to audit.
    # Distinct from a turn whose pool held nothing apt, which IS a finding (`nothing_apt`).
    if not selected and not passed_over:
        return None, model_setting

    prompt = build_judge_prompt(
        counsellor_turn=counsellor_turn,
        client_reply=client_reply,
        selected=selected,
        passed_over=passed_over,
        stance=stance,
        cue_tier=cue_tier,
        rubric=rubric,
    )

    from app.core.llm.dispatch import PROVIDER_GEMINI, generate_structured
    from app.core.llm_usage.tasks import LLMTask

    output, meta = await generate_structured(
        schema=RecallQualityOutput,
        prompt=prompt,
        task=LLMTask.RECALL_QUALITY_JUDGE.value,
        provider=PROVIDER_GEMINI,
        model=model_setting,
        temperature=0,
        max_tokens=None,
    )

    if meta.get("fell_back_from"):
        # These rows carry a different judgeModel, so a trend that looks like a recall change
        # may just be this.
        logger.warning(
            "recall quality judge fell back from %s to %s/%s",
            meta["fell_back_from"],
            meta["provider"],
            meta["model"],
        )

    if output is None:
        logger.warning("[recall_quality] judge returned nothing for a turn")
        return None, meta["model"]

    judgment = output.judgment

    # `better_fact` must quote a fact we actually showed it. A hallucinated one would be stored
    # as "the ranking buried this", sending someone to retune a weight over a fact that never
    # existed — the same guard the RAG judge applies to chunk ids, for the same reason.
    if judgment.verdict == "missed_better":
        candidates = {(f.get("text") or "").strip() for f in passed_over}
        claimed = (judgment.better_fact or "").strip()
        if not claimed or claimed not in candidates:
            logger.warning(
                "[recall_quality] missed_better named a fact not in the passed-over list; "
                "recording the verdict without it"
            )
            judgment = judgment.model_copy(update={"better_fact": None})
    elif judgment.better_fact:
        # Only `missed_better` carries one. Anything else is the model filling a field.
        judgment = judgment.model_copy(update={"better_fact": None})

    # Same treatment for unused_selected: keep only facts that were actually recalled.
    if judgment.unused_selected:
        recalled = {(f.get("text") or "").strip() for f in selected}
        kept = [t for t in judgment.unused_selected if (t or "").strip() in recalled]
        if len(kept) != len(judgment.unused_selected):
            logger.warning(
                "[recall_quality] dropped %d unused_selected entries that were not recalled",
                len(judgment.unused_selected) - len(kept),
            )
        judgment = judgment.model_copy(update={"unused_selected": kept})

    return judgment, meta["model"]
