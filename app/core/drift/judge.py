"""The conversation drift judge — one LLM call per session.

Whole transcript in → per-turn structured array out (see drift-metrics-spec.md).
The session rollup (drifted / first-drift turn / attribution mix) is derived in
``compute_session_rollup`` deterministically from the per-turn rows.

Gemini is the selected model (``DRIFT_JUDGE__MODEL``) and stays so. What changed
is that this module no longer holds a Gemini client: the call goes through
``app.core.llm.dispatch.generate_structured``, which picks the SDK from the
resolved model and falls back to OpenAI when the selected provider cannot run.
The dispatch module is imported inside the functions rather than at module
scope, so this module and ``schemas`` can still be imported without any
provider SDK installed or key configured.
"""

from __future__ import annotations

from typing import List, Optional

from app.core.config import settings
from app.core.drift.prompt import (
    TranscriptTurn,
    build_judge_prompt,
    build_lean_labels_prompt,
)
from app.core.drift.schemas import (
    COHERENCE_DRIFT_CUTOFF,
    COHERENCE_RANK,
    AttributionMix,
    DriftJudgmentResult,
    LeanJudgeOutput,
    LeanTurnLabels,
    LiveJudgeOutput,
    PerTurnJudgment,
    SessionRollup,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Min consecutive drift turns to count the session as drifted (spec: K=2).
DRIFT_RUN_K = 2


def _is_drift_turn(t: PerTurnJudgment) -> bool:
    """A turn counts toward drift if it's topic-bad OR
    coherence-bad-and-not-in-character.

    Unifies the spec's two clauses (>=K consecutive off_topic/gibberish, OR
    >=K consecutive coherence<=degrading while not in_character) into a single
    per-turn predicate; a run of >=K such turns is a drift event.
    """
    topic_bad = t.topic_label in ("off_topic", "gibberish")
    cohere_bad = (
        COHERENCE_RANK.get(t.coherence, 4) <= COHERENCE_DRIFT_CUTOFF
        and not t.in_character
    )
    return topic_bad or cohere_bad


def compute_session_rollup(per_turn: List[PerTurnJudgment]) -> SessionRollup:
    """Derive drifted / first-drift-turn / attribution-mix from per-turn rows."""
    turns = sorted(per_turn, key=lambda t: t.turn_index)
    flags = [_is_drift_turn(t) for t in turns]

    # Find the first run of >= DRIFT_RUN_K consecutive drift turns.
    first_drift_turn: Optional[int] = None
    run_start = None
    run_len = 0
    drift_turn_indices: set[int] = set()
    for i, flag in enumerate(flags):
        if flag:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= DRIFT_RUN_K:
                # Mark every turn in this qualifying run as a drift turn.
                for j in range(run_start, i + 1):
                    drift_turn_indices.add(j)
                if first_drift_turn is None:
                    first_drift_turn = turns[run_start].turn_index
        else:
            run_len = 0
            run_start = None

    mix = AttributionMix()
    for i in sorted(drift_turn_indices):
        attr = turns[i].root_attribution
        if attr in ("stt_direct", "stt_cascade", "llm_direct", "context_lockin"):
            setattr(mix, attr, getattr(mix, attr) + 1)

    return SessionRollup(
        drifted=first_drift_turn is not None,
        first_drift_turn=first_drift_turn,
        attribution_mix=mix,
    )


async def judge_session(
    transcript: List[TranscriptTurn],
    persona: str,
    language: str,
    scenario_goal: Optional[str] = None,
    rubric: Optional[str] = None,
) -> DriftJudgmentResult:
    """Run the drift judge over one whole session transcript.

    `rubric` is the static instruction block sourced from prompt management
    (DRIFT_JUDGE_PROMPT_CODE); callers should fetch it once (e.g. via
    AllyCoreService.get_prompts_by_codes) and pass it in to avoid re-fetching
    per session. Falls back to the inline DEFAULT_JUDGE_RUBRIC when None.

    Returns the per-turn judgments plus the code-derived session rollup.

    Runs through ``generate_structured`` rather than holding its own Gemini
    client. Two things come with that, neither of which changes what a judgment
    says:

    * An OpenAI fallback if Gemini is unreachable. Safe here specifically
      because `judgeModel` is stored on every row and is part of the row's
      uniqueness key, so a judgment produced by a different model lands as its
      own series instead of contaminating the pinned one — the same mechanism a
      deliberate re-judge already relies on.
    * The call stops blocking the event loop. It was a synchronous SDK call
      inside an ``async def`` endpoint, so one judge held the whole worker for
      the duration.

    Gemini remains the SELECTED model (`DRIFT_JUDGE__MODEL`), and `max_tokens`
    stays uncapped as it was: this output is one element per turn, so a cap
    sized for a reply truncates a long session's array.
    """
    prompt = build_judge_prompt(
        transcript, persona, language, scenario_goal, rubric=rubric
    )

    from app.core.llm.dispatch import PROVIDER_GEMINI, generate_structured
    from app.core.llm_usage.tasks import LLMTask

    # Strict schema: the v2 labels are required, so the model answers them on
    # every turn instead of only where they fired.
    output, meta = await generate_structured(
        schema=LiveJudgeOutput,
        prompt=prompt,
        task=LLMTask.DRIFT_JUDGE.value,
        provider=PROVIDER_GEMINI,
        model=settings.DRIFT_JUDGE.MODEL,
        temperature=0,
        max_tokens=None,
    )
    if meta.get("fell_back_from"):
        # Worth a warning rather than silence: the rows about to be written
        # carry a different judgeModel, so a trend that looks like a behaviour
        # change may just be this.
        logger.warning(
            "drift judge fell back from %s to %s/%s",
            meta["fell_back_from"],
            meta["provider"],
            meta["model"],
        )

    if output is None or not output.per_turn:
        # A model occasionally returns no parsable content despite the schema;
        # fail this session loudly so the backfill loop logs + skips it rather
        # than crashing on a None deref.
        raise RuntimeError("drift judge returned no parsable output")
    rollup = compute_session_rollup(output.per_turn)
    return DriftJudgmentResult(per_turn=output.per_turn, session=rollup)


async def judge_session_labels_only(
    transcript: List[TranscriptTurn],
    persona: str,
    language: str,
    scenario_goal: Optional[str] = None,
    rubric: Optional[str] = None,
) -> List[LeanTurnLabels]:
    """Judge ONLY the v2 labels, for turns already judged under the old rubric.

    Same model, same rubric text, same temperature as :func:`judge_session` —
    the single difference is the response schema, which is what the cost is in.
    Measured on production sessions the full judge averages 2.4k prompt against
    3.8k completion, and completion is eight times the price, so constraining
    the response is worth roughly three quarters of the spend while re-sending
    the transcript costs almost nothing.

    No session rollup is computed: `drifted` / `first_drift_turn` are derived
    from the v1 labels this call deliberately does not re-emit, and the rows
    being topped up already carry them. Recomputing a rollup from a partial
    label set would overwrite a real answer with a blind one.
    """
    prompt = build_lean_labels_prompt(
        transcript, persona, language, scenario_goal, rubric=rubric
    )

    from app.core.llm.dispatch import PROVIDER_GEMINI, generate_structured
    from app.core.llm_usage.tasks import LLMTask

    # Its own task label: the whole point is to compare its cost against
    # DRIFT_JUDGE, which is impossible if both land in one bucket.
    output, meta = await generate_structured(
        schema=LeanJudgeOutput,
        prompt=prompt,
        task=LLMTask.DRIFT_JUDGE_LABELS.value,
        provider=PROVIDER_GEMINI,
        model=settings.DRIFT_JUDGE.MODEL,
        temperature=0,
        max_tokens=None,
    )
    if meta.get("fell_back_from"):
        logger.warning(
            "lean drift judge fell back from %s to %s/%s",
            meta["fell_back_from"],
            meta["provider"],
            meta["model"],
        )

    if output is None or not output.per_turn:
        raise RuntimeError("lean drift judge returned no parsable output")
    return output.per_turn
