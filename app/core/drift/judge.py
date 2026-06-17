"""The conversation drift judge — one Gemini call per session.

Whole transcript in → per-turn structured array out (see drift-metrics-spec.md).
The session rollup (drifted / first-drift turn / attribution mix) is derived in
``compute_session_rollup`` deterministically from the per-turn rows.

The Gemini SDK and client are imported/constructed lazily so this module (and
``schemas``) can be imported without the ``google-genai`` dependency installed
or a key configured — only ``judge_session`` requires them.
"""

from __future__ import annotations

from typing import List, Optional

from app.core.config import settings
from app.core.drift.prompt import TranscriptTurn, build_judge_prompt
from app.core.drift.schemas import (
    COHERENCE_DRIFT_CUTOFF,
    COHERENCE_RANK,
    AttributionMix,
    DriftJudgmentResult,
    JudgeOutput,
    PerTurnJudgment,
    SessionRollup,
)

# Min consecutive drift turns to count the session as drifted (spec: K=2).
DRIFT_RUN_K = 2

_client = None


def _get_client():
    """Lazily build the Gemini client; clear error if the key is missing."""
    global _client
    if _client is None:
        if not settings.GEMINI.API_KEY:
            raise RuntimeError(
                "GEMINI__API_KEY is not configured — cannot run the drift judge."
            )
        from google import genai  # imported lazily; optional dependency

        _client = genai.Client(api_key=settings.GEMINI.API_KEY)
    return _client


def _is_drift_turn(t: PerTurnJudgment) -> bool:
    """A turn counts toward drift if it's topic-bad OR coherence-bad-and-not-in-character.

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


def judge_session(
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
    """
    from google.genai import types  # imported lazily; optional dependency

    prompt = build_judge_prompt(
        transcript, persona, language, scenario_goal, rubric=rubric
    )
    client = _get_client()
    response = client.models.generate_content(
        model=settings.DRIFT_JUDGE.MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=JudgeOutput,
        ),
    )
    output: Optional[JudgeOutput] = response.parsed
    if output is None or not output.per_turn:
        # Gemini occasionally returns no parsable content despite the schema;
        # fail this session loudly so the backfill loop logs + skips it rather
        # than crashing on a None deref.
        raise RuntimeError("drift judge returned no parsable output")
    rollup = compute_session_rollup(output.per_turn)
    return DriftJudgmentResult(per_turn=output.per_turn, session=rollup)
