"""The thinking-filler judge — one Gemini call per session.

The session's played fillers in → a judgement per filler out. Post-processing is
deterministic code: score validation (out-of-range or unmatched judgements are
dropped and counted, never guessed), repeat detection, and the acceptability
rule. The LLM is never asked for repeat facts, rates, or a session verdict.

The Gemini SDK and client are imported/constructed lazily so this module (and
``schemas``) can be imported without the ``google-genai`` dependency installed
or a key configured — only ``judge_session`` requires them.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from app.core.config import settings
from app.core.filler_quality.prompt import FillerStyleParams, build_judge_prompt
from app.core.filler_quality.schemas import (
    DEFAULT_REPEAT_WINDOW_PLAYS,
    DIMENSION_CATEGORIES,
    MAX_EVIDENCE_CHARS,
    STYLE_CONDITIONED_CATEGORIES,
    FillerJudgment,
    FillerJudgmentResult,
    FillerObservation,
    JudgeOutput,
    ProcessedFiller,
    ProcessedFinding,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

_client = None


def _get_client():
    """Lazily build the Gemini client; clear error if the key is missing."""
    global _client
    if _client is None:
        if not settings.GEMINI.API_KEY:
            raise RuntimeError(
                "GEMINI__API_KEY is not configured — cannot run the filler judge."
            )
        from google import genai  # imported lazily; optional dependency

        _client = genai.Client(api_key=settings.GEMINI.API_KEY)
    return _client


def _normalize(phrase: str) -> str:
    """Phrase identity for repeat detection: case- and whitespace-insensitive."""
    return " ".join((phrase or "").split()).strip().lower()


def compute_repeats(
    observations: List[FillerObservation],
    window_plays: int = DEFAULT_REPEAT_WINDOW_PLAYS,
) -> Dict[int, tuple]:
    """Work out, per observation, whether its phrase repeated recently.

    Returns ``{turn_index: (repeated_within_window, plays_since_last_use)}``.

    This is arithmetic over the sequence of played phrases, so it is computed
    rather than asked of the model: the LLM sees each filler in isolation and
    would have to guess, and a guessed repeat rate is worse than none. The
    window is counted in PLAYS, matching the player's own anti-repeat guard —
    one conversational turn can play two fillers, so a window expressed in turns
    would mean something different here than it does there.

    Observations are considered in the order given, which is the order they were
    played; ties on turn index (a turn with a continuation filler) keep their
    input order for the same reason.
    """
    out: Dict[int, tuple] = {}
    last_play_index: Dict[str, int] = {}
    for play_index, obs in enumerate(observations or []):
        key = _normalize(obs.filler_text)
        if not key:
            continue
        previous = last_play_index.get(key)
        if previous is None:
            out[obs.turn_index] = (False, None)
        else:
            distance = play_index - previous
            out[obs.turn_index] = (distance <= max(0, window_plays), distance)
        last_play_index[key] = play_index
    return out


def process_output(
    per_filler: List[FillerJudgment],
    observations: List[FillerObservation],
    window_plays: int = DEFAULT_REPEAT_WINDOW_PLAYS,
    style_configured: bool = True,
) -> FillerJudgmentResult:
    """Deterministic post-processing of the model's per-filler annotations.

    - drops findings whose category doesn't belong to their dimension, and
      annotations naming a turn we never sent (counted in
      ``dropped_annotations`` — a rubric-tuning signal, never guessed)
    - joins each annotation to its observation, so the stored row carries the
      phrase, source and type the model was not asked to echo back
    - marks ``conditioned_out`` on style-dependent findings when the scenario
      configured no style: calling a filler generic on a character who was
      never given a voice blames the model for a configuration gap. Kept and
      flagged rather than dropped, because it is real signal about the
      scenario — just not about the model.
    - computes the repeat facts in code
    - clamps evidence quotes to MAX_EVIDENCE_CHARS
    """
    by_turn = {obs.turn_index: obs for obs in observations or []}
    repeats = compute_repeats(observations, window_plays)

    processed: List[ProcessedFiller] = []
    dropped = 0
    for annotation in per_filler or []:
        obs = by_turn.get(annotation.turn_index)
        if obs is None:
            dropped += 1
            logger.warning(
                "filler judge: dropping annotation for unknown turn "
                f"{annotation.turn_index}"
            )
            continue

        findings: List[ProcessedFinding] = []
        for finding in annotation.findings:
            if finding.category not in DIMENSION_CATEGORIES.get(
                finding.dimension, set()
            ):
                dropped += 1
                logger.warning(
                    "filler judge: dropping invalid finding "
                    f"dimension={finding.dimension} category={finding.category} "
                    f"turn={annotation.turn_index}"
                )
                continue
            findings.append(
                ProcessedFinding(
                    **{
                        **finding.model_dump(),
                        "evidence_quote": finding.evidence_quote[:MAX_EVIDENCE_CHARS],
                    },
                    conditioned_out=(
                        finding.category in STYLE_CONDITIONED_CATEGORIES
                        and not style_configured
                    ),
                )
            )

        repeated, distance = repeats.get(annotation.turn_index, (False, None))
        processed.append(
            ProcessedFiller(
                turn_index=annotation.turn_index,
                filler_text=obs.filler_text,
                source=obs.source,
                filler_type=obs.filler_type,
                findings=findings,
                repeated_within_window=repeated,
                plays_since_last_use=distance,
            )
        )

    distinct_ratio = None
    played = [
        _normalize(obs.filler_text)
        for obs in (observations or [])
        if _normalize(obs.filler_text)
    ]
    if played:
        distinct_ratio = round(len(set(played)) / len(played), 4)

    return FillerJudgmentResult(
        per_filler=sorted(processed, key=lambda p: p.turn_index),
        fillers_judged=len(processed),
        dropped_annotations=dropped,
        distinct_phrase_ratio=distinct_ratio,
        repeat_window_plays=window_plays,
    )


def judge_session(
    observations: List[FillerObservation],
    persona: str,
    language: str,
    style_params: Optional[FillerStyleParams] = None,
    rubric: Optional[str] = None,
    window_plays: int = DEFAULT_REPEAT_WINDOW_PLAYS,
) -> FillerJudgmentResult:
    """Judge every filler played in one session.

    ``rubric`` is the static instruction block sourced from prompt management
    (FILLER_JUDGE_PROMPT_CODE); callers should fetch it once and pass it in.
    Falls back to the inline DEFAULT_FILLER_RUBRIC when None.

    A session that played no fillers is not an error and costs no LLM call — it
    is the normal state of a fast session, and the caller needs to be able to
    tell it apart from a session the judge failed on.
    """
    if not observations:
        return FillerJudgmentResult(repeat_window_plays=window_plays)

    from google.genai import types  # imported lazily; optional dependency

    prompt = build_judge_prompt(
        observations,
        persona,
        language,
        style_params=style_params,
        rubric=rubric,
    )
    client = _get_client()
    response = client.models.generate_content(
        model=settings.FILLER_JUDGE.MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=JudgeOutput,
        ),
    )
    # Best-effort token-usage emission for the cost-by-model/task dashboard.
    try:
        from app.core.llm_usage.emitter import emit_llm_usage_blocking
        from app.core.llm_usage.tasks import LLMTask

        um = getattr(response, "usage_metadata", None)
        if um is not None:
            prompt_tokens = int(getattr(um, "prompt_token_count", 0) or 0)
            completion_tokens = int(getattr(um, "candidates_token_count", 0) or 0)
            total_tokens = int(getattr(um, "total_token_count", 0) or 0) or (
                prompt_tokens + completion_tokens
            )
            emit_llm_usage_blocking(
                provider="gemini",
                model=settings.FILLER_JUDGE.MODEL,
                task=LLMTask.FILLER_JUDGE.value,
                usage=(prompt_tokens, completion_tokens, total_tokens),
            )
    except Exception:
        # Never fails the judge over cost telemetry; logged so a bug here
        # doesn't vanish with zero trace.
        logger.debug("filler judge usage emit skipped (best-effort)", exc_info=True)

    output: Optional[JudgeOutput] = response.parsed
    if output is None or not output.per_filler:
        # Fail loudly so the backfill loop logs + skips this session rather than
        # persisting an empty judgment as "every filler was fine".
        raise RuntimeError("filler judge returned no parsable output")
    return process_output(
        output.per_filler,
        observations,
        window_plays,
        style_configured=bool(style_params and style_params.style_exemplars),
    )
