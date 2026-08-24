"""The language-quality judge — one Gemini call per session.

Whole transcript in → per-turn error annotations out (see
language-eval-judge-schema.md). Post-processing is deterministic code:
category↔dimension validation (invalid annotations dropped, counted), layer
derivation from dimension, and STT conditioning (``conditioned_out``). The LLM
is never asked for layers, rates, or session verdicts.

The Gemini SDK and client are imported/constructed lazily so this module (and
``schemas``) can be imported without the ``google-genai`` dependency installed
or a key configured — only ``judge_session`` requires them.
"""

from __future__ import annotations

from typing import List, Optional, Set

from app.core.config import settings
from app.core.language_quality.prompt import (
    LanguageEvalParams,
    ScenarioStyleParams,
    TranscriptTurn,
    build_judge_prompt,
)
from app.core.language_quality.schemas import (
    CONDITIONED_DIMENSIONS,
    DIMENSION_CATEGORIES,
    DIMENSION_LAYER,
    INTERRUPTION_CONDITIONED_CATEGORIES,
    MAX_EVIDENCE_CHARS,
    JudgeOutput,
    LanguageJudgmentResult,
    ProcessedError,
    ProcessedTurn,
    TurnJudgment,
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
                "GEMINI__API_KEY is not configured — cannot run the language judge."
            )
        from google import genai  # imported lazily; optional dependency

        _client = genai.Client(api_key=settings.GEMINI.API_KEY)
    return _client


def _interrupted_turns(transcript: List[TranscriptTurn]) -> Set[int]:
    """Turn indices the learner barged in on, read off the transcript.

    The caller marks client turns with ``interrupted: true``; a turn without the
    key is not evidence of anything either way, so it simply does not join the
    set. Tolerant of dicts and objects because the transcript crosses a service
    boundary as plain JSON.
    """
    out: Set[int] = set()
    for turn in transcript or []:
        if isinstance(turn, dict):
            interrupted = turn.get("interrupted")
            idx = turn.get("turn_index")
        else:
            interrupted = getattr(turn, "interrupted", None)
            idx = getattr(turn, "turn_index", None)
        if not interrupted:
            continue
        if isinstance(idx, int):
            out.add(idx)
    return out


def process_output(
    per_turn: List[TurnJudgment],
    interrupted_turns: Optional[Set[int]] = None,
) -> LanguageJudgmentResult:
    """Deterministic post-processing of the LLM's per-turn annotations.

    - drops annotations whose category doesn't belong to their dimension
      (counted in ``dropped_annotations`` — a rubric-tuning signal, never guessed)
    - derives ``layer`` from ``dimension`` (fixed mapping)
    - marks ``conditioned_out`` on understanding/adequacy errors of turns whose
      counselor input was garbled (PRD conditioning rule), and on `truncation`
      of turns the learner talked over (see the schemas module for why the two
      causes are scoped differently)
    - clamps evidence quotes to MAX_EVIDENCE_CHARS

    ``interrupted_turns`` carries turn indices the learner barged in on. It is
    runtime metadata the LLM cannot see, so it arrives as INPUT and is joined to
    the model's output by turn index here. Absent (the pre-2026-08-17 sessions,
    where the flag was never written) nothing is conditioned on it — an unknown
    must not silently read as "not interrupted".
    """
    interrupted = interrupted_turns or set()
    turns = sorted(per_turn, key=lambda t: t.turn_index)
    processed: List[ProcessedTurn] = []
    dropped = 0
    for turn in turns:
        errors: List[ProcessedError] = []
        for err in turn.errors:
            if err.category not in DIMENSION_CATEGORIES.get(err.dimension, set()):
                dropped += 1
                logger.warning(
                    "language judge: dropping invalid annotation "
                    f"dimension={err.dimension} category={err.category} "
                    f"turn={turn.turn_index}"
                )
                continue
            errors.append(
                ProcessedError(
                    **{
                        **err.model_dump(),
                        "evidence_quote": err.evidence_quote[:MAX_EVIDENCE_CHARS],
                    },
                    layer=DIMENSION_LAYER[err.dimension],
                    conditioned_out=(
                        (
                            err.dimension in CONDITIONED_DIMENSIONS
                            and turn.input_garbled != "none"
                        )
                        or (
                            err.category in INTERRUPTION_CONDITIONED_CATEGORIES
                            and turn.turn_index in interrupted
                        )
                    ),
                )
            )
        processed.append(
            ProcessedTurn(
                turn_index=turn.turn_index,
                input_garbled=turn.input_garbled,
                errors=errors,
            )
        )
    return LanguageJudgmentResult(
        per_turn=processed, turns_judged=len(processed), dropped_annotations=dropped
    )


def judge_session(
    transcript: List[TranscriptTurn],
    persona: str,
    language: str,
    language_params: Optional[LanguageEvalParams] = None,
    style_params: Optional[ScenarioStyleParams] = None,
    rubric: Optional[str] = None,
) -> LanguageJudgmentResult:
    """Run the language-quality judge over one whole session transcript.

    `rubric` is the static instruction block sourced from prompt management
    (LANGUAGE_JUDGE_PROMPT_CODE); callers should fetch it once and pass it in.
    Falls back to the inline DEFAULT_JUDGE_RUBRIC when None.
    """
    from google.genai import types  # imported lazily; optional dependency

    prompt = build_judge_prompt(
        transcript,
        persona,
        language,
        language_params=language_params,
        style_params=style_params,
        rubric=rubric,
    )
    client = _get_client()
    response = client.models.generate_content(
        model=settings.LANGUAGE_JUDGE.MODEL,
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
                model=settings.LANGUAGE_JUDGE.MODEL,
                task=LLMTask.LANGUAGE_JUDGE.value,
                usage=(prompt_tokens, completion_tokens, total_tokens),
            )
    except Exception:
        # Never fails the judge over cost telemetry; logged so a bug here
        # (as opposed to the emitter's own send failures, which it logs
        # itself) doesn't vanish with zero trace.
        logger.debug("language judge usage emit skipped (best-effort)", exc_info=True)

    output: Optional[JudgeOutput] = response.parsed
    if output is None or not output.per_turn:
        # Fail loudly so the backfill loop logs + skips this session rather
        # than persisting an empty judgment as "no errors".
        raise RuntimeError("language judge returned no parsable output")
    return process_output(output.per_turn, _interrupted_turns(transcript))
