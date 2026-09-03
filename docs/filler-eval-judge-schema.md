# Filler Eval — Judge Schema & Rubric v1

**Status:** v1. Changes to the enums once judging is live = new `judge_prompt_version` (and re-judging for comparability).
**Companion docs:** `language-eval-judge-schema.md` (the judge this one is modelled on), and ally-ai-learn's `docs/thinking-fillers-perf-plan.md` (why the metrics exist).
**Implementation:** `ally-ai/app/core/filler_quality/{schemas,prompt,judge}.py` + endpoint `POST /api/v1/filler-quality/judge`. Rubric text lives in prompt management as `filler_quality_judge_rubric` (versioned; inline fallback in `prompt.py`).

---

## 1. Why this judge exists

The thinking filler is the short back-channel the AI client utters the instant the learner stops speaking, while its real reply is still being generated. It exists to mask latency, and ally-ai-learn already records per turn how fast it arrived, whether it played, and how much silence followed.

None of that says whether the filler was any **good**. And the gap is not neutral: because the filler is the character's first words, `response_latency_ms` is measured to it. A filler that arrives instantly but sounds nothing like the character — or answers the turn before last — makes the roleplay worse while improving every latency chart on the dashboard. That is the specific failure this judge exists to catch.

## 2. Design invariants (inherited from the drift and language judges — do not relitigate)

1. **Observations in, structured annotations out.** One call per session; per-filler output; no session-level free-text verdict.
2. **The LLM annotates; code computes.** The judge emits findings only. Rates, rollups and deltas are computed downstream by ally-be.
3. **No scalar quality scores.** Anywhere. Character fit and context fit surface as *finding rates*, not as mean ratings — an LLM's 1-5 score is poorly calibrated and averages into a dashboard number nobody can act on.
4. **Roles stated explicitly:** the AI plays the CLIENT (judged); the human trainee is the COUNSELOR.
5. **Most fillers have no findings.** "Hmm" is a correct, complete filler. A rubric that rewards finding fault marks a healthy session down, so the rubric spends most of its length on what is *not* a fault.
6. **Judge each filler on its own**, using only what preceded it.
7. Temperature 0, pinned model, structured output (`response_schema`), token usage emitted (`LLMTask.FILLER_JUDGE`).
8. `judge_version = (judge_model, judge_prompt_version)` echoed in every response and stamped on every stored row.

## 3. Enums (frozen for v1)

### 3.1 Dimensions

| Dimension | What it answers | Plan metric |
|---|---|---|
| `character_fit` | Does this sound like THIS character, in this moment? | M7 |
| `context_fit` | Does it fit what the learner just said? | M8 |
| `safety` | Could the real reply contradict it? | — |

### 3.2 Categories per dimension

| Dimension | Category | Meaning |
|---|---|---|
| `character_fit` | `generic_for_character` | Could belong to any character in any scenario. **Only** valid when a style is configured — see §4. |
| | `wrong_register` | Formality or warmth wrong for this character in this moment. |
| | `persona_break` | Breaks the character's voice or knowledge, or is in the wrong language/script. |
| `context_fit` | `answers_earlier_turn` | Responds to a previous moment rather than the turn it followed. |
| | `incongruous_reaction` | A reaction this turn does not support. |
| `safety` | `committed` | Asserts, answers, agrees, takes a side, names a specific, or presupposes an outcome. |
| | `echoes_specific` | Repeats a name, number or place back at the learner. |

Severity: `minor | major | critical`.

Category↔dimension consistency is validated in code; an invalid pairing is dropped and counted in `dropped_annotations`, never repaired by guessing.

## 4. Conditioning: style-configured vs not

`generic_for_character` is conditioned on whether the scenario actually configured a style for the character (authored speech samples). On a character who was never given a voice, a generic filler is a **configuration gap, not a model failure** — and the two need telling apart, or a push to configure more scenarios will look like a model regression.

The finding is therefore kept and flagged `conditioned_out=true` rather than dropped: it still says something true about the scenario. This mirrors how the language judge conditions understanding/adequacy errors on garbled STT.

## 5. What code computes, and why the LLM is not asked

| Computed | Why not asked |
|---|---|
| `repeated_within_window` | A fact about the sequence of played phrases. The judge sees each filler too narrowly to know, and a guessed repeat rate is worse than none. |
| `plays_since_last_use` | Same. |
| `distinct_phrase_ratio` | Arithmetic. |

**The repeat window is counted in PLAYS, not conversational turns.** One turn can play two fillers (the continuation), so a window expressed in turns would be roughly half as wide here as the player's own anti-repeat guard, and the judge's "repeated" would stop meaning the same thing as the player's.

## 6. Ownership

Identical to the language judge. ally-be selects which sessions to judge, builds the observations from its own transcript plus the per-turn filler metadata ally-ai-learn records (`fillerDecision`, `fillerClipSource`, …), and persists the rows. This service performs no database access and no aggregation.

An empty observation list returns an empty result rather than a 400: a session that played no fillers is the normal state of a fast session, and the caller needs to record it as judged-and-nothing-to-judge rather than retry it forever.

## 7. Downstream dependency

`LLMTask.FILLER_JUDGE` (`"filler_judge"`) needs the matching value in ally-be's `LlmTask` enum, or these rows arrive unlabelled on the cost-by-task dashboard. Deliberately its own label rather than folded into `LANGUAGE_JUDGE`: the two run on different cadences over different slices, and sharing a label makes both lines unreadable.

## 8. Tuning

There is no hand-labeled calibration set. Tune by spot-checking judged sessions; bump `FILLER_JUDGE.PROMPT_VERSION` whenever the rubric changes and re-judge affected slices so comparisons stay within one judge version. `dropped_annotations` rising is the signal that the rubric has started drifting from the typology.
