# Language Eval — Judge Schema & Rubric v1 (error typology freeze)

**Status:** v1 — the frozen contract for Phase 1. Changes to enums or severity weights once judging is live = new `judge_prompt_version` (and re-judging for comparability).
**Companion docs:** `language-capability-eval-implementation-plan.md`, `language-eval-metadata-inventory.md`.
**Implementation target:** `ally-ai/app/core/language_quality/{schemas,prompt,judge}.py` + endpoint `POST /api/v1/language-quality/judge` (clone of the drift-judge seam). Rubric text lives in prompt management as `language_quality_judge_rubric` (versioned; inline fallback in `prompt.py`).

---

## 1. Design invariants (inherited from the drift judge — do not relitigate)

1. **Whole transcript in, structured annotations out.** One call per session; per-turn output; no session-level free-text verdict.
2. **The LLM annotates; code computes.** The judge emits error annotations only. Weighted error rates, layer rollups, gating, and deltas are deterministic code.
3. **No scalar quality scores.** Anywhere. (FR14.)
4. **Roles stated explicitly:** the AI plays the CLIENT (judged); the human trainee is the COUNSELOR (input, subject to STT noise).
5. **No transcript echo** in output — evidence quotes are short spans; `user_text`/`ai_text` evidence rows are reconstructed at persistence time by ally-be.
6. **Judge each AI turn given only what preceded it** (anti-halo instruction + few-shots that discriminate within one transcript).
7. Temperature 0, pinned model, structured output (`response_schema`), token usage emitted (`LLMTask.LANGUAGE_JUDGE`).
8. `judge_version = (judge_model, judge_prompt_version)` echoed in every response and stamped on every stored row.

---

## 2. Enums (the frozen typology)

### 2.1 Layers (judged by this judge — text only)

| Layer | PRD dims | Notes |
|---|---|---|
| `comprehension` | 2 | Understanding of counselor input (conditioned — see §2.6). |
| `content` | 3, 4, 5 | What the persona said. |
| `appropriateness` | 10, 11, 12, 13, 14 | How it was said, socially/linguistically. |

(`realization` — dims 6–9 + accent — is NOT judged by this judge: round-trip WER + manual listening cover it.)

### 2.2 Dimensions

`understanding | adequacy | fluency | coherence | register | dialect_lexicon | colloquialness | persona_social | codeswitch`

### 2.3 Error categories per dimension (PRD Appendix A starter set, frozen for v1)

| Dimension | Categories | Category definitions (one line each, verbatim into the rubric) |
|---|---|---|
| `understanding` | `misinterpreted_intent` | Reply answers a different intent than the counselor's utterance expressed. |
| | `ignored_context` | Reply ignores information clearly established earlier in the conversation. |
| `adequacy` | `off_topic` | Content unrelated to the turn or scenario. |
| | `hallucination` | Asserts persona/backstory/world facts not in (and not reasonably implied by) the configured persona. |
| | `omission` | Fails to convey content the turn clearly required (see disclosure carve-out §3.4). |
| `fluency` | `grammar` | Morphological/syntactic error a native speaker would not make. |
| | `script_error` | Wrong script, broken glyphs/combining marks, transliteration where native script expected. |
| | `disfluency` | Unnatural repetition/fragmentation beyond configured fillers (see fillers carve-out §3.5). |
| | `truncation` | Utterance cut off mid-thought. |
| `coherence` | `contradiction` | Contradicts something the persona previously said/established. |
| | `non_sequitur` | No discernible logical connection to the conversation. |
| `register` | `too_formal_diglossia` | Literary/textbook variety where the colloquial spoken variety is expected (diglossic languages). |
| | `too_casual` | Below the socially expected register for the persona/relationship. |
| `dialect_lexicon` | `wrong_regional_variety` | Lexical items from outside the configured target variety. |
| `colloquialness` | `literal_translation_stilt` | Calqued, translated-sounding phrasing no native speaker would produce. |
| `persona_social` | `too_blunt` | Socially inappropriate directness given the emotional context. |
| | `persona_break` | Voice/knowledge/attitude inconsistent with the configured character (incl. assistant-speak). |
| `codeswitch` | `foreign_token_leak` | Unintended other-language tokens where the target language was expected. |
| | `unnatural_switch` | Code-switch at a boundary or of a kind a real bilingual speaker would not produce. |

*(v1 extension beyond the PRD starter set: `register:too_casual` only — everything else is verbatim Appendix A. Extension process: add a category → new judge_prompt_version; never rename or reuse an existing category name.)*

### 2.4 Severity (weights are code constants, not judge output semantics)

| Severity | Weight | Definition (anchors the judge) |
|---|---|---|
| `minor` | 1 | Noticeable to a native speaker; meaning and training value intact. |
| `major` | 5 | Degrades believability or meaning; a trainee would notice something is off. |
| `critical` | 10 | Breaks the simulation for this turn: meaning lost, persona shattered, or output unusable. |

### 2.5 Per-turn conditioning flag

`input_garbled: none | partial | severe` — quality of the *counselor's preceding utterance* as transcribed. Same semantics as drift's `GarbleLevel`. **Aggregation rule (code):** turns with `input_garbled != none` are excluded from the `understanding` and `adequacy` denominators (and their errors on those dimensions are stored but flagged `conditioned_out = true`); all other dimensions are judged regardless (a garbled input does not excuse bad grammar or wrong register).

### 2.6 `isolation_basis` (closed enum — keeps it aggregatable)

| Value | Meaning |
|---|---|
| `input_clean` | Counselor input this turn (and recent turns) clean → error is attributable to generation, not mishearing. |
| `input_garbled` | Error plausibly caused/excused by garbled input → conditioned out of comprehension/adequacy rates. |
| `persona_specified` | The configured persona/metadata explicitly specifies the violated expectation (e.g. colloquial directive present but ignored). |
| `persona_unspecified` | The metadata never asked for it (e.g. no register directive) → likely metadata fix, not model failure. |
| `pattern_systemic` | Same error class recurring across turns (not a one-off sampling artifact). |

`persona_specified` vs `persona_unspecified` is the judge-side half of the PRD's prompt-before-model decision rule: the orchestrator injects into the prompt whether the per-language style elements are populated (§4 request contract), so the judge can label which side of the rule applies.

---

## 3. Rubric v1 — judge prompt content (→ prompt-management entry `language_quality_judge_rubric`)

Assembled by `build_judge_prompt()` as: **rubric** (below) + **session parameter block** + **persona/scenario prompt** + **transcript** with `[turn N]` tags.

### 3.1 Session parameter block (filled from `languages.evalConfig` + scenario config by the orchestrator)

```
SESSION LANGUAGE: {language_label} ({language_code})
TARGET VARIETY: {target_variety}                # e.g. "colloquial spoken Chennai Tamil"
DIGLOSSIA APPLIES: {yes/no}                     # if no, too_formal_diglossia is off the table
CODE-SWITCH PARTNERS: {partners}                # e.g. "English (Tanglish)"; natural mixing with these is CORRECT
REGISTER DIRECTIVE CONFIGURED: {yes/no}         # scenario:language_characteristics populated for this language?
STYLE EXEMPLARS CONFIGURED: {yes/no}            # scenario:linguistic_style_samples populated?
ALLOWED FILLERS: {filler_list or "none configured"}
ENGINE: {SIMULATION | ROLEPLAY_V2}
LOCKED CONTENT EXISTS: {yes/no}                 # v2 disclosure ledger present
```

### 3.2 Core instructions (rubric body, abridged spec — full text authored at implementation)

- **Roles.** The AI plays the CLIENT (a distressed help-seeker persona). The human is the COUNSELOR (trainee). You judge ONLY the CLIENT's turns. The COUNSELOR's text is speech-to-text output and may be garbled — that is never the CLIENT's error, but you must rate each CLIENT turn's `input_garbled` from the immediately preceding COUNSELOR turn.
- **Task.** For each CLIENT turn, emit zero or more error annotations. **Most turns should have zero.** Do not manufacture findings; absence of errors is the expected common case. Never emit scores, grades, or overall judgments.
- **Judge turn-by-turn**, using only prior context; do not let a strong or weak session color individual turns.
- **Evidence.** Every annotation quotes the shortest span (≤ ~15 words) that exhibits the error, verbatim in the original script.
- **One annotation per distinct error.** The same span may carry two annotations only if two genuinely different dimensions fail (e.g. wrong variety AND ungrammatical).
- **Conditioning.** If the counselor's input was garbled, still annotate fluency/register/persona errors, set `input_garbled` accordingly, and use `isolation_basis=input_garbled` for any understanding/adequacy oddity plausibly caused by it.
- **Prompt-vs-model basis.** When annotating register/dialect/colloquialness/codeswitch errors, set `isolation_basis=persona_unspecified` if the parameter block says the relevant directive/exemplars are NOT configured; `persona_specified` if they are.

### 3.3 Carve-outs — NOT errors (the false-positive-critical list; each gets few-shot negatives)

1. **In-character distress**: rambling, hesitation, "I don't know", emotional repetition — realistic portrayal of a distressed client (drift's `in_character` lesson).
2. **Natural code-switching** with the configured partner language(s) — correct, expected behavior; only leakage into *other* languages or unnatural switch points are errors.
3. **Configured fillers/backchannels** (per the allowed-fillers list, or when filler injection is enabled) — never `disfluency`.
4. **Intentional withholding/deflection of locked content** (v2 disclosure ledger; also v1 gradual-disclosure design): a vague or deflecting answer about a secret is CORRECT persona behavior, never `omission` or `non_sequitur`. When `LOCKED CONTENT EXISTS: yes`, prefer no annotation over guessing.
5. **Counselor-led topic changes** — following the trainee somewhere new is not `off_topic`.
6. **Terse-but-valid replies** — brevity is not an error.
7. **Register mirroring**: matching a casual counselor's register is not `too_casual` unless it breaks persona.

### 3.4–3.5 (referenced above; expanded in the full rubric text with 2–3 few-shot examples per launch-language tier, including at least one garbled-STT negative, one in-character-distress negative, and one diglossia positive.)

---

## 4. API contract

### 4.1 Request — `POST /api/v1/language-quality/judge`

```jsonc
{
  "transcript": [ {"role": "client|counselor", "text": "...", "turn_index": 0}, ... ],
  "persona": "<scenarios.prompt or spec identity_core+scenario_context>",
  "language": "ta-IN",
  "language_eval_config": {              // from languages.evalConfig
    "language_label": "Tamil (India)",
    "target_variety": "colloquial spoken Tamil",
    "diglossia": true,
    "code_switch_partners": ["en"]
  },
  "scenario_style_config": {             // presence flags + fillers, from scenario config
    "register_directive_configured": true,
    "style_exemplars_configured": false,
    "allowed_fillers": ["அங்கனே", "..."],
    "engine": "SIMULATION",
    "locked_content_exists": false
  },
  "rubric": "<resolved prompt-management text; optional, falls back to inline default>"
}
```

### 4.2 Pydantic schema (draft — `schemas.py`)

```python
Layer = Literal["comprehension", "content", "appropriateness"]
Dimension = Literal["understanding", "adequacy", "fluency", "coherence",
                    "register", "dialect_lexicon", "colloquialness",
                    "persona_social", "codeswitch"]
ErrorCategory = Literal[
    "misinterpreted_intent", "ignored_context",
    "off_topic", "hallucination", "omission",
    "grammar", "script_error", "disfluency", "truncation",
    "contradiction", "non_sequitur",
    "too_formal_diglossia", "too_casual",
    "wrong_regional_variety",
    "literal_translation_stilt",
    "too_blunt", "persona_break",
    "foreign_token_leak", "unnatural_switch",
]
Severity = Literal["minor", "major", "critical"]
GarbleLevel = Literal["none", "partial", "severe"]          # same as drift
IsolationBasis = Literal["input_clean", "input_garbled", "persona_specified",
                         "persona_unspecified", "pattern_systemic"]

class ErrorAnnotation(BaseModel):
    dimension: Dimension
    category: ErrorCategory
    severity: Severity
    evidence_quote: str          # <= ~15 words, verbatim, original script
    isolation_basis: IsolationBasis
    reasoning: str               # one sentence

class TurnJudgment(BaseModel):
    turn_index: int
    input_garbled: GarbleLevel
    errors: list[ErrorAnnotation]        # usually empty

class JudgeOutput(BaseModel):            # exactly what the LLM returns
    per_turn: list[TurnJudgment]
```

`layer` is **derived in code** from `dimension` (fixed mapping §2.1) — never asked of the LLM (one less thing to get wrong). Code also validates category∈dimension (reject + retry on mismatch) and clamps `evidence_quote` length.

### 4.3 Response

```jsonc
{ "judge_model": "gemini-2.5-pro", "judge_prompt_version": 1,
  "result": { "per_turn": [...], "turns_judged": 14 } }
```

---

## 5. Aggregation formulas (deterministic, ally-be read-side — never stored, never from the LLM)

```
SEVERITY_WEIGHT = {minor: 1, major: 5, critical: 10}

# per dimension d, over an experiment slice:
eligible_turns(d) = all judged CLIENT turns
                    minus (turns with input_garbled != none, IF d ∈ {understanding, adequacy})

weighted_error_rate(d) = Σ_{errors e: e.dimension=d, not conditioned_out}
                            SEVERITY_WEIGHT[e.severity]
                          / Σ eligible_turns(d) × 100

layer_rate(L) = Σ_{d ∈ L} weighted_error_rate(d)          # displayed per-dimension AND rolled up
dominant_category(d) = argmax_category weighted count      # named on the dashboard
```

- Deltas computed vs the pinned reference experiment; only valid when `judge_version` matches (FR13/NFR3 — enforced in the query, surfaced as a warning otherwise).
- `n_turns` (eligible per dimension) displayed with every rate (NFR2).
- Gate interplay: script fidelity and round-trip WER live outside this judge; the dashboard's ladder composes them (plan Phase 4).

## 6. Storage keys (ally-be, from plan Phase 1)

- `language_judgment_sessions`: PK (scenario_session_id, judge_model, judge_prompt_version); `n_turns_judged`, `n_turns_garbled`, script-fidelity %, denormalized dims (language, scenarioId, scenarioVersionId, engine, llmModel, llmProvider, promptVersion, occurredAt).
- `language_error_annotations`: FK → session-judgment row; one row per error; all §4.2 fields + derived `layer` + `conditioned_out` + reconstructed `user_text`/`ai_text`; upsert key mirrors drift (`ON CONFLICT (scenario_session_id, turn_index, seq, judge_model, judge_prompt_version)`).

## 7. Judge reliability (PRD NFR1 — **no human annotation, no hand-labeled calibration set**)

The judge runs directly on session transcripts; there is no hand-labeling step. Reliability comes from:

1. **Pinning** — `judge_version = (judge_model, judge_prompt_version)` on every row; comparisons are only valid within one judge_version, so a metric change always traces to a system change, not a judge change (NFR3).
2. **Determinism hygiene** — temperature 0, structured output with schema validation + retry-on-mismatch.
3. **Deltas over levels** — the system's readouts are experiment deltas under one-variable changes, which are robust to constant judge bias in a way absolute levels are not.
4. **Confidence tiering, declared not gated** — `languages.evalConfig.judgeReliabilityTier` records how much to trust text-layer sociolinguistic judgments per language (lower-resource languages = lower-confidence, per PRD L4); the dashboard displays the tier, nothing blocks on it.
5. **Informal spot-checks where feasible** — reading judged sessions in the error-log drill-down (native speakers when available) is the tuning loop for the rubric's carve-outs (§3.3); rubric changes = new `judge_prompt_version` + re-judge affected slices.
