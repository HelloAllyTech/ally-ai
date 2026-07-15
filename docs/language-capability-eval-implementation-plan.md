# Language-Capability Evaluation & RCA System — Implementation Plan

**Source PRD:** Language-Capability Evaluation & RCA System for Voice Roleplay Simulations
**Date:** 2026-07-08
**Repos touched:** `ally-ai` (offline eval service), `ally-be` (NestJS, owns Postgres), `ally-web` (admin dashboard), `ally-ai-learn` (live voice agent — read-mostly, one optional TTS-synthesis surface)

---

## Part 1 — PRD vs. codebase: what already exists

The PRD assumes a mostly-greenfield build. It isn't. The shipped **conversation-drift judge** (per `drift-metrics-spec.md`, fully landed) is architecturally ~60% of what the PRD asks for. The correct framing for this project is: **generalize the drift-judge architecture from one construct (drift) to the PRD's 4-layer / 14-dimension language-quality taxonomy, and add the two objective metrics + the human audio track.**

### 1.1 Service topology (corrected naming — the PRD's mental model needs this)

| Service | Role | Relevant assets |
|---|---|---|
| **ally-ai-learn** ("learn-core") | The live LiveKit voice agent (STT→LLM→TTS). v1 engine (`app/core/graph/` + `app/core/scenario/`) and v2 (`app/roleplay_v2/`) | `PromptData` + `ScenarioSpec` (the prompt-metadata inventory), TTS factory (elevenlabs/deepgram/sarvam/google/hume), per-language `LanguageConfig`, rehearsal harness, V2V tester, turn-metrics emitter |
| **ally-ai** ("lifeline-ai") | Offline AI backend: transcription pipeline + **stateless drift judge** | `app/core/drift/{schemas,prompt,judge}.py`, `POST /api/v1/drift/judge`, Gemini 2.5 Pro structured output, batch STT services (sarvam/deepgram/openai), language detector (Unicode-script based) |
| **ally-be** | Owns all Postgres + S3; orchestrates judging | `turn_drift_judgment` table, `drift-judge.service/repository`, `drift-analytics.repository`, session/transcript/turn-metrics entities, prompt & scenario versioning, session recordings (S3), roleplay-session-logs endpoints |
| **ally-web** (admin, :8081) | Dashboards | Analytics tab registry + `chartKit.tsx`, `ConversationDrift.tsx`, **RoleplaySessionLogs** list + per-session detail (transcript, audio player, models, latency, actor eval) |

### 1.2 PRD requirement → existing asset map

| PRD requirement | Status | Where |
|---|---|---|
| FR1 ingest transcript + config + audio | ✅ exists | `scenario_session_messages` (`senderId=-1` = AI turn, `startSeconds`), `scenario_session_turn_metrics`, `scenario_session_recordings` (S3, presigned on read) |
| FR3 LLM-judge categorized annotations | 🟡 pattern exists, wrong taxonomy | Drift judge emits per-turn categorized labels (coherence, failure mode, root attribution) — but not the PRD's 14-dimension / severity-weighted error typology |
| FR15 experiment-config capture | ✅ landed | `promptVersions` + `scenarioVersionId` on `scenario_sessions`, `llmProvider`/`llmModel` + gen params on `turn_metrics` (drift-spec Tier-1 shipped) |
| FR13/16 experiment slicing + deltas | 🟡 partial | Drift dashboard "By experiment" selector (promptVersion/model/sttModel/scenarioVersion); no pinned-reference/delta readout yet |
| FR12 error log / drill-down | 🟡 partial | `RoleplaySessionLogDetail` shows transcript + audio + models; per-turn judge annotations not yet surfaced there |
| FR2 round-trip WER + script fidelity | ❌ new | Nothing exists. All ingredients present (TTS factory in learn-core, batch STT in ally-ai, S3) |
| FR4 human audio testers (CMOS/win-rate) | ❌ new | Nothing exists |
| FR17/18 metadata inventory + sub-element experiments | 🟡 raw material exists | `PromptData` (v1) / `ScenarioSpec` (v2) are the inventory; `scenario_versions.config` (full jsonb per version, `parentVersionId`) enables **diff-derived** `changed_from_prev` |
| FR19 per-language declarative config | 🟡 partial | `languages` entity has `sttProviderConfig`; needs script/CER/ASR/code-switch/diglossia/judge-tier fields |
| FR14 no 1–5 scores | ⚠️ conflict | Existing evaluators (scenario report judge) emit 0–100 scalar metrics incl. "Colloquialism". The new system must not reuse them; they stay for the trainee-facing product |

### 1.3 Discrepancies between PRD and reality — RESOLVED (2026-07-08)

1. **"Prompt metadata" is two product surfaces, not one.** *(Resolved — adopt two-surface inventory; fix versioning gap.)* In the product it is split across: (a) **scenario config** — `scenarios.prompt` plus character fields, behaviors/states, and the per-language levers in `scenarios.metadata` (`languageCharacteristics`, `linguisticStyleSamples`, `allowedFillerWords`, keyed by languageId — verified captured in `scenario_versions.config` via `buildConfigFromScenario`'s `...metadata` spread); and (b) **prompt templates** — `prompts`/`prompts_versions`, captured per session in `metadata.promptVersions`. The FR17 inventory and "implicates" vocabulary use `surface:element` naming.
   - **PRD correction (verified):** "backstory" and "role instructions" are **one field** — `scenarios.prompt` is sent to learn-core as `roleInstructions` verbatim (scenario-shared.service.ts:394) and doubles as the judges' `persona`. They cannot be varied independently; the inventory treats them as a single element. A studio schema split is explicitly out of scope.
   - **Gap found + decision: FIX in Phase 3.** `scenario_translations` content (per-language opening statements, translated title/description) is NOT included in `buildConfigFromScenario` — changes there are invisible to version diffs (non-attributable, non-reconstructable). Fix: include translation fields in the snapshot. Side effect (accepted): publishing a version then faithfully restores its translations — fixing a latent product round-trip bug.
2. **Two engines.** *(Resolved — cover both.)* `engine` is a slice dimension; FR17 inventory is per-engine; `changed_from_prev` diffs `scenario_versions.config` (v1) / `roleplay_spec_versions` (v2).
3. **STT scope.** *(Resolved — self-contained garble flag.)* The language judge itself emits an `input_garbled` flag per turn; aggregation **excludes garbled-input turns from the Comprehension/Adequacy denominators** (the PRD's conditioning rule, implemented). No pipeline dependency on `turn_drift_judgment`; drift's independent garble detection is used only as an offline cross-check (disagreement = rubric-tuning signal).
4. **Session TTS audio is not stored per-utterance.** *(Resolved — re-synthesis; no new capture.)* Round-trip WER re-synthesizes from the LLM's own text using the session's captured TTS config (voiceId on session metadata; provider/model already surfaced in session logs). Per-turn TTS audio capture rejected (cost, not needed by the PRD formula). The mixed egress recording remains the manual-listening surface. Caveat: re-synthesis measures the TTS system's *current* behavior — correct for an experiment scoreboard, but not forensic playback of a historical session.
5. **Existing scalar judges.** *(Resolved — keep, review later.)* The scenario-report evaluator's 0–100 Colloquialism/Context-Appropriateness scalars remain as trainee/product-facing features; the new eval dashboard never displays them (FR14 holds within the eval system). Revisit deprecating the overlapping scalars once the categorized system is established.

### 1.4 FR17 prompt-metadata inventory — starting answer (to be finalized as Phase 0 deliverable)

From `ally-ai-learn/app/core/scenario/base.py` (`PromptData`) and `app/roleplay_v2/spec/models.py` (`ScenarioSpec`):

| Element (v1 name) | Independently variable? | Plausibly drives (PRD dims) |
|---|---|---|
| `character profile` (name/age/profession/context/character_profile_text) | ✅ scenario field | adequacy (hallucinated backstory), persona (13) |
| `role_instructions` | ✅ scenario field | register (10), colloquialness (12), persona (13) |
| `language_characteristics` (per-language free-text style) | ✅ per-language scenario field | register (10), dialect-lexicon (11), colloquialness (12), code-switching (14) |
| `language_dialogue_samples` (few-shot, per-language) | ✅ | register (10), colloquialness (12), code-switching (14) |
| `allowed_fillers` | ✅ | colloquialness (12), naturalness |
| `opening_statements` / `agent_dialogues` | ✅ | register, colloquialness |
| behaviors/states (`helpful/unhelpful_behaviours`, `states`) | ✅ | coherence (5), persona (13) |
| main prompt template (`selected_main_prompt_code` → `prompts_versions`) | ✅ separately versioned | all Content + Appropriateness dims (it's the skeleton the above are injected into) |
| `temperature` / llm_config | ✅ | fluency (4), coherence (5) |
| v2: `persona.identity_core` / `chunks`, `state_machine`, `disclosure_ledger`, `rubric.behaviors` | ✅ spec-versioned | persona, coherence, adequacy |

Notable: the product **already has per-language register/style levers** (`language_characteristics`, `language_dialogue_samples`) — the PRD's "prompt never asked for it vs model can't do it" decision rule maps directly onto whether these fields are populated for a language.

---

## Part 2 — Implementation plan

### Architecture principle
Reuse the drift seam exactly (documented as the "extending the judge" recipe in `drift-metrics-spec.md`): **ally-ai = stateless judge/metric transform; ally-be = selection, persistence, aggregation, endpoints; ally-web = chartKit tab + drill-down.** New capabilities are siblings of drift, not modifications of it.

### Phase 0 — Inventory & design freeze — ✅ DONE (2026-07-08)
1. **FR17 deliverable:** `language-eval-metadata-inventory.md` — canonical `surface:element` IDs across scenario config (v1), prompt templates, and v2 spec; element→dimension attribution map (the dashboard's "implicates" vocabulary); operationalized register decision rule; known limitations.
2. **Error-typology freeze:** `language-eval-judge-schema.md` — frozen enums (9 dimensions, 19 categories, severity weights 1/5/10, `input_garbled`, closed `isolation_basis`), draft Pydantic schema, rubric v1 spec with carve-outs (incl. the v2 disclosure-ledger omission carve-out), API contract, aggregation formulas, judge-reliability approach (no hand-labeling).
3. **Decisions:** all resolved — see Part 3.

### Phase 1 — Language-quality judge (text layers: dims 2–5, 10–14) — ✅ CORE LANDED (2026-07-10, uncommitted)

Shipped: ally-ai `app/core/language_quality/` (schemas/prompt/judge + `POST /api/v1/language-quality/judge`, 13 unit tests); ally-be `language_judgment_sessions` + `language_error_annotations` (migrations run), seeded `language_quality_judge_rubric` prompt, `LanguageJudgeRepository`/`LanguageJudgeService` (transcript builder reused from drift), shared `countableSessionPredicate` (preview+seed exclusion), backfill endpoints + 30-min catch-up scheduler, `language_judge` LLM task in both repos. Remaining in later phases: read-side endpoints/UI (Phase 4), script fidelity + round-trip WER (Phase 2), per-language evalConfig (Phase 3).


**ally-ai — new module `app/core/language_quality/`** (clone of `app/core/drift/`):
- `schemas.py`: `ErrorAnnotation{turn_index, layer, dimension, category, severity(minor|major|critical), evidence_quote, isolation_basis, reasoning}` + per-turn `input_garbled: none|partial|severe` (the STT-conditioning flag — §1.3.3); judge output = `{per_turn: [{turn_index, input_garbled, errors: [ErrorAnnotation]}]}`. LLM emits only annotations; **all rates computed in code** (drift pattern). Garbled-input turns are excluded from Comprehension/Adequacy denominators at aggregation time.
- `prompt.py`: rubric fetched from prompt management (`language_quality_judge_rubric` prompt code, versioned) + per-language parameter block (target variety, diglossia yes/no, code-switch partners, register expectations) injected from the request. Roles stated explicitly (AI=CLIENT / human=COUNSELOR — reuse drift's hard-won framing). Conditioning instruction: treat the counselor transcript as-heard; do not re-attribute STT (cross-reference drift for that).
- `judge.py`: Gemini structured output, temperature 0, pinned model; token-usage emission (`LLMTask.LANGUAGE_JUDGE`).
- Endpoint `POST /api/v1/language-quality/judge` — request `{transcript, persona, language, language_eval_config, rubric}`; response echoes `judge_model` + `judge_prompt_version`.
- **Separate call from the drift judge** (recommended): independent rubric versioning and tuning; ~doubles judge cost per session (~1–2¢ on 2.5 Pro; mitigate via sampling + cheaper tier once the rubric stabilizes). Revisit merging only if cost bites.

**ally-be — orchestration + storage** (clone of drift-judge service/repo):
- `language_judgment_sessions` table — **one row per (session, judge_model, judge_prompt_version)**: `n_turns_judged`, script-fidelity result (Phase 2), gate flags, denormalized dims (`language, scenarioId, scenarioVersionId, engine, llmModel, llmProvider, promptVersion, occurredAt`). This is the **denominator** — zero-error sessions must count.
- `language_error_annotations` table — **one row per error** (a turn can have many): FK to the session-judgment row + all `ErrorAnnotation` fields + the same denormalized dims + `user_text`/`ai_text` evidence reconstructed at write time (never echoed by the LLM — drift cost lesson).
- `language-judge.service.ts` / `language-judge.repository.ts`: reuse `buildTranscript()` selection logic (extract shared helper from drift repo), Redis job state, backfill endpoint + scheduled registration — all straight clones.
- Weighted error rate per 100 turns = Σ(count × weight{1,5,10}) / Σ n_turns × 100, computed in the analytics repository, never stored.

**Coverage & validation (NFR1) — decisions 2026-07-08:** judge **all languages sessions actually run in** from day one (language is a slice dimension, never an inclusion gate — same rule as drift). **No human annotation / no hand-labeled calibration set** — the judge runs directly on session transcripts. Reliability comes from pinning (`judge_version` on every row; comparisons valid only within one version), temperature-0 structured output, reading deltas rather than levels, and informal spot-checks via the error-log drill-down. `judgeReliabilityTier` in language config is a **declared confidence label** (sociolinguistic judgments in lower-resource languages shown as lower-confidence per PRD L4), not a gated promotion. Rubric tuning = new `judge_prompt_version` + re-judge affected slices.

### Phase 2 — Objective metrics (FR2) — ✅ LANDED (2026-07-10; verified: local en sessions scored 1.2–5.3% round-trip WER via OpenAI TTS+Whisper, script fidelity 100%)

Shipped: script fidelity (Unicode target-script check in ally-be at judge time, Latin tolerated for code-switching) and round-trip WER (`POST /api/v1/round-trip-wer` in ally-ai — sarvam for Indic / openai fallback, hand-rolled WER/CER safe for Brahmic combining marks; ally-be samples 5 longest turns per session, best-effort). Both persisted per session, averaged into the dashboard's objective panel, gate drives ladder masking.


**Script fidelity (trivial, do first):** pure code in ally-be — Unicode-block check per AI message against the language's configured script; % clean turns stored on `language_judgment_sessions`. No LLM.

**Round-trip WER/CER:**
- Home: **ally-ai** (recommended) — batch STT-from-URL with provider fallback already exists there; it's the offline-eval service. Add a thin TTS-synthesis client for the providers used in prod (start with the 1–2 TTS providers actually configured for launch languages; learn-core's `app/tts/factory.py` is the reference implementation).
- New endpoint `POST /api/v1/round-trip-wer`: `{utterances: [{text, turn_index}], tts_config{provider, voice, language}, asr_config{provider, model, language}, unit: wer|cer}` → per-utterance `{wer, ref, hyp}` + aggregate. Compute with `jiwer`; normalize text per language (strip punctuation, normalize Unicode) before scoring.
- ally-be orchestrates: **sample** N AI utterances per experiment cell (not every turn — cost), call ally-ai, persist to `round_trip_wer_results` (per-utterance) with the same denormalized dims. Thresholds from PRD (≤20 good / 20–30 warn / >30 critical) applied at read time; the session/experiment-level gate flag written to `language_judgment_sessions` (masks the human-audio layer in the dashboard).
- Per-language precondition honored via language config: if no reliable ASR is declared for a language, the metric is marked unavailable (dashboard shows "leaning on human testers"), not silently wrong.

### Phase 3 — Experiment tracking & per-language config (FR15/16/18/19) — ✅ LANDED (2026-07-10)

Shipped: `languages.evalConfig` (seeded for 13 languages; feeds judge params, fidelity script, WER/CER unit); `eval_experiments` + pinned reference (POST/GET `language-quality/reference`, per-dimension deltas in the response + delta chips in the UI); `changed_from_prev` via scenario-version config diff vs parent (shown under the scenario-version experiment chart, >1 element flagged); translations-versioning fix in `buildConfigFromScenario`. Backfill accepts `rejudge: true`.


- **`languages.evalConfig` (jsonb, migration):** `{script, errorRateUnit: "wer"|"cer", roundTripAsr: {provider, model}|null, codeSwitchPartners: string[], diglossia: boolean, targetVariety: string, judgeReliabilityTier: "high"|"low"|"unsupported"}`. All judge prompts, metrics, and dashboards read this — nothing hardcodes a language (FR19).
- **`changed_from_prev` derivation:** diff `scenario_versions.config` against its `parentVersionId`'s config (top-level keys + the per-language sub-objects) → list of changed elements, exposed on the experiment API. Prompt-template changes are already versioned per code in `promptVersions`. Surface a **one-variable warning** when a comparison spans versions differing in >1 element (PRD's isolation rule made visible, not enforced).
- **Translations versioning fix (§1.3.1 decision):** include `scenario_translations` content (`translationOpeningStatements`, `translationDescription`, `translationTitle`) in `buildConfigFromScenario` (`scenario-version.service.ts:311`) so per-language opening-statement changes are version-diffable and publish round-trips faithfully. Verify the publish path (`updateScenario` replay) handles these DTO fields idempotently; add a regression test for snapshot→publish→snapshot equality.
- **`eval_experiments` table (lightweight):** `{id, name, description, scope (language/scenario/version/model filters as jsonb), changed_element, is_pinned_reference, judge_version_at_pin, created_by}`. The dashboard's pinned-reference selector (FR13) and delta readouts (FR16) read from this. An "experiment" is a saved filter tuple, not a new runtime concept — sessions already carry everything needed.

### Phase 4 — Dashboard (FR6–14) — ✅ LANDED (2026-07-10; verified in Chrome against real Gemini-judged local sessions)

Shipped: Analytics **Language tab** — diagnostic ladder w/ gate+masking (FR6), objective panel w/ unmeasured states (FR7, fills in with Phase 2), entanglement panel (FR8), severity-stacked dimension rollup (FR9), per-layer weekly trend (FR10), experiment slicing by scenario-version/prompt-version/model (FR13-lite; pinned-reference deltas ship with Phase 3), error log w/ isolation basis + session deep links (FR12), no scalar scores (FR14). **Session-logs detail** — Language quality section (denominators, per-session weighted rate, dimension chips, prompt-vs-model verdict, annotation cards), per-turn category badges on transcript bubbles (messageId-anchored), and a **Conversation drift** section + drift/garble chips (closing the drift-invisible-on-sessions gap). Endpoints: `GET /v1/analytics/language-quality`; drift + language blocks on `GET /v1/roleplay-session-logs/:id`.


**ally-be:** `language-analytics.repository.ts` (clone `applyDriftFilters` pattern; add experiment/reference params) with: weighted-error-rate by dimension/layer (severity-stacked), dominant category per dimension, per-layer trend across experiment values, objective-metric aggregates + deltas, gate status, error-log page query. Assemble in `platform-analytics.service.ts` → `GET /v1/analytics/language-quality` (+ error-log pagination endpoint), SUPER_ADMIN.

**ally-web:** new **"Language" tab** in the `TABS` registry (`Analytics.tsx`) built from `chartKit.tsx`:
1. **Diagnostic ladder** — 4 layers bottom-up; the round-trip-WER gate visually masks the human-audio layer when failed (grey-out + "unmeasured, not fine").
2. **Objective panel** — round-trip WER/CER + script fidelity: value, delta vs pinned reference, "isolates: TTS pronunciation / rendering" tags.
3. **Error rollup** — severity-stacked weighted error rate/100 per dimension (stacked bar), dominant category label, implicated element from the FR17 map, delta vs reference.
4. **Per-layer trend** — line chart per layer across experiments (the isolation-leak check, FR10).
5. **Error log** — Carbon DataTable of annotations (evidence, isolation basis) with filters; row → session detail.
6. Header: pinned-reference selector, `judge_version`, `n_turns`, per-language slicing. **No 1–5 score anywhere** (FR14).

**Roleplay session logs integration:** add `findLanguageAnnotations(sessionId)` (and `findDriftJudgment` — closing the same gap for drift) to `roleplay-session-logs.repository.ts`; render per-turn badges + reasoning tooltips on the transcript bubbles in `RoleplaySessionLogDetail.tsx`. Reconcile the AI-turn-ordinal vs message-id mismatch by returning the computed turn index from the backend.

### Phase 5 — Human audio track (FR4) — **DESCOPED (decision 2026-07-08)**

Decision: no dedicated CMOS/win-rate rater tooling. Human evaluation of the audio layer happens informally via the **existing session-log audio player** (`RoleplaySessionLogDetail` recording playback) — listeners take it from there. Consequences, stated so the dashboard doesn't overclaim:
- **Round-trip WER/CER is the only measured Realization metric.** Naturalness, prosody, affect, and dialect-accent (dims 7–9 + accent) are *unmeasured*, not "fine" — the dashboard's diagnostic ladder shows them as "manual listening only".
- The pinned-reference comparison for TTS experiments is therefore round-trip WER deltas + anecdotal listening, which cannot rank two intelligible voices on naturalness. If a TTS-swap decision ever hinges on that, revisit a minimal pair-rating protocol (the Phase-2 synthesis client makes pair generation cheap when needed).

### Phase 6 — Ops & iteration (ongoing)
- Nightly scheduled judging for new sessions (clone drift scheduler); sampling policy per language; batch/cheap-tier judge once the rubric stabilizes.
- Error-analysis loop: failures → test cases → the existing **rehearsal harness** (`ally-ai-learn/app/roleplay_v2/rehearsal/`) and **V2V tester** run one-variable experiment batches pre-deployment (§9 of PRD) — these harnesses already exist and become the experiment execution engine.

### Sequencing & effort summary

| Phase | Duration | Depends on |
|---|---|---|
| 0 Inventory & design freeze | 2–3 d | — |
| 1 Language judge (text) | 1.5–2 wk | 0 |
| 2 Objective metrics | 1–1.5 wk | 0 (parallel w/ 1) |
| 3 Experiment/lang config | 1 wk | 0 |
| 4 Dashboard | 1.5–2 wk | 1–3 |
| 5 Human audio track | **descoped** | — |
| 6 Ops/iteration | ongoing | 1–4 |

Rough total to the full shipped scope (Phases 0–4): **~6–7 weeks** of focused work.

---

## Part 3 — Decisions (resolved 2026-07-08)

1. **Engine scope:** both `SIMULATION` (v1) and `ROLEPLAY_V2` from day one; `engine` is a slice dimension.
2. **Judge topology:** separate language-quality judge call (independent rubric versioning/tuning; ~2× judge cost accepted, mitigated by sampling + cheaper tier once stable).
3. **Language coverage:** all languages sessions run in, from day one. **No hand-labeling / no κ calibration gate** — LLM judge runs directly on transcripts; `judgeReliabilityTier` is a declared confidence label only.
4. **Human audio track:** descoped — no CMOS/rater tooling; listeners use the session-log audio player. Round-trip WER is the only measured Realization metric; dashboard must show dims 7–9 + accent as "manual listening only".
5. **Translations versioning gap (§1.3.1):** fix in Phase 3 — include `scenario_translations` in version snapshots; accepted that publish then restores translations.
6. **STT conditioning (§1.3.3):** self-contained `input_garbled` flag in the language judge; garbled turns excluded from Comprehension/Adequacy denominators; drift judgment used only as offline cross-check.
7. **Existing scalar judges (§1.3.5):** keep scenario-report 0–100 scalars as product features; eval dashboard never shows them; review deprecation post-calibration.

### Remaining defaults (proceeding unless overridden)
- **Round-trip WER home:** ally-ai (batch STT + eval role live there; add a thin TTS-synthesis client, using learn-core's `app/tts/factory.py` as reference).
- **Judge model:** the pinned Gemini already used by drift (pinned per NFR1); benchmark alternatives later per the drift spec's expansion recipe.
