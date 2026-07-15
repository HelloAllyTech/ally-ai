# Language Eval — Prompt-Metadata Inventory & Attribution Map (FR17)

**Status:** v1 — authoritative for the Language-Capability Evaluation & RCA system.
**Companion docs:** `language-capability-eval-implementation-plan.md` (plan), `language-eval-judge-schema.md` (judge schema + rubric).
**Sources verified in code:** `ally-be/src/learn/dto/update-scenario.dto.ts` (the studio form = the variable surface), `ally-be/src/learn/service/scenario-shared.service.ts` (what is actually sent to the agent), `ally-ai-learn/app/core/scenario/base.py` (`PromptData`, what the agent consumes), `ally-ai-learn/app/roleplay_v2/spec/models.py` (`ScenarioSpec`), `ally-be/src/learn/service/scenario-version.service.ts` (`buildConfigFromScenario`, what is versioned).

This document is normative for the dashboard's **"implicates"** column and for experiment design: an experiment's `changed_from_prev` must name an **element ID** from this inventory. It replaces the PRD §4.6 illustrative list.

---

## 0. Reading guide

- **Element ID** — canonical `surface:element` name used everywhere (judge output, experiment tags, dashboard).
- **Indep?** — can the studio vary this element alone (one-variable experiment possible without code changes)?
- **Versioned?** — is a change captured in a version snapshot (attributable + reconstructable)?
- **Drives** — PRD §5 dimension numbers this element plausibly moves. Judged text dimensions: **2** understanding, **3** adequacy, **4** fluency, **5** coherence, **10** register, **11** dialect-lexicon, **12** colloquialness, **13** persona/social, **14** code-switching. Realization (audio): **6–9** + accent.

Two structural corrections to the PRD, established in plan §1.3:
1. **Backstory and role instructions are ONE element** (`scenario:prompt`). `scenarios.prompt` is delivered to the agent verbatim as `roleInstructions` (scenario-shared.service.ts:394-396) and doubles as the judges' `persona`. Experiments cannot vary "backstory" and "role instructions" separately today.
2. **Prompt metadata spans two versioned surfaces** — scenario config (`scenario_versions.config`) and prompt templates (`prompts_versions`, captured per-session in `metadata.promptVersions`) — plus, for v2, `roleplay_spec_versions`.

---

## 1. Surface A — Scenario config (v1 SIMULATION engine)

Storage: `scenarios` row + `scenarios.metadata` jsonb + related tables; snapshot = `scenario_versions.config`. Delivered to the agent as LiveKit room metadata → parsed into `PromptData`.

### 1.1 Persona & content elements

| Element ID | Studio field(s) | Indep? | Versioned? | Drives | Notes |
|---|---|---|---|---|---|
| `scenario:prompt` | `prompt` | ✅ | ✅ | **3, 5, 13** (+10/12 if it embeds style directives) | The persona/backstory/role-instruction blob — single element (correction #1). The highest-leverage content element. |
| `scenario:character_facts` | `name, age, gender, genderIdentity, sexualOrientation, currentLocation, profession` | ✅ (each field) | ✅ | **3** (hallucinated facts), **13**; `currentLocation` also weakly **11** (regional cue) | Structured persona facts; underspecification here → backstory hallucination (PRD's canonical adequacy example). |
| `scenario:character_profile_text` | `characterProfileText` | ✅ | ✅ | **3, 13** | Long-form profile supplement. |
| `scenario:custom_fields` | `customFields[]` | ✅ | ✅ | **3, 13** | Arbitrary name/value persona facts. |
| `scenario:behavior_instructions` | `behaviorInstructions[]` (→ helpful/unhelpful behaviours) | ✅ | ✅ | **2, 5, 13** | How the persona reacts to trainee behaviors. |
| `scenario:states` | `states[]`, `stateNames[]` | ✅ | ✅ | **5, 13** | Score-banded emotional states + guidelines. |
| `scenario:knowledge_sources` | `knowledgeSources[]` (RAG) | ✅ | ✅ | **3** | Grounding; absence → hallucination risk. |
| `scenario:previous_memory` | (runtime, from prior session summary) | ❌ (runtime) | ❌ | **3, 5** | Cross-session continuity; not an experiment variable, but a confound to record. |

### 1.2 Language-style elements (the primary Appropriateness levers — all per-language, keyed by languageId)

| Element ID | Studio field(s) | Indep? | Versioned? | Drives | Notes |
|---|---|---|---|---|---|
| `scenario:language_characteristics` | `languageCharacteristics{langId→text}` | ✅ per language | ✅ (in `scenarios.metadata`) | **10, 11, 12, 14** | Free-text per-language style directive (e.g. "simple colloquial Chennai Tamil; code-mixes with English"). **The PRD's register/dialect decision rule keys on whether this is populated.** |
| `scenario:linguistic_style_samples` | `linguisticStyleSamples{langId→[utterances]}` | ✅ per language | ✅ | **10, 11, 12, 14** | Few-shot exemplars of how the persona talks in the target language. Delivered as `languageDialogueSamples`. |
| `scenario:allowed_filler_words` | `allowedFillerWords{langId→[words]}` | ✅ per language | ✅ | **12** (+ text-level naturalness) | Whitelisted fillers/backchannels per language. |
| `scenario:opening_statements` | `openingStatements[]` + `translationOpeningStatements{langId→[lines]}` | ✅ per language | ⚠️ **primary yes; translations NO until Phase-3 fix** | **10, 12, 14** | Per-language openings are stored in `scenario_translations`, currently omitted from `buildConfigFromScenario` — changes are non-attributable until the planned fix lands (plan §1.3.1). |
| `platform:language_label` | `languages.label` (e.g. "Tamil (India)") | ❌ (platform-wide) | ❌ | **10, 11** | Injected as the dialect signal to the LLM; platform config, not per-scenario. Changing it is a platform experiment, not a scenario one. |

### 1.3 Generation & orchestration elements

| Element ID | Studio field(s) | Indep? | Versioned? | Drives | Notes |
|---|---|---|---|---|---|
| `scenario:temperature` | `temperature` | ✅ | ✅ | **4, 5** | Simulation-level; precedence: code → per-language `llm_config` → prompt-level → simulation-level. |
| `config:llm` | per-language `llm_config` (provider/model), prompt-level overrides | ✅ (config/prompt mgmt) | ✅ via turn-metrics capture (`llmProvider`,`llmModel`, params in `metadata`) | **all text dims** (capability fallback axis) | The "model swap" axis — tested only after metadata variants fail (PRD prompt-before-model rule). |
| `config:stt` | per-language `stt_config`; `languages.sttProviderConfig` | ✅ | ✅ via language config + turn metrics | (out of scope — conditioning only) | Recorded for slicing; not attributed by this system. |
| `scenario:voice` | `languageVoices{langId→voiceId}` | ✅ per language | ✅ | **6–9 + accent** (Realization) | The TTS axis: voice/provider choice drives everything human listeners hear. Round-trip WER isolates 6; the rest is manual listening (plan Phase 5 descope). |
| `scenario:orchestration_toggles` | `fillerEnabled, continuousBackchanneling, interimReplyEnabled, comfortAudioEnabled` | ✅ each | ✅ | **4** (transcript disfluency/truncation artifacts) | These inject speech-like artifacts into output text; the judge must not blame the LLM for enabled-filler disfluency — recorded as a conditioning input. |
| `scenario:history_trim` | `historyTrimEnabled` | ✅ | ✅ | **5** (context loss → contradiction) | Maps to drift's `context_lockin`; cross-check with drift judgments. |

### 1.4 Not language-relevant (recorded for completeness, never "implicated")

`title, description, coverImageUrl/coverVideoUrl, isPublic/isGlobal, status, difficultyLevel, competencyId, agentTestCaseIds, selectedEvaluatorPromptCode, terminationEvents/autoTermination*, triggerWarningIds, timerMode/maxTimeValue, optGuardrails, showScoreMeter, enableFeedback, pauseEnabled, experienceMode, checklistType, helperAgentPrompt, agentBuilder*` — orchestration/product concerns. `translationTitle`/`translationDescription` are trainee-facing localization, not agent behavior.

---

## 2. Surface B — Prompt templates (prompt management)

Storage: `prompts` (mutable head, `promptCode` unique, `promptType ∈ {main_agent, branching, multilingual, …}`) + `prompts_versions` (immutable snapshots). Per-session capture: `scenario_sessions.metadata.promptVersions` (`{promptCode: version}`).

| Element ID | What | Indep? | Versioned? | Drives | Notes |
|---|---|---|---|---|---|
| `prompt_template:<promptCode>` | The system-prompt skeleton the scenario elements are injected into; selected per scenario via `selectedMainPromptCode` | ✅ (versioned per code; scenario picks the code) | ✅ (`prompts_versions` + per-session capture) | **all text dims** | Two distinct experiment moves: **(a)** edit a template (new version, same code) — attributed as `prompt_template:<code>@vN`; **(b)** switch which template a scenario uses (`selectedMainPromptCode` change) — attributed as a scenario-config change. Don't conflate them. |
| `prompt_template:llm_overrides` | Prompt-level `provider/model/temperature` columns on `prompts` | ✅ | ⚠️ columns on the mutable head, not snapshotted in `prompts_versions` | 4, 5 + model axis | Same mutable-pointer risk class as the old `currentVersion` bug; per-session capture of effective provider/model/params via turn metrics is the mitigation. |

**Granularity limit (accepted):** a template version change is one element even if the edit touched multiple concerns inside the template text. If a single template edit changes both a register directive and a formatting constraint, `changed_from_prev` cannot see inside it — keep template edits single-concern by convention.

---

## 3. Surface C — v2 `ScenarioSpec` (ROLEPLAY_V2 engine)

Storage: `roleplay_specs` + `roleplay_spec_versions` (ally-be); schema `SPEC_SCHEMA_VERSION="1.0"`; per-session capture via `spec_version_id` on the envelope. Diff for `changed_from_prev` = spec-version config diff by top-level section.

| Element ID | Spec section | Drives | Notes |
|---|---|---|---|
| `spec:persona_identity` | `persona.identity_core` | **3, 13** | |
| `spec:persona_chunks` | `persona.chunks[]` | **3** | Retrievable persona-bible; retrieval miss → hallucination/omission. |
| `spec:scenario_context` | `persona.scenario_context` | **3** | |
| `spec:state_machine` | `state_machine` (incl. `emotional_register`, `state_card`, `default_stage_direction`) | **5, 13**, register nudges **10** | `prosody_hints` also feeds Realization via TTS direction. |
| `spec:disclosure_ledger` | `disclosure_ledger.secrets` | **3, 13** | ⚠️ **Judge interaction:** intentional withholding/deflection of locked content is CORRECT behavior, never an `omission` error. The rubric carries this carve-out (judge-schema doc §3). |
| `spec:opening_statement` | `opening_statement` | **10, 12, 14** | |
| `spec:voice` | `voice.language_voices` | **6–9 + accent** | |
| `spec:actor_model` / `spec:director_model` | `actor_model`, `director_model` | model axis | |
| `spec:rubric` / `spec:engineered_events` | `rubric.behaviors`, `engineered_events` | — | Trainee-scoring/direction config; not agent language. Recorded, never implicated. |

**v2 gap vs v1:** the spec has no per-language style elements equivalent to `languageCharacteristics`/`linguisticStyleSamples` — language style rides inside `identity_core`/`state_card` prose. If v2 language experiments need element-level attribution for style, that's a future spec-schema addition, not something this system can retrofit.

---

## 4. Attribution map (reverse index — what the dashboard "implicates" per dimension)

Order within a cell = test-first priority (PRD prompt-before-model rule; cheapest/most-likely lever first).

| Dim | Dimension | Implicated elements (v1) | (v2) | Fallback axis |
|---|---|---|---|---|
| 2 | Understanding | `prompt_template:main_agent`, `scenario:behavior_instructions` | `spec:state_machine` | `config:llm` |
| 3 | Adequacy | `scenario:prompt`, `scenario:character_facts`, `scenario:character_profile_text`, `scenario:custom_fields`, `scenario:knowledge_sources` | `spec:persona_identity`, `spec:persona_chunks`, `spec:scenario_context`, `spec:disclosure_ledger`¹ | `config:llm` |
| 4 | Fluency | `scenario:temperature`, `scenario:orchestration_toggles`², `prompt_template:main_agent` | `spec:actor_model` config | `config:llm` |
| 5 | Coherence | `scenario:history_trim`, `scenario:states`, `scenario:behavior_instructions`, `scenario:temperature` | `spec:state_machine` | `config:llm` |
| 10 | Register | `scenario:language_characteristics`, `scenario:linguistic_style_samples`, `prompt_template:main_agent`, `scenario:opening_statements` | `spec:opening_statement`, `spec:state_machine` | `config:llm` |
| 11 | Dialect-lexicon | `scenario:language_characteristics`, `scenario:linguistic_style_samples`, `platform:language_label`, `scenario:character_facts` (location) | — (gap, §3) | `config:llm` |
| 12 | Colloquialness | `scenario:linguistic_style_samples`, `scenario:language_characteristics`, `scenario:allowed_filler_words`, `scenario:opening_statements` | `spec:opening_statement` | `config:llm` |
| 13 | Persona/social | `scenario:prompt`, `scenario:character_facts`, `scenario:states`, `scenario:behavior_instructions` | `spec:persona_identity`, `spec:state_machine`, `spec:disclosure_ledger` | `config:llm` |
| 14 | Code-switching | `scenario:language_characteristics`, `scenario:linguistic_style_samples`, `prompt_template:main_agent` | — (gap) | `config:llm` |
| 6–9 | Realization | `scenario:voice` (TTS provider+voice) | `spec:voice` | TTS model axis |

¹ Only via the carve-out: ledger *misconfiguration* can look like omission; the judge never flags intentional deflection.
² Filler/backchannel toggles explain disfluency-shaped artifacts — conditioning input, not an LLM error.

**Register decision rule, operationalized (PRD §9):** for a `too_formal_diglossia` finding in language L on scenario S, check in order: (a) is `scenario:language_characteristics[L]` populated with a colloquial directive? (b) are `scenario:linguistic_style_samples[L]` present and colloquial? If either missing → metadata fix (cheap experiment: populate it, one variable). If both present and error persists → escalate to `config:llm` (model capability limit).

---

## 5. Known limitations (accepted, tracked)

1. `scenario:prompt` is a composite (backstory+role-instructions) — sub-element attribution inside it is impossible without a studio schema split (out of scope).
2. `translationOpeningStatements` non-attributable until the Phase-3 versioning fix lands.
3. Prompt-template internal granularity — single-concern edits by convention (§2).
4. v2 has no per-language style elements (§3) — v2 style attribution stops at `spec:persona_identity`.
5. `platform:language_label` and `languages.sttProviderConfig` are platform-wide — changes affect all scenarios simultaneously; never run them concurrently with scenario-level experiments.
