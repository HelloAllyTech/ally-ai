# Conversation Drift Metrics — Spec v2

**Goal:** Upload N transcripts → for each: did it drift, where it started, what *kind* of failure, and whether the root was STT or the LLM — with enough evidence attached to actually fix it. Aggregate across sessions to prioritize and to measure whether a fix worked.

**Two purposes, kept distinct:**
- **Detection + KPI** (is there a problem, how big, trending which way) → the drift-rate metric.
- **Diagnosis** (what specifically to change) → the failure-type enums + evidence. This is what makes the output *fixable* rather than just *informative*.

**Design:** One LLM judge call per *session* takes the whole transcript and emits a per-turn array + a session rollup. Fleet-level numbers are computed from those in code. Everything serves either detection or diagnosis; nothing is there for its own sake.

---

## Roles in this product (read first — easy to get backwards)

In Ally roleplay sessions the **AI plays the client** (e.g. "Kamakshi") and the **human trainee is the counselor** (e.g. "Kriti").

- **"Agent we judge for drift"** = the **AI client**. Drift = the AI client going incoherent, off-character, or off-topic.
- **"User utterance subject to STT"** = the **human counselor's** speech. The `*_garbled` signal is about *the counselor's* transcript.

The judge prompt MUST state these roles explicitly, or it will mis-assign blame — the same role-confusion the AI itself exhibits when it drifts.

---

## The one LLM call (per *session* — whole transcript in, per-turn out)

One call per session. Input: scenario goal **and the AI client's persona** (`scenarios.prompt` in session language) + the **entire transcript** (all turns, with timestamps and speaker roles). Whole-transcript input is deliberate: gradual decay, STT→LLM cascades, and the in-character arc are trajectory-level — a global view sees them naturally where a per-turn window can't. It is also cheaper (each turn sent once, not re-sent in N overlapping windows).

Output: a **per-turn array** (one object per AI-client turn) + a **session rollup**. The input is whole-transcript but the output stays per-turn so storage, calibration, and aggregation are unchanged — never collapse to a single free-text session verdict (that goes vague and can't be validated turn-by-turn).

```json
{
  "per_turn": [
    {
      "turn_index": 12,
      "coherence": "degrading",            // anchored ordinal — see scale below; catches gradual decay
      "topic_label": "on_topic",           // on_topic | tangent | off_topic | gibberish
      "in_character": true,                // is odd output actually realistic distressed-client portrayal?
      "counselor_utterance_garbled": "partial",   // none | partial | severe  (STT signal)
      "ai_reply_failure_mode": "repetition",       // see enum below; "none" if clean
      "root_attribution": "stt_cascade",   // see attribution rule below
      "reasoning": "<one sentence>"
    }
    // ... one object per AI turn
  ],
  "session": {
    "drifted": true,
    "first_drift_turn": 12,
    "attribution_mix": { "stt_direct": 1, "stt_cascade": 2, "llm_direct": 0, "context_lockin": 0 }
  }
}
```

**Judge each AI turn independently** for coherence given only what preceded it — state this explicitly in the prompt and show few-shot examples that discriminate turn-by-turn *within* one transcript. This counters the halo effect: with the whole conversation visible, a model tends to smooth over a bad turn ("it resolved later") or over-flag neighbors of one bad turn. The isolated-window approach got this for free; whole-transcript must be instructed into it.

**Length guard:** process whole-transcript up to ~30–40 turns. Above that, attention degrades mid-transcript ("lost in the middle" — the same failure we study), so chunk with a few turns of overlap and stitch the per-turn arrays. Most sim sessions are short and take a single call.

### Why each field exists

- **`coherence` (anchored ordinal, NEW vs v1):** the binary topic label misses the most common real failure — *gradual* decay where the AI stays loosely on-topic but degrades into repetition / role-slips / garbled words (see the Tamil example that motivated this). An anchored level catches that; the label alone scored those turns "on_topic" and would have missed the drift. It is a **5-level ordinal**, not a raw 0–1 score, because a free-floating number isn't reproducible across runs/languages/prompt versions and can't be hand-labeled (you can't agree on "0.35," but you can agree on "degrading"):

  | Level | Meaning | Numeric (for plots only) |
  |---|---|---|
  | `fully_coherent` | clean, responsive, in-character | 4 |
  | `minor_disfluency` | small awkwardness, still clearly fine | 3 |
  | `degrading` | noticeably off — repetition, mild role-slip, an odd word | 2 |
  | `mostly_incoherent` | hard to follow, multiple failures | 1 |
  | `gibberish` | not a coherent utterance | 0 |

  The judge emits the **level**; code maps it to the number when a plot needs an axis. Calibrate against hand labels on the level, not the number.
- **`topic_label`:** still useful for clean off-topic/gibberish jumps. Carve-outs (NOT drift): counselor-led topic change, Hinglish/code-switching, backchannels, terse-but-valid replies.
- **`in_character` (NEW):** the AI is *playing* a distressed person — rambling, "I don't know what to do," and repetition can be **realistic portrayal**, not failure. The judge must separate in-character distress from genuine drift. This is the subtlest part of the task and the main reason calibration (below) is non-negotiable.
- **`counselor_utterance_garbled`:** STT-failure signal on the human's turn. Three levels, not a boolean, because partial garbles (recoverable) and severe garbles (meaning destroyed) lead to different conclusions.
- **`ai_reply_failure_mode`** and **`root_attribution`:** the diagnosis layer — see below.
- **`user_text` / `ai_text` (evidence — NOT in the LLM output):** the failing turn's verbatim text is stored on the judgment row so an engineer can read it without re-joining four tables. But the **judge does not echo it** — the batch job / worker populates it by joining `turn_index → scenario_session_messages` at write time. Echoing the transcript back in the output would roughly *double* token cost (transcript in, transcript out) for zero benefit. So: evidence lives in storage, reconstructed from source, not generated by the LLM.

---

## The diagnosis layer (what makes it fixable, not just informative)

A single "STT vs LLM" boolean collapses bugs with different fixes. Two closed enums make it aggregatable — so you can find "**62% of Tamil drift is wrong-language replies**" (a fixable finding) instead of "drift is LLM-induced" (a shrug).

### `counselor_utterance_garbled` → STT error sub-type (when garbled)

`entity_swap | phonetic_garble | wrong_language | number_format | code_mix_fail | truncation`
(e.g. the Tamil "1 2" = `number_format`/`phonetic_garble`; "கென்னல்லா" = `phonetic_garble`.)

### `ai_reply_failure_mode` → LLM failure sub-type

`hallucination | context_lockin | wrong_language_reply | repetition | role_slip | wrong_intent | none`
(absorbs role-adherence as `role_slip`; the Tamil example showed `repetition` + `role_slip`.)

### `root_attribution` — looks BACK, not just at this turn (NEW vs v1)

The naive per-turn test (is *this* input garbled?) mis-blames cascades: drift often *shows up* at turn N with a clean input, but the *root* was a garble 2–3 turns earlier that poisoned the context ("lost in conversation"). So attribution considers the prior 2–3 turns:

| Pattern | `root_attribution` |
|---|---|
| Garble on this turn, reply sensible given it | `stt_direct` |
| Reply degrades now, but a garble in the last 2–3 turns is the likely root | `stt_cascade` |
| Inputs clean across the window, reply still incoherent | `llm_direct` |
| Incoherent given clean input that referenced earlier context | `context_lockin` (summarizer/memory/window — needs LLM I/O to confirm) |

`stt_direct` + `stt_cascade` together = the true STT-attributable share. Reporting only `stt_direct` (the v1 behavior) systematically undercounts STT.

---

## The derived numbers (per session, pure code)

**1. Drifted? (yes/no)** — true if EITHER: ≥2 consecutive turns with `topic_label ∈ {off_topic, gibberish}`, OR a run of ≥2 turns at `coherence ≤ degrading` while `in_character = false`. (The second clause is the gradual-decay catch the Tamil example proved we need.)

**2. First-drift turn** — index of the first qualifying turn; reported raw and as a fraction of total turns → "after the nth utterance."

**3. Attribution mix** — share of drift turns by `root_attribution` (STT-direct / STT-cascade / LLM-direct / context-lockin) and by failure sub-type. This is the number that tells you what to fix.

---

## The aggregate (across sessions, SQL → REST → Carbon — NOT Metabase)

Follow the existing super-admin analytics page pattern: pure SQL in `ally-be` (DataSource query builder, like `getVoiceLatencyByBucket`) → REST endpoint → React/Carbon charts. No Metabase dependency.

- **Drift rate by language** (primary KPI) — % sessions drifted, grouped by `language` (then `scenario_id`, `llm_model`, **`prompt_version`, `provider`** for experiments — see experiment-config capture below).
- **Attribution mix per language** — stacked STT/LLM/cascade/context split. The most decision-relevant plot.
- **Failure-mode breakdown** — which specific failures dominate per language.

---

## Analytics UI — two modes: prioritize, then diagnose

Built on the existing super-admin analytics page (Carbon charts, lazy-loaded, SUPER_ADMIN-gated). Two layers, because the data serves two jobs:

**A. Aggregate view (prioritize + track KPI)** — reuses existing Carbon components:
- **Filter bar** (net-new but small): language / scenario / LLM model / date range. Drives every chart below. This is the one piece the current analytics page lacks.
- **KPI tiles** (reuse `Analytics.tsx` tiles): overall drift rate %, worst-language drift rate, STT-attributable share %, median first-drift turn.
- **Drift-rate trend** — `LineChart` (clone of voice-latency), one line per language. The experiment KPI: "Tamil 31% → 14%."
- **Attribution mix** — `StackedBarChart` (clone of retention): per language, share of `stt_direct / stt_cascade / llm_direct / context_lockin`. The "what to fix" plot.
- **Failure-mode breakdown** — `DonutChart` (clone of users-by-role): which specific failures dominate (e.g. `wrong_language_reply`, `repetition`).
- **Coherence-by-turn-position** — `LineChart` of mean coherence (using the ordinal→number map) against turn index, per language. Answers "after the nth utterance" and shows the decay curve visually.

**B. Drill-down view (diagnose + fix)** — net-new, the part that makes this fixable rather than just informative:
- Click any chart slice → **`DataTable`** (Carbon) of the matching sessions: language, scenario, first-drift turn, attribution, dominant failure mode.
- Click a session → **turn inspector**: the per-turn rows rendered as a conversation, each AI turn color-coded by `coherence` level, showing `user_text` / `ai_text` / failure mode / `reasoning` inline. For non-English, show the literal translation alongside. This is where an engineer forms the fix hypothesis — it's the in-app version of reading the Tamil transcript we did by hand.

The aggregate view answers "where's the problem and is it shrinking"; the drill-down answers "what exactly happened in these turns." Without B, the dashboard is analytics-only — exactly the gap raised earlier.

## Storage

One **wide, denormalized** `turn_drift_judgment` table, mirroring how `scenario_session_turn_metrics` is already shaped — so the analytics query is single-table, not a fragile 4-way join.

Each row: `scenario_session_id`, `turn_index`, + every judge field above, + **denormalized `language`, `scenario_id`, `llm_model`, `occurred_at`** (stamped from `turn_metrics`, since language lives there, not on sessions or messages). Judge output is mutable eval data and re-run when the prompt changes → keep it in its own table, not as columns on `turn_metrics`.

**Inputs already exist** for historic backfill: transcripts (`scenario_session_messages`), persona (`scenarios.prompt` + translations), language/scenario/model (`turn_metrics`), timestamps. Nothing new needed to *start* judging — but see experiment-config capture below for what's needed to make the results *attributable to a prompt variant*.

### Experiment-config capture (required for prompt experiments)

The dashboard's whole purpose is to compare experiments, and an experiment is a tuple — **(prompt_version, model, generation params, STT config)**. The KPI must be sliceable per tuple, so each session's tuple has to be recorded at generation time and denormalized onto the judgment row alongside `language`/`llm_model`.

**Tier 1 — config identifiers (do first; small, structured, cheap):**
- `prompt_id` + `prompt_version` (which system-prompt variant ran — the primary experiment dimension)
- generation params: `temperature`, `top_p`, `max_tokens`, `provider` (we run OpenAI / Gemini / Claude — provider is a real dimension)
- `stt_provider` + config; retrieval/guardrail config if they vary per experiment

These become **slice dimensions on the dashboard** (add `prompt_version` and `provider` to the filter bar next to language/model). The experiment question — "did prompt v3 cut Tamil drift vs v2?" — is then a single grouped query.

> ⚠️ **Confirmed gap (investigated 2026-06-15).** None of the Tier-1 fields are persisted today:
> - **prompt_version: NOT captured.** Prompts are resolved at session start in [scenario-shared.service.ts:898](ally-be/src/learn/service/scenario-shared.service.ts) `getPromptsForScenarioSession()`, which fetches `prompts.currentVersion` and sends the text to ally-ai in `roomMetadata` — but the version number is never written to the session. **Worse than non-sliceable: it's non-reconstructable.** Because only the *mutable* `currentVersion` is read and never recorded, the moment anyone edits a prompt (bumping `currentVersion`), every past session silently loses which version it actually ran. So historic sessions **cannot** be attributed to a prompt version retroactively — the backfill can slice by language/scenario/model but **not** by prompt_version.
> - **generation params (temperature/top_p/max_tokens): NOT captured.** Defined in `LlmProviderConfig` and resolved at call time; never persisted on session or turn.
> - **provider: only partially distinguishable.** `turn_metrics.llm_model` holds the model id only (no provider column) — OpenAI/Gemini/Claude must be inferred from the model string.
>
> **Smallest forward-capture changes (do these on the generation side, urgently — they only help future sessions):**
> - prompt_id + `prompts_versions.id` (the immutable version row, not `currentVersion`) → `scenario_sessions.metadata`, at [scenario-session.repository.ts:144](ally-be/src/learn/repository/scenario-session.repository.ts).
> - temperature/top_p/max_tokens → add to `LearnTurnMetricsData` ([learn-message.interface.ts:23](ally-be/src/learn/interface/learn-message.interface.ts)) emitted by ally-ai, ingested in [scenario-session.service.ts:1430](ally-be/src/learn/service/scenario-session.service.ts) `addTurnMetrics()`.
> - explicit `llm_provider` (`openai`/`anthropic`/`gemini`) alongside `llm_model` on `turn_metrics`.
>
> Capture the immutable version id, not `currentVersion` — recording a mutable pointer reproduces the same non-reconstructability bug.

**Tier 2 — assembled prompt + raw completion text (heavier, forward-only):** the LLM-diagnosis prerequisite — needed to confirm `context_lockin` (was the context even in the prompt?). Tier 1 lets you *compare* variants; Tier 2 lets you *diagnose why* one drifts. Tier 1 is the priority.

---

## Cost & pipeline integration

### Per-session cost

One whole-transcript call. Rough budget: **~8,000 input tokens** (≈2,500 static prefix + ~500 persona + ~5,000 transcript) and **~1,200 output tokens** (per-turn labels + rollup, *not* echoing transcript). The swing factor is **Indic scripts tokenize 2–4× heavier than English** — verify with `count_tokens` on real Tamil transcripts before committing. Order-of-magnitude with batch pricing: small/cheap judges (Gemini Flash, gpt-4o-mini) land well under a cent/session; mid-tier (Sonnet-class) ~2¢. Backfilling ~50k sessions ≈ low hundreds to ~$1k depending on model.

### Cost levers (ordered by impact)

1. **Pre-filter which sessions to judge** — judge **all** simulations including English (drift happens in English too); pre-filter only by risk signals if cost-constrained (guardrail fired / low score / long), **not** by language. Language is a slice dimension on the dashboard, not an inclusion gate. Cuts *volume* — the dominant cost driver — without dropping English.
2. **Batch API (−50%)** — drift analysis is offline; use it for both backfill and the nightly path.
3. **Don't echo transcript in the output** — reconstruct `user_text`/`ai_text` from `turn_index` (see judge schema). ~4× output reduction.
4. **Whole-transcript single call** (already chosen) — each turn sent once, not re-sent in N windows.
5. **Sample for the fleet KPI** — the drift-rate dashboard needs a statistically valid sample per language/week, not 100% coverage. Judge everything only for per-session drill-down or rare-failure hunts.
6. **Model tiering** — validate a cheap judge against the seed set; if it clears the κ bar per language, run the bulk on it and escalate only ambiguous sessions.

Avoid: transliterating Indic text to cut tokens — it corrupts the wrong-language / script-mismatch signals the judge must detect.

### Pipeline integration — write the judge once, call it from two places

- **Judge function + `turn_drift_judgment` table + dashboard** — new, shared by both paths below.
- **Ongoing sessions → reuse the post-session summarizer worker.** Add the judge as a sibling call (`asyncio.gather` alongside the summary) so it reuses the already-loaded transcript and the existing SQS→Postgres return path. Make it **gated** (only in-scope languages), **non-blocking, and failure-isolated** — a slow/failed judge must never delay or break the summary; re-run misses via the batch job.
- **Historic backlog → standalone Batches job** over `scenario_session_messages`. Not duplication — a post-session hook can't reach already-ended sessions. Same judge function, same schema, same table; the only unique code is the paging loop.

> ⚠️ **Domain caveat:** the summarizer pipeline traced earlier runs on the **live-client-chat** domain (`chat_id` → `call_details`), but drift (and the Tamil example) is in the **scenario-session** domain (`scenario_session_*`). Hook the **scenario-session** summarizer, not the chat one — confirm that hook exists before wiring it in.

---

## Validation — non-negotiable, not a phase to skip

The Tamil example proved the judge task is subtle (in-character distress vs. real drift; gradual decay; cascades). An unvalidated judge would mislabel it. So:

- Hand-label ~30 sessions; for languages you can't read (Tamil, etc.), label on a **literal** translation (instruct the translator to PRESERVE garble, mark `[GARBLED]`, not smooth it) **and** get a **native-speaker check on a ~20% slice**.
- Tune the judge prompt (esp. the `in_character` and `coherence` boundaries) until `topic_label`/failure-mode agreement ≈ 85% (Cohen's κ ≥ 0.7) per language.
- No dashboard ships for a language below threshold. Re-run agreement whenever the judge model or prompt version changes.

---

## Sequencing — STT first, LLM logging in parallel

1. **Now:** stand up `turn_drift_judgment` + the batch judge job; run the single detection+attribution pass over historic sessions (all languages, English included). This yields drift rate, the STT/LLM/cascade split, and STT failure sub-types — all from existing data, fully diagnosable today (the transcript IS the STT evidence).
2. **Now, in parallel — forward-only capture, do NOT defer.** These only help *future* sessions, so every day of delay is data you can never recover. Land them while you work the STT half:
   - **2a. Experiment-config (Tier 1) — highest priority.** Capture `prompt_version` (immutable `prompts_versions.id`), generation params (temp/top_p/max_tokens), and explicit `llm_provider`. Without these, prompt/provider A/Bs cannot be measured *at all*, and the mutable-`currentVersion` bug makes every un-captured session **permanently unattributable** — there is no backfill. Land this **before running any prompt experiment.** Sites in the experiment-config capture section above.
   - **2b. LLM prompt + completion text (Tier 2).** Log the assembled prompt + raw completion per turn to our own Postgres (NOT LangSmith — ships PHI to an external cloud, awkward to join). Needed to *diagnose* the LLM bucket (confirm `context_lockin`).
3. **First fixes:** act on the STT-attributable bucket (`stt_direct` + `stt_cascade`) — fully diagnosable now. Fixing clear STT cases should also shrink the LLM bucket (cascades disappear).
4. **Then:** diagnose the LLM bucket using the Tier-2 logs accumulated since step 2b, sliced by the Tier-1 config from 2a (which prompt version / params / provider drifts more).

---

## Optional STT ground-truth (only if attribution comes back STT-heavy)

If the STT bucket is large and you want to compare providers: sample ~50–100 utterances/language from S3 session audio, human-verify references, and run Indic semantic metrics — **LLM-WER, Intent Score, Entity Preservation** ([sarvam llm_wer](https://github.com/sarvamai/llm_wer), [llm_intent_entity](https://github.com/sarvamai/llm_intent_entity)). The `counselor_utterance_garbled` flag already tells you whether this is worth doing.

**On our self-hosted [Calibrate](https://calibrate.artpark.ai/docs) fork (`ally-calibrate`):** its STT eval is real and reusable for the **provider comparison** (Sarvam/Google/Deepgram + others, Indic, CSV/S3 batch, auto-leaderboard) — but it computes **WER / CER / semantic-similarity only, NOT Intent Score or Entity Preservation.** So Calibrate covers the WER/CER half; run sarvam's `llm_intent_entity` separately for the Intent/Entity half (the metrics most tied to drift). Calibrate is **not** suited to hosting the drift judge itself — it runs predefined test cases (not raw transcripts), its judge is hardcoded to OpenAI (we chose Gemini), and it emits pass/fail per test, not a per-turn array + rollup. Build the judge ourselves; use Calibrate only for STT/model benchmarking.

---

## Extending the judge — recipe for adding a new judged signal (as-built)

> This section documents the **shipped** architecture, which refined the
> "write the judge once" plan above. The judge is **stateless in ally-ai**;
> **ally-be owns the data and orchestrates**. `ally-ai` never touches ally-be's
> Postgres — it's a pure transform (transcript in → per-turn judgments out).
> Use this recipe to add a *new judged signal* (a new per-turn field — e.g. an
> `empathy_level` label, a new STT sub-type) end-to-end.

**Where each responsibility lives (the seam):**

| Concern | Repo / file |
| --- | --- |
| Judge prompt + Pydantic schema | `ally-ai/app/core/drift/prompt.py`, `schemas.py` |
| Judge logic + session rollup (`DRIFT_RUN_K`) | `ally-ai/app/core/drift/judge.py` |
| Stateless judge endpoint (`POST /api/v1/drift/judge`) | `ally-ai/app/api/v1/endpoints/drift.py` |
| Session selection, transcript build, **persistence** | `ally-be/.../repository/drift-judge.repository.ts` |
| Orchestration (select → call ally-ai → persist), Redis job state | `ally-be/.../service/drift-judge.service.ts` |
| Aggregation SQL over `turn_drift_judgment` | `ally-be/.../repository/drift-analytics.repository.ts` |
| Assemble the dashboard payload | `ally-be/.../service/platform-analytics.service.ts` (`getConversationDrift`) |
| Response contract | `ally-be/.../dto/platform-analytics.dto.ts` (`ConversationDriftResponseDto`) |
| Endpoint | `ally-be/.../controller/analytics.controller.ts` (`GET conversation-drift`) |
| Chart kit (palette, `ChartCard`, option factories) | `ally-web/.../pages/Analytics/chartKit.tsx` |
| Drift charts + types | `ally-web/.../pages/Analytics/ConversationDrift.tsx`, `types/auth.ts`, `api/analytics.ts` |

**Recipe — add one new per-turn judged field, in dependency order:**

1. **ally-ai — teach the judge.** Add the field to the prompt rubric
   (`prompt.py`) and to the per-turn Pydantic model (`schemas.py`). If the field
   is unconditional — an answer exists on every turn — also declare it
   **required** on `LiveTurnJudgment`, the strict subclass handed to Gemini as
   `response_schema`. `PerTurnJudgment` stays lenient so older stored rows read
   back, and an optional field is one the model returns only where it fired: a
   rate computed over the turns carrying the label then reads far too high. If
   it feeds the session rollup, extend `compute_session_rollup` in `judge.py`.
   The endpoint needs no change — it returns whatever the schema holds.
   Add/extend a judge unit test.
2. **ally-be — persist it.** Add the column via a **new** idempotent migration
   (`ADD COLUMN IF NOT EXISTS`; never edit a released migration; unique
   timestamp), then store it in `upsertJudgments` (drift-judge.repository.ts).
   Backfill re-judges to populate it.
3. **ally-be — aggregate it.** Add a query method to
   **`drift-analytics.repository.ts`**, routing through `applyDriftFilters` so
   the new chart inherits the shared time/language/experiment filters for free.
   Whitelist any column interpolated into SQL (see `getDriftSessionCountsBy`).
4. **ally-be — expose it.** Call the new method in
   `PlatformAnalyticsService.getConversationDrift`, add the field to
   `ConversationDriftResponseDto`, and it flows out the existing endpoint.
   Add the repo mock to `platform-analytics.service.spec.ts` if you add a
   constructor dep.
5. **ally-web — chart it.** Add the field to `ConversationDriftResponse`
   (`types/auth.ts`), then render a `ChartCard` + an option factory from
   `chartKit.tsx` in `ConversationDrift.tsx`. Reuse `PALETTE`; don't hand-roll
   option objects or hex literals.
6. **Gate:** run **all** tests in every touched repo and keep them green before
   committing (`ally-be` jest, `ally-ai` pytest, `ally-web` jest). Add new tests
   for the new field; don't commit on red.

**Adding a whole new analytics *tab* (not just a chart):** the dashboard is a
tab registry — add an entry to `TABS` in `Analytics.tsx` (`{id, label, uses,
render}`), declare `uses.language` to opt into the shared page-level language
picker, and build the tab component from the chart kit (see
`tabs/OverviewTab.tsx` / `LatencyTab.tsx` as templates). No page-shell changes
needed.

---

## Explicitly cut (still not needed to fix)

Recovery-latency curves, hazard/survival plots, guardrail precision/recall. They characterize the problem more finely but don't change what you'd do about it. Add only if a specific question demands them.

## Config & defaults (calibrate against the seed set, then freeze)

Judge: temperature 0, **pinned model version**, structured JSON output, 2–3 per-language few-shots including garbled-STT and in-character-distress negatives plus turn-by-turn discrimination within one transcript, version-controlled prompts. Tunables: K = 2 consecutive, coherence drift cutoff = `degrading` (i.e. `degrading` or worse counts), attribution look-back = 3 turns (the span the judge considers when deciding `stt_cascade` vs `llm_direct`), whole-transcript length cap ≈ 30–40 turns before chunking. All are starting hypotheses.

**Judge model: Gemini for now (decided), benchmark and expand later.** Start with a pinned Gemini model as the judge — its strong Indic coverage fits Tamil/Telugu/Bengali, which is where the hard calls are. This is the initial choice, not a permanent lock-in. Two conditions still hold:

- **The calibration gate still applies** — Gemini must clear the per-language κ bar (§validation) before its output ships to a dashboard for that language. If a cheaper Gemini tier (e.g. Flash) misses the bar on the subtle calls (in-character-vs-drift, Indic gibberish), step up to a stronger Gemini tier for that language rather than shipping a weak judge.
- **Watch the same-family caveat.** If/when the *agent itself* generates a session on Gemini, the judge shares that family and may go easy on `llm_direct` failures for those sessions. Note it; revisit in the expand phase.

**Later expansion:** when broadening beyond Gemini, select by running the hand-labeled seed set through 2–3 candidates (OpenAI / Gemini / Claude) and picking per-language by agreement (κ) and cost — exactly what self-hosted **Calibrate** does (LLM text-to-text comparison + leaderboard). Pin the winner per language.
