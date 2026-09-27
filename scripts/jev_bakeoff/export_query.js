// Runs INSIDE the prod ally-be core container (read-only), one call per page.
// argv[2] = base64 JSON params. Prints `BAKEOFF_B64:<gzip+base64 JSON>:END` so
// the caller can detect SSM's 24KB stdout truncation (a missing `:END`).
//
// Transcripts are rebuilt with each judge's OWN builder, so Jev sees exactly
// the conversation the Gemini judge saw:
//   drift        — drift-judge.repository.ts buildTranscript (startSeconds
//                  order, fillers/interims skipped, empty lines kept)
//   groundedness — feedback-groundedness.repository.ts buildTranscript
//                  (createdAt order, empty lines dropped, fillers kept)
const { Client } = require('/app/node_modules/pg');
const zlib = require('zlib');

const AI_SENDER_ID = -1;
const params = JSON.parse(Buffer.from(process.argv[2], 'base64').toString());

const TABLE = { drift: 'turn_drift_judgment', groundedness: 'feedback_claim_judgment' };

// Internal / demo / QA orgs ONLY. Session content is PHI-adjacent (wiki:
// analytics-agent.md), and there is no data agreement with TypeSafe, so no
// real learner's session may leave the platform. Same match as ally-be's
// src/analytics/util/test-tenant.util.ts, inverted: tenant_id holds a tenant
// uuid OR a tenant code.
const TEST_ORG_SESSION = (col) =>
  `EXISTS (SELECT 1 FROM scenario_sessions tts
            JOIN tenants tt ON (tt.id::text = tts."tenant_id" OR tt.code = tts."tenant_id")
           WHERE tts.id = ${col} AND tt."isTestOrganization" = true)`;

async function plan(c) {
  // Pinned version = the (judgeModel, judgePromptVersion) with the most rows in
  // the window. Cross-version comparison is invalid, so everything else is out.
  const out = {};
  for (const [kind, table] of Object.entries(TABLE)) {
    const pin = (
      await c.query(
        `SELECT "judgeModel" AS m, "judgePromptVersion" AS v, count(*)::int AS n
           FROM ${table} WHERE "occurredAt" >= now() - make_interval(days => $1)
          GROUP BY 1, 2 ORDER BY n DESC LIMIT 1`,
        [params.sinceDays],
      )
    ).rows[0];
    if (!pin) continue;
    // Stratified, deterministic sample: up to perStratum sessions per
    // language × actor model, ordered by md5(id) so re-runs pick the same set.
    const rows = (
      await c.query(
        `WITH s AS (
           SELECT "scenarioSessionId" AS id,
                  COALESCE(max(language), 'unknown') AS language,
                  COALESCE(max("llmModel"), 'unknown') AS llm_model,
                  max("scenarioId") AS scenario_id
             FROM ${table}
            WHERE "judgeModel" = $1 AND "judgePromptVersion" = $2
              AND "occurredAt" >= now() - make_interval(days => $3)
              AND ${TEST_ORG_SESSION('"scenarioSessionId"')}
            GROUP BY 1),
         r AS (SELECT *, row_number() OVER (PARTITION BY language, llm_model
                                             ORDER BY md5(id::text)) AS rn FROM s)
         SELECT id, language, llm_model, scenario_id FROM r WHERE rn <= $4`,
        [pin.m, pin.v, params.sinceDays, params.perStratum],
      )
    ).rows;
    out[kind] = { judgeModel: pin.m, judgePromptVersion: pin.v, pinnedRows: pin.n, sessions: rows };
  }
  return out;
}

async function personas(c) {
  const r = await c.query(`SELECT id, prompt FROM scenarios WHERE id = ANY($1::int[])`, [
    params.scenarioIds,
  ]);
  return Object.fromEntries(r.rows.map((x) => [x.id, x.prompt || '']));
}

async function driftTranscript(c, id) {
  const rows = (
    await c.query(
      `SELECT "senderId" AS sender_id, content, metadata->>'utteranceKind' AS kind
         FROM scenario_session_messages WHERE "scenarioSessionId" = $1
        ORDER BY COALESCE("startSeconds", 0), id`,
      [id],
    )
  ).rows;
  const t = [];
  let ai = 0;
  for (const r of rows) {
    if (r.sender_id === AI_SENDER_ID && (r.kind === 'filler' || r.kind === 'interim')) continue;
    if (r.sender_id === AI_SENDER_ID) t.push({ role: 'client', turn_index: ai++, text: r.content ?? '' });
    else t.push({ role: 'counselor', text: r.content ?? '' });
  }
  return t;
}

async function groundednessTranscript(c, id) {
  const rows = (
    await c.query(
      `SELECT "senderId" AS sender_id, content FROM scenario_session_messages
        WHERE "scenarioSessionId" = $1 ORDER BY "createdAt" ASC, id ASC`,
      [id],
    )
  ).rows;
  const t = [];
  let ai = 0;
  for (const r of rows) {
    const text = (r.content ?? '').trim();
    if (!text) continue;
    if (Number(r.sender_id) === AI_SENDER_ID) t.push({ role: 'client', text, turn_index: ai++ });
    else t.push({ role: 'counselor', text });
  }
  return t;
}

async function sessions(c) {
  // Re-checked here, not trusted from the plan: this is the step that returns
  // message content, so it refuses any id outside a test org outright.
  const allowed = new Set(
    (
      await c.query(
        `SELECT s.id FROM scenario_sessions s WHERE s.id = ANY($1::uuid[])
            AND ${TEST_ORG_SESSION('s.id')}`,
        [params.ids],
      )
    ).rows.map((r) => r.id),
  );
  const out = [];
  for (const id of params.ids) {
    if (!allowed.has(id)) throw new Error(`session ${id} is not in a test organization`);
    if (params.kind === 'drift') {
      const labels = (
        await c.query(
          `SELECT "turnIndex" AS turn_index, coherence, "topicLabel" AS topic_label,
                  "inCharacter" AS in_character,
                  "counselorUtteranceGarbled" AS counselor_utterance_garbled,
                  "aiReplyFailureMode" AS ai_reply_failure_mode,
                  "roleInversion" AS role_inversion, "offeredSolution" AS offered_solution,
                  "introducedNewInformation" AS introduced_new_information,
                  "resistanceBriefed" AS resistance_briefed
             FROM turn_drift_judgment
            WHERE "scenarioSessionId" = $1 AND "judgeModel" = $2 AND "judgePromptVersion" = $3
            ORDER BY "turnIndex"`,
          [id, params.judgeModel, params.judgePromptVersion],
        )
      ).rows;
      out.push({ id, transcript: await driftTranscript(c, id), labels });
    } else {
      const claims = (
        await c.query(
          `SELECT "claimKind" AS kind, "claimIndex" AS claim_index, "claimText" AS text,
                  verdict, "quotesTranscript" AS quotes_transcript,
                  "quoteIsAccurate" AS quote_is_accurate
             FROM feedback_claim_judgment
            WHERE "scenarioSessionId" = $1 AND "judgeModel" = $2 AND "judgePromptVersion" = $3
            ORDER BY "claimKind", "claimIndex"`,
          [id, params.judgeModel, params.judgePromptVersion],
        )
      ).rows;
      out.push({ id, transcript: await groundednessTranscript(c, id), claims });
    }
  }
  return out;
}

(async () => {
  const c = new Client({
    host: process.env.DB_HOST,
    port: Number(process.env.DB_PORT || 5432),
    database: process.env.DB_DATABASE,
    user: process.env.DB_USERNAME,
    password: process.env.DB_PASSWORD,
    ssl: { rejectUnauthorized: false },
    statement_timeout: 60000,
  });
  await c.connect();
  await c.query('SET default_transaction_read_only = on');
  const fn = { plan, personas, sessions }[params.mode];
  const data = await fn(c);
  await c.end();
  process.stdout.write('BAKEOFF_B64:' + zlib.gzipSync(JSON.stringify(data)).toString('base64') + ':END\n');
})().catch((e) => {
  console.error('ERR', e.message);
  process.exit(1);
});
