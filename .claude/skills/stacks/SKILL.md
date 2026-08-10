---
name: stacks
description: Search the Stacks library for product guidance — how a feature should behave, not how to code it. Use before writing an implementation plan, and again mid-implementation whenever a product judgement comes up: defining an empty, loading, edge or failure state; naming a user-facing label, button or error message; deciding what a view shows and what it omits; picking a threshold, limit, cadence or reward rule; or reviewing a change for its behaviour. Skip for mechanical work — rename, dependency bump, typo.
---

# Stacks

A retrieval service holding product-management and engineering-practice guidance distilled from
books the team has vetted. Reached over MCP as the `stacks` server, declared in this repo's
committed `.mcp.json`. Tools: `search_chunks`, `get_chunks`, `list_tags`, `list_documents`.

## When

Any point a decision about **how the product should behave** is being made — not only planning.
The moments that most often get invented instead of retrieved:

| Moment | Query the topic, e.g. |
|---|---|
| Writing an implementation plan | 2–3 queries across the task's distinct aspects |
| An empty, loading, edge or failure state | `empty state design`, `progressive disclosure` |
| Naming a label, button or error message | `error message tone`, `microcopy clarity` |
| Deciding what a view shows vs. omits | `dashboard information density`, `sample size honesty` |
| A threshold, limit, cadence or reward rule | `notification frequency`, `extrinsic rewards` |
| Reviewing a change for behaviour | the behaviour in question, not the diff |

## How

**Use 2–3 queries, not one.** One query retrieves a single neighbourhood of the corpus. Split the
task along its actual topics so the queries land in different places.

**Query the topic, not the ticket title.** `"scenario_voices table migration"` retrieves nothing;
`"database migration rollout strategy"` retrieves the guidance that matters.

**Read the results before you decide, not after.** The point is that returned guidance changes
the outcome — which slice ships first, which states the UI needs, what the non-goals are.

`get_chunks` fetches the verbatim source excerpt behind a chunk when the exact wording matters.
`list_tags` and `list_documents` show coverage when you are not sure the corpus holds anything.

## Citing

Name the **chunk title** inline where the guidance shaped a decision, so a reviewer can trace it:

```markdown
3. Ship the streak counter read-only in v1; no notifications until we have retention data.
   (Stacks: *Gamification — Extrinsic Rewards and Motivation Crowding*)
```

Cite only chunks you actually applied. If the queries return nothing relevant, say so in one line
— "Stacks searched (`x`, `y`, `z`) — nothing applicable" — rather than skipping silently. A silent
skip and a genuine miss look identical to a reviewer.

## When a query comes back empty

Stacks **replaced** the wiki's Product Management Best Practices (deprecated 2026-08-07): nothing
there is a gate, and Stacks wins on conflict. But those pages still record Ally-specific traps a
general corpus has no reason to hold — Carbon's chart-overflow behaviour, the `roles[]`-vs-
collapsed-`role` gating trap, minimum group size for tenant-isolated metrics. Check them when a
query comes back empty on something Ally-specific.

Full rule, setup and troubleshooting:
[Planning with Stacks](https://tech.helloally.ai/#/wiki/contributing/planning-with-stacks.md).
