# ally-ai — start here

FastAPI conversational-AI service. Owns the Weaviate vector DB. Python 3.12+, Poetry.

This file is a **router**: find your task below, read what it points at, skip the rest.
Conventions with a canonical home are linked, never restated — if you find a rule written
twice anywhere in this platform, that's a bug worth fixing.

## Before you touch anything user-facing

Read [Product Best Practices](https://tech.helloally.ai/#/wiki/product/best-practices.md)
and the one subsection that matches your change. Internal work — refactors, migrations,
infra, tests — is exempt. Don't read the whole product section; the hub tells you which
subsection applies.

## What am I doing?

| Task | Read first |
|---|---|
| Adding an AI capability | Prompt in `app/prompts/`, logic in `app/core/`, endpoint in `app/api/v1/endpoints/` |
| Changing a Weaviate collection | `app/core/vector_db/constants.py`, then add `app/migrations/NNN-description.py` |
| Anything touching stored data | [`ally-be/DATA_SCHEMA.md`](https://github.com/HelloAllyTech/ally-be/blob/main/DATA_SCHEMA.md) — the cross-store map; this repo owns its vector half |
| Editing prompts | `app/prompts/` — but see the prompt-override gotcha below before assuming your edit takes effect |
| Transcription / queue work | `app/core/queue/transcription_request_sqs_worker.py` |
| Language-quality evaluation | [`docs/language-eval-judge-schema.md`](docs/language-eval-judge-schema.md) and [`docs/drift-metrics-spec.md`](docs/drift-metrics-spec.md); platform context in the [wiki](https://tech.helloally.ai/#/wiki/platform/language-quality-eval.md) |
| Releasing | [`.github/RELEASE_GUIDE.md`](.github/RELEASE_GUIDE.md) |
| Anything else | [`WIKI-ROUTING.md`](WIKI-ROUTING.md) — one line per wiki page, tells you which to fetch |

## Repo shape

- `app/core/` — service logic: `conversations/`, `summaries/`, `text_generations/`,
  `reference_documents/`, `transcriptions/`, `queue/`, `vector_db/`.
- `app/api/v1/endpoints/` — HTTP surface. Entry: `app/main.py`.
- `app/prompts/` — file-based, loaded dynamically.
- `app/migrations/` — Weaviate schema migrations, `NNN-description.py`.
- Pattern: abstract base classes per service, FastAPI DI, Pydantic settings.

## Gotchas that change what you write

- **HIPAA.** Never log PII/PHI outside the designated audit path — use `phi_logger`.
- **Your prompt edit may not be what runs.** Prompts here are defaults; ally-be's prompt
  management can override them at runtime. `scripts/sync_prompts.py` pushes defaults up.
  Check whether an override exists before debugging a prompt that "isn't applying".
- **Weaviate migrations are ordered and immutable.** Follow `NNN-description.py`; never
  renumber or edit a merged one.
- **SQS in dev is LocalStack**, started by docker-compose. Queue URLs differ from production.
- **Service-to-service auth is `X-API-Key`**, not the client JWT.
- **Python 3.12+.** Not the system Python on most machines.

## Commands

```bash
make install
make test
make migrate          # Weaviate schema migrations
make sync-prompts     # push default prompts to ally-be
```

## When your change outdates a doc

[`.docs-map.yml`](.docs-map.yml) declares which docs cover which code, and CI enforces it.
Changing `app/core/vector_db/constants.py` or `app/migrations/**` requires a wiki update —
either open one, or apply the `docs:skip` label with a reason.

Wiki edits do **not** need a hand-rolled second PR:

```bash
git clone --depth=1 https://github.com/helloallytech/helloallytech.github.io .wiki-tmp
# edit .wiki-tmp/wiki/**
.wiki-tmp/scripts/wiki-pr.sh "<url of this PR>"     # prints the Wiki-PR: trailer to paste
```

`.wiki-tmp/` is gitignored. The wiki PR merges when this one does.

## Canonical docs

The [Ally Developer Wiki](https://tech.helloally.ai) is the source of truth for platform
architecture, SDLC rules, and product practice —
[this repo's page](https://tech.helloally.ai/#/wiki/repos/ally-ai.md) ·
[architecture](https://tech.helloally.ai/#/wiki/platform/architecture.md) ·
[contributing](https://tech.helloally.ai/#/wiki/contributing/guide.md) ·
[how the docs system works](https://tech.helloally.ai/#/wiki/contributing/docs-system.md).

> ⚠️ The wiki is **public**. Never add secrets, credentials, IP addresses, internal
> hostnames/domains, or cloud region details to it.
