# Release Guide — ally-ai

**The shared release process lives in the wiki:
[Release Process](https://tech.helloally.ai/#/wiki/contributing/release-process.md).**
Read that for semantic-versioning policy, how to trigger the workflow, the image tag
scheme, publishing the draft, and troubleshooting. This file carries only what is specific
to this service.

## This service

| | |
|---|---|
| Workflow | **Production Release** (`.github/workflows/production-release.yaml`) |
| Release branch | `master` |
| Runtime in CI | Python 3.12 |
| Package manager | Poetry (venv cached on `poetry.lock`) |
| Deployment | AWS ECS |
| Runs migrations | Weaviate schema migrations — see `app/migrations/` |

## Deployment target

- **Cluster**: `ally-prd-mb-ecs-cluster`
- **ECS service**: `ally-prd-svc-core-ai`
- **Task definition**: `ally-prd-td-core-ai`
- **Container**: `ally-prd-cntr-core-ai`

## Required repository variables

Settings → Secrets and variables → Actions → Variables:

```
PRD_AWS_ROLE          # AWS IAM role ARN for production
PRD_AWS_REGION
PRD_ECR_REPOSITORY
```

## Verify a deployment

```bash
aws ecs describe-services \
  --cluster ally-prd-mb-ecs-cluster \
  --services ally-prd-svc-core-ai

aws ecs list-tasks \
  --cluster ally-prd-mb-ecs-cluster \
  --service-name ally-prd-svc-core-ai

aws logs tail /ecs/ally-prd-cntr-core-ai --follow
```

## Notes specific to this service

- **Weaviate migrations are ordered and immutable.** Follow the `NNN-description.py`
  convention in `app/migrations/`; never renumber or edit one that has shipped.
- **A release pushes the prompt defaults itself.** The image's start command (`Dockerfile`
  `CMD`) runs `scripts/sync_prompts.py` before the API starts, so every deploy publishes
  this repo's default prompts to ally-be's prompt management — `v1.21.0` brought the new
  helpline prompts to production with no manual step. In that default command the chain is
  `&&` and the script exits 1 when `ALLY_CORE__ENDPOINT`/`ALLY_CORE__API_KEY` are unset or
  ally-be rejects the sync, so a failed sync stops the API from starting (the ECS task
  definitions live in AWS, not here — check them before relying on that). Local compose
  (`docker-compose.local.yml`) continues past a failed sync instead, which is why a local
  ally-be can hold older prompts than the code. `make sync-prompts` pushes defaults
  without a deploy.
