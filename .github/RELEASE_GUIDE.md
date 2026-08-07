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
- **Prompt defaults are pushed separately.** `make sync-prompts` publishes this repo's
  default prompts to ally-be's prompt management; a release does not do it for you.
