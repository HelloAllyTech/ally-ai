#!/usr/bin/env bash
set -euo pipefail

# Default credentials and endpoints for LocalStack
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-southeast-1}"
export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://localstack:4566}"
export LOCALSTACK_HOST="${LOCALSTACK_HOST:-localstack}"

QUEUE_RESULTS_URL="${QUEUE__TRANSCRIPTION_RESULTS_QUEUE_URL:-}"
QUEUE_RESPONSE_URL="${QUEUE__TRANSCRIBE_AND_SUMMARIZE_RESPONSE_QUEUE_URL:-}"
RESULTS_BUCKET="${QUEUE__TRANSCRIBE_AND_SUMMARIZE_RESULTS_BUCKET:-}"

extract_queue_name() {
  local url="$1"
  local fallback="$2"
  if [[ -n "$url" ]]; then
    local name="${url##*/}"
    if [[ -n "$name" ]]; then
      echo "$name"
      return
    fi
  fi
  echo "$fallback"
}

QUEUE_RESULTS_NAME="$(extract_queue_name "$QUEUE_RESULTS_URL" "TRANSCRIPTION_RESULTS_QUEUE")"
QUEUE_RESPONSE_NAME="$(extract_queue_name "$QUEUE_RESPONSE_URL" "TRANSCRIBE_AND_SUMMARIZE_RESPONSE_QUEUE")"

wait_for_localstack() {
  local max_attempts=30
  local attempt=1
  until awslocal sqs list-queues >/dev/null 2>&1; do
    if [[ $attempt -ge $max_attempts ]]; then
      echo "LocalStack did not become ready in time" >&2
      return 1
    fi
    sleep 2
    attempt=$((attempt + 1))
  done
  return 0
}

bootstrap_queues() {
  awslocal sqs create-queue --queue-name "$QUEUE_RESULTS_NAME" >/dev/null 2>&1 || true
  awslocal sqs create-queue --queue-name "$QUEUE_RESPONSE_NAME" >/dev/null 2>&1 || true
}

bootstrap_bucket() {
  local bucket="$1"
  if [[ -z "$bucket" ]]; then
    return 0
  fi
  awslocal s3 mb "s3://$bucket" >/dev/null 2>&1 || true
}

if wait_for_localstack; then
  bootstrap_queues
  bootstrap_bucket "$RESULTS_BUCKET"
else
  echo "Skipping LocalStack resource bootstrap due to readiness check failure" >&2
fi

