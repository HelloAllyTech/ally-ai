#!/usr/bin/env bash
set -euo pipefail

# Install awscli-local if not already installed (needed for LocalStack)
if ! command -v awslocal &> /dev/null; then
  pip install --no-cache-dir awscli-local awscli >/dev/null 2>&1
  # Ensure awslocal is in PATH (pip installs to /usr/local/bin)
  export PATH="/usr/local/bin:$PATH"
fi

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

# Docker Compose ensures LocalStack is healthy before starting this container,
# so we can proceed directly with bootstrap operations.
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

bootstrap_queues
bootstrap_bucket "$RESULTS_BUCKET"

