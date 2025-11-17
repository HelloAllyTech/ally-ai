#!/usr/bin/env bash
set -euo pipefail

# Install awscli-local if not already installed (needed for LocalStack)
if ! command -v awslocal &> /dev/null; then
  pip install --no-cache-dir awscli-local awscli >/dev/null 2>&1
  # Ensure awslocal is in PATH (pip installs to /usr/local/bin)
  export PATH="/usr/local/bin:$PATH"
fi

# Default credentials and endpoints for LocalStack
# Uses LOCALSTACK_HOST from environment (set by docker-compose) or defaults to localhost
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-southeast-1}"

# Use LOCALSTACK_HOST if set (from docker-compose), otherwise default to localhost
if [ -n "${LOCALSTACK_HOST:-}" ]; then
  export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://${LOCALSTACK_HOST}:4566}"
else
  export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://localhost:4566}"
fi

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

# Construct correct LocalStack queue URLs
LOCALSTACK_QUEUE_RESULTS_URL="${AWS_ENDPOINT_URL}/000000000000/${QUEUE_RESULTS_NAME}"
LOCALSTACK_QUEUE_RESPONSE_URL="${AWS_ENDPOINT_URL}/000000000000/${QUEUE_RESPONSE_NAME}"

# Bootstrap operations - shared LocalStack should be ready (waited for by localstack-wait service)
bootstrap_queues() {
  echo "Creating SQS queues..."
  
  # Helper function to check if queue exists and create if needed
  ensure_queue_exists() {
    local queue_name="$1"
    local queue_url="${AWS_ENDPOINT_URL}/000000000000/${queue_name}"
    
    # Check if queue already exists
    if awslocal sqs get-queue-attributes --queue-url "$queue_url" --attribute-names QueueArn >/dev/null 2>&1; then
      echo "✓ Queue $queue_name already exists"
      echo "$queue_url"
      return 0
    fi
    
    # Queue doesn't exist, create it
    echo "Creating queue: $queue_name..."
    if awslocal sqs create-queue --queue-name "$queue_name" >/dev/null 2>&1; then
      echo "✓ Queue $queue_name created successfully"
      echo "$queue_url"
      return 0
    else
      echo "✗ Failed to create queue $queue_name"
      return 1
    fi
  }
  
  # Ensure queues exist
  ACTUAL_RESULTS_URL=$(ensure_queue_exists "$QUEUE_RESULTS_NAME")
  ACTUAL_RESPONSE_URL=$(ensure_queue_exists "$QUEUE_RESPONSE_NAME")
  
  # Verify both queues exist
  if [[ -z "$ACTUAL_RESULTS_URL" ]] || [[ -z "$ACTUAL_RESPONSE_URL" ]]; then
    echo "✗ Error: Failed to ensure queues exist"
    exit 1
  fi
  
  # Export queue URLs as environment variables for the application to use
  # These will be available to processes started after this script
  export QUEUE__TRANSCRIPTION_RESULTS_QUEUE_URL="$ACTUAL_RESULTS_URL"
  export QUEUE__TRANSCRIBE_AND_SUMMARIZE_RESPONSE_QUEUE_URL="$ACTUAL_RESPONSE_URL"
  
  echo ""
  echo "Queues ready:"
  echo "  - $QUEUE_RESULTS_NAME -> $ACTUAL_RESULTS_URL"
  echo "  - $QUEUE_RESPONSE_NAME -> $ACTUAL_RESPONSE_URL"
  echo ""
  echo "Exported environment variables:"
  echo "  QUEUE__TRANSCRIPTION_RESULTS_QUEUE_URL=\"$ACTUAL_RESULTS_URL\""
  echo "  QUEUE__TRANSCRIBE_AND_SUMMARIZE_RESPONSE_QUEUE_URL=\"$ACTUAL_RESPONSE_URL\""
  echo ""
  
  # Small delay to ensure queues are fully ready
  sleep 1
}

bootstrap_bucket() {
  local bucket="$1"
  if [[ -z "$bucket" ]]; then
    return 0
  fi
  
  # Check if bucket already exists
  if awslocal s3 ls "s3://$bucket" >/dev/null 2>&1; then
    echo "✓ S3 bucket $bucket already exists"
    return 0
  fi
  
  # Bucket doesn't exist, create it
  echo "Creating S3 bucket: $bucket..."
  if awslocal s3 mb "s3://$bucket" >/dev/null 2>&1; then
    echo "✓ S3 bucket $bucket created successfully"
    return 0
  else
    echo "✗ Warning: Failed to create S3 bucket $bucket"
    return 1
  fi
}

bootstrap_queues
bootstrap_bucket "$RESULTS_BUCKET"

