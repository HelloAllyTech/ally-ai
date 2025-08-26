# Lambda Transcription Function

## Manual Deployment

### Step 1: Build and Push Docker Image

```bash
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -t 135808949334.dkr.ecr.ap-southeast-1.amazonaws.com/ally-dev-ai-transcription-lambda:latest \
  --push \
  app/functions/transcriptions/
```

### Step 2: Deploy to AWS Lambda

1. Go to AWS Lambda Console
2. Navigate to your transcription Lambda function
3. Click on "Deploy" tab
4. Select "Container image"
5. Choose the newly pushed ECR image
6. Click "Save"

## Function Overview

This Lambda function:
- Processes SQS messages from the transcription requests queue
- Downloads and converts audio files using FFmpeg
- Performs transcription using OpenAI/Deepgram
- Sends results to the transcription results queue
- Deletes processed messages from the requests queue

## Configuration

### Environment Variables Required:
- `OPENAI_API_KEY`
- `OPENAI_ORGANIZATION_ID`
- `DEEPGRAM_API_KEY`
- `TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL`
- `TRANSCRIPTION_RESULTS_QUEUE_URL`
- `TRANSCRIPTION_PROVIDER` (default: "openai")


