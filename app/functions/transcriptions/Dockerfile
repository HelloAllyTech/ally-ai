# Stage 1: Use Debian to download static ffmpeg
# Reason: public.ecr.aws/lambda/python:3.12 isn't Debian- or Amazon Linux-based in the traditional sense. 
# It’s a minimal AWS Lambda runtime image that doesn’t include common package managers like yum or apt.
# So we need to download static ffmpeg and copy it to the final image.
# Solution: Use a full-featured base (e.g., debian or python:3.12-slim) to download static FFmpeg.
# Copy just the FFmpeg binary into the final Lambda image.
FROM python:3.12-slim AS ffmpeg-downloader

RUN apt-get update && apt-get install -y curl xz-utils && rm -rf /var/lib/apt/lists/*

# Download static FFmpeg
RUN curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar -xJ && \
    cp ffmpeg-*-static/ffmpeg /ffmpeg && \
    chmod +x /ffmpeg

# Stage 2: AWS Lambda base image
FROM public.ecr.aws/lambda/python:3.12

# Copy static ffmpeg from previous stage
COPY --from=ffmpeg-downloader /ffmpeg /usr/bin/ffmpeg

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY lambda_handler.py .
COPY core/ ./core/
COPY utils/ ./utils/
COPY services/ ./services/

CMD ["lambda_handler.lambda_handler"]
